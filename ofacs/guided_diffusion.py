"""DiffGDA-inspired guided diffusion for V5 OFA-CS.

This keeps the senior DDPM-style q_sample/noise-prediction semantics, but adds a
label-free structure-semantic condition branch.  It is not V4 identity
correction, and it does not train on unseen graphs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .errors import V5ProtocolError, V5DataLeakageError


@dataclass(frozen=True)
class GuidedDiffusionConfig:
    latent_dim: int
    base_cond_dim: int
    structure_cond_dim: int = 16
    diffusion_T: int = 200
    time_emb_dim: int = 128
    hidden_dim: int = 256
    guidance_eta: float = 0.50
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def validate(self) -> "GuidedDiffusionConfig":
        for name in ("latent_dim", "base_cond_dim", "structure_cond_dim", "diffusion_T", "time_emb_dim", "hidden_dim"):
            if int(getattr(self, name)) <= 0:
                raise V5ProtocolError(f"{name} must be positive")
        if not (0.0 <= float(self.guidance_eta) <= 10.0):
            raise V5ProtocolError("guidance_eta must be in [0,10]")
        if not (0.0 < float(self.beta_start) < float(self.beta_end) < 1.0):
            raise V5ProtocolError("require 0 < beta_start < beta_end < 1")
        return self


@dataclass(frozen=True)
class GuidedDiffusionLossReport:
    eps_loss: float
    correction_norm: float
    pre_post_cosine: float
    t_min: int
    t_max: int
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_used: bool = False
    unseen_training_used: bool = False

    def __post_init__(self) -> None:
        if self.labels_used or self.ground_truth_used or self.metrics_used or self.unseen_training_used:
            raise V5DataLeakageError("guided diffusion report indicates forbidden supervision")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(1, half))
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class GuidedEpsPredictor(nn.Module):
    """Base denoising prior plus structure-semantic guidance branch."""

    def __init__(self, config: GuidedDiffusionConfig):
        super().__init__()
        cfg = config.validate()
        self.config = cfg
        self.time_emb = SinusoidalPosEmb(cfg.time_emb_dim)
        self.time_proj = nn.Linear(cfg.time_emb_dim, cfg.hidden_dim)
        self.base_cond_proj = nn.Linear(cfg.base_cond_dim, cfg.hidden_dim)
        self.struct_cond_proj = nn.Linear(cfg.structure_cond_dim, cfg.hidden_dim)
        self.base_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.guidance_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, base_cond: torch.Tensor, structure_cond: torch.Tensor | None = None) -> torch.Tensor:
        if z_t.ndim != 2:
            raise V5ProtocolError("z_t must be [B,D]")
        if base_cond.ndim != 2:
            raise V5ProtocolError("base_cond must be [B,C]")
        if z_t.shape[0] != base_cond.shape[0]:
            raise V5ProtocolError("z_t and base_cond batch mismatch")
        if structure_cond is None:
            structure_cond = torch.zeros(z_t.shape[0], self.config.structure_cond_dim, device=z_t.device, dtype=z_t.dtype)
        if structure_cond.ndim == 1:
            structure_cond = structure_cond.unsqueeze(0).expand(z_t.shape[0], -1)
        if structure_cond.shape[0] != z_t.shape[0]:
            raise V5ProtocolError("structure_cond batch mismatch")
        if structure_cond.shape[1] != self.config.structure_cond_dim:
            raise V5ProtocolError("structure_cond dim mismatch")

        te = self.time_proj(self.time_emb(t))
        base_ctx = torch.tanh(te + self.base_cond_proj(base_cond))
        struct_ctx = torch.tanh(te + self.struct_cond_proj(structure_cond.to(dtype=z_t.dtype, device=z_t.device)))
        base_eps = self.base_net(torch.cat([z_t, base_ctx], dim=-1))
        guidance_eps = self.guidance_net(torch.cat([z_t, struct_ctx], dim=-1))
        return base_eps + float(self.config.guidance_eta) * guidance_eps


class GuidedDiffusionRefiner(nn.Module):
    """DDPM-style representation refiner with label-free guidance condition."""

    def __init__(self, config: GuidedDiffusionConfig):
        super().__init__()
        cfg = config.validate()
        self.config = cfg
        self.eps_model = GuidedEpsPredictor(cfg)
        betas = torch.linspace(float(cfg.beta_start), float(cfg.beta_end), int(cfg.diffusion_T))
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

    @property
    def T(self) -> int:
        return int(self.config.diffusion_T)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if z0.ndim != 2:
            raise V5ProtocolError("z0 must be [B,D]")
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(-1).to(dtype=z0.dtype, device=z0.device)
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).to(dtype=z0.dtype, device=z0.device)
        return sqrt_acp * z0 + sqrt_1m * noise, noise

    def training_loss(
        self,
        z0: torch.Tensor,
        *,
        base_cond: torch.Tensor,
        structure_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GuidedDiffusionLossReport]:
        """Noise prediction loss on source representations only."""

        b = int(z0.shape[0])
        t = torch.randint(0, self.T, (b,), device=z0.device, dtype=torch.long)
        z_t, noise = self.q_sample(z0, t)
        pred = self.eps_model(z_t, t, base_cond, structure_cond)
        loss = F.mse_loss(pred, noise)
        # one-step x0 estimate for diagnostics only.
        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(-1).to(dtype=z0.dtype, device=z0.device)
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).to(dtype=z0.dtype, device=z0.device)
        z0_hat = (z_t - sqrt_1m * pred) / torch.clamp(sqrt_acp, min=1e-12)
        report = GuidedDiffusionLossReport(
            eps_loss=float(loss.detach().cpu().item()),
            correction_norm=float((z0_hat - z0).norm(dim=-1).mean().detach().cpu().item()),
            pre_post_cosine=float(F.cosine_similarity(z0_hat, z0, dim=-1).mean().detach().cpu().item()),
            t_min=int(t.min().detach().cpu().item()),
            t_max=int(t.max().detach().cpu().item()),
        )
        return loss, report

    def reverse_denoise(
        self,
        z0: torch.Tensor,
        *,
        base_cond: torch.Tensor,
        structure_cond: torch.Tensor | None = None,
        t_eval: int | None = None,
        noise: torch.Tensor | None = None,
        stochastic: bool = False,
        ratio_guidance: torch.nn.Module | None = None,
        ratio_guidance_weight: float = 0.0,
        ratio_guidance_clip: float = 5.0,
        adaptive_ratio_guidance: bool = False,
        ratio_guidance_confidence_scale: float = 1.0,
        ratio_guidance_min_mult: float = 0.25,
        ratio_guidance_max_mult: float = 1.25,
        guidance_diagnostics: object | None = None,
        guidance_runtime_timer: object | None = None,
        guidance_branch: str = "unknown",
    ) -> torch.Tensor:
        """Reverse denoise, optionally with target-domain evidence guidance.

        Without ``ratio_guidance`` this is the previous deterministic DDPM-style
        path.  With ``ratio_guidance`` it implements the representation-space
        analogue of DiffGDA's target-guided reverse process.  The legacy path
        accepts ``grad_z log q_T(z_t)/p_S(z_t)`` from a learned domain
        classifier.  The strict one-for-all path instead accepts
        ``grad_z log e_T(z_t)`` from a closed-form, parameter-free target
        evidence potential fitted at inference.

        The guidance module never receives target labels or target ground truth.
        It must expose either ``log_target_evidence(z, t)`` or the legacy
        ``log_density_ratio(z, t)`` and return one scalar potential per node.
        """

        if t_eval is None:
            t_eval = self.T - 1
        t_eval = max(0, min(int(t_eval), self.T - 1))
        b = int(z0.shape[0])
        t = torch.full((b,), t_eval, device=z0.device, dtype=torch.long)
        if noise is None:
            noise = torch.zeros_like(z0)
        z, _ = self.q_sample(z0, t, noise=noise)
        use_ratio = ratio_guidance is not None and float(ratio_guidance_weight) != 0.0

        for step in reversed(range(0, t_eval + 1)):
            t_step = torch.full((b,), step, device=z0.device, dtype=torch.long)
            beta_t = self.betas[step].to(dtype=z0.dtype, device=z0.device)
            alpha_t = self.alphas[step].to(dtype=z0.dtype, device=z0.device)
            alpha_cum_t = self.alphas_cumprod[step].to(dtype=z0.dtype, device=z0.device)
            if use_ratio:
                # Keep the senior/source denoiser fixed for inference; enable
                # gradients only for the target/source density-ratio potential.
                with torch.no_grad():
                    eps_pred = self.eps_model(z, t_step, base_cond, structure_cond)
                if guidance_runtime_timer is not None:
                    guidance_runtime_timer.start(z.device)
                with torch.enable_grad():
                    z_req = z.detach().requires_grad_(True)
                    if hasattr(ratio_guidance, "log_target_evidence"):
                        log_ratio = ratio_guidance.log_target_evidence(z_req, t_step)
                    elif hasattr(ratio_guidance, "log_density_ratio"):
                        log_ratio = ratio_guidance.log_density_ratio(z_req, t_step)
                    else:
                        raise V5ProtocolError(
                            "guidance module must expose log_target_evidence or log_density_ratio"
                        )
                    grad = torch.autograd.grad(
                        log_ratio.sum(), z_req, retain_graph=False, create_graph=False
                    )[0]
                grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                raw_grad_norm = grad.norm(dim=-1)
                eps_norm = eps_pred.detach().norm(dim=-1)
                scale_mode = str(getattr(ratio_guidance, "guidance_scale_mode", "legacy_raw"))
                clip = float(ratio_guidance_clip)
                if scale_mode == "relative_eps":
                    # The legacy update multiplied a tiny raw classifier gradient
                    # by a small scalar, so it was numerically invisible.  V2
                    # preserves the density-ratio direction but calibrates its
                    # magnitude against the denoiser prediction at this step.
                    direction = grad / torch.clamp(raw_grad_norm.unsqueeze(-1), min=1e-12)
                    if hasattr(ratio_guidance, "guidance_confidence"):
                        confidence = ratio_guidance.guidance_confidence(
                            z.detach(), t_step
                        ).detach().view(-1, 1)
                        confidence = torch.nan_to_num(
                            confidence, nan=0.0, posinf=1.0, neginf=0.0
                        ).clamp(0.0, 1.0)
                    else:
                        conf_scale = max(
                            float(
                                getattr(
                                    ratio_guidance,
                                    "guidance_confidence_scale",
                                    1.0,
                                )
                            ),
                            1e-6,
                        )
                        confidence = torch.tanh(
                            torch.abs(log_ratio.detach()).view(-1, 1) / conf_scale
                        )
                    relative_fraction = float(ratio_guidance_weight) * confidence
                    max_relative = max(0.0, float(getattr(ratio_guidance, "max_relative_update", 0.10)))
                    relative_fraction = torch.clamp(relative_fraction, min=0.0, max=max_relative)
                    guidance_update = direction.to(dtype=eps_pred.dtype) * eps_norm.view(-1, 1).to(dtype=eps_pred.dtype) * relative_fraction.to(dtype=eps_pred.dtype)
                    clipped_grad = direction
                elif scale_mode == "legacy_raw":
                    clipped_grad = grad
                    if clip > 0:
                        grad_norm = torch.clamp(clipped_grad.norm(dim=-1, keepdim=True), min=1e-12)
                        clipped_grad = clipped_grad * torch.clamp(
                            torch.tensor(clip, device=clipped_grad.device, dtype=clipped_grad.dtype) / grad_norm,
                            max=1.0,
                        )
                    sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_cum_t, min=1e-12))
                    guidance_weight = torch.full((b, 1), float(ratio_guidance_weight), device=z.device, dtype=z.dtype)
                    if bool(adaptive_ratio_guidance):
                        scale = max(float(ratio_guidance_confidence_scale), 1e-6)
                        conf = torch.sigmoid(torch.abs(log_ratio.detach()).view(-1, 1) / scale - 0.5)
                        min_m = float(ratio_guidance_min_mult)
                        max_m = float(ratio_guidance_max_mult)
                        if max_m < min_m:
                            min_m, max_m = max_m, min_m
                        guidance_weight = guidance_weight * (min_m + (max_m - min_m) * conf.to(dtype=z.dtype))
                    guidance_update = guidance_weight * sigma_t * clipped_grad.to(dtype=eps_pred.dtype)
                else:
                    raise V5ProtocolError(f"unsupported density-ratio guidance scale mode: {scale_mode!r}")
                if bool(adaptive_ratio_guidance) and scale_mode == "relative_eps":
                    # Node-adaptive DiffGDA guidance.  Large |log q_T/p_S| means
                    # the domain-ratio network is confident that the node lies in
                    # a target/source-discriminative region; small confidence keeps
                    # the update close to the source denoiser to avoid over-shifting
                    # Cora/Citeseer-like fine-grained rankings.
                    scale = max(float(ratio_guidance_confidence_scale), 1e-6)
                    conf = torch.sigmoid(torch.abs(log_ratio.detach()).view(-1, 1) / scale - 0.5)
                    min_m = float(ratio_guidance_min_mult)
                    max_m = float(ratio_guidance_max_mult)
                    if max_m < min_m:
                        min_m, max_m = max_m, min_m
                    adaptive_mult = min_m + (max_m - min_m) * conf.to(dtype=z.dtype)
                    max_relative = max(0.0, float(getattr(ratio_guidance, "max_relative_update", 0.10)))
                    rel = guidance_update.norm(dim=-1, keepdim=True) / torch.clamp(eps_norm.view(-1, 1), min=1e-12)
                    capped = torch.clamp(rel * adaptive_mult, max=max_relative)
                    guidance_update = guidance_update / torch.clamp(guidance_update.norm(dim=-1, keepdim=True), min=1e-12)
                    guidance_update = guidance_update * eps_norm.view(-1, 1) * capped
                if guidance_diagnostics is not None:
                    guidance_diagnostics.record(
                        branch=str(guidance_branch),
                        step=int(step),
                        log_ratio=log_ratio.detach(),
                        raw_gradient_norm=raw_grad_norm,
                        clipped_gradient_norm=clipped_grad.norm(dim=-1),
                        guidance_update_norm=guidance_update.norm(dim=-1),
                        eps_prediction_norm=eps_norm,
                        scale_mode=scale_mode,
                        clip=float(ratio_guidance_clip),
                    )
                eps_pred = eps_pred.detach() - guidance_update
                if guidance_runtime_timer is not None:
                    guidance_runtime_timer.stop(z.device)
            else:
                with torch.no_grad():
                    eps_pred = self.eps_model(z, t_step, base_cond, structure_cond)
            mu = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(torch.clamp(1.0 - alpha_cum_t, min=1e-12))) * eps_pred)
            if step > 0:
                alpha_cum_prev = self.alphas_cumprod[step - 1].to(dtype=z0.dtype, device=z0.device)
                tilde_beta = ((1.0 - alpha_cum_prev) / torch.clamp(1.0 - alpha_cum_t, min=1e-12)) * beta_t
                if stochastic:
                    z = mu + torch.sqrt(torch.clamp(tilde_beta, min=0.0)) * torch.randn_like(z)
                else:
                    z = mu
            else:
                z = mu
            z = z.detach()
        return z
