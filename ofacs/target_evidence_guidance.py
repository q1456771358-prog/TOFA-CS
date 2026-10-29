"""Parameter-free target evidence for strict one-for-all reverse diffusion.

The learnable encoder and denoiser are trained on source graphs only.  At fixed
target inference, this module estimates diagonal first/second moments from the
unlabeled target representations and exposes a differentiable evidence
potential to the reverse diffusion process.  It creates no optimizer, owns no
trainable parameter, and never reads target labels, communities, or metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .errors import V5ProtocolError


@dataclass(frozen=True)
class TargetEvidenceFitReport:
    protocol: str
    num_nodes: int
    latent_dim: int
    fit_chunk_size: int
    variance_floor: float
    variance_shrinkage: float
    node_mean_norm: float
    neighbor_mean_norm: float
    node_variance_mean: float
    neighbor_variance_mean: float
    trainable_parameter_count: int = 0
    target_optimizer_steps: int = 0
    target_labels_used: bool = False
    target_gt_used: bool = False
    target_metrics_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DiagonalGaussianTargetEvidence(nn.Module):
    """Closed-form target evidence potential at every diffusion timestep."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        variance: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        confidence_scale: float = 2.0,
        max_relative_update: float = 0.05,
    ):
        super().__init__()
        mean = mean.detach().float().reshape(-1)
        variance = variance.detach().float().reshape(-1)
        alphas_cumprod = alphas_cumprod.detach().float().reshape(-1)
        if mean.numel() == 0 or variance.shape != mean.shape:
            raise V5ProtocolError("target evidence mean/variance shape mismatch")
        if alphas_cumprod.numel() == 0:
            raise V5ProtocolError("target evidence requires a non-empty diffusion schedule")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise V5ProtocolError("target evidence moments must be finite")
        if bool((variance <= 0).any()):
            raise V5ProtocolError("target evidence variance must be positive")
        self.register_buffer("mean", mean)
        self.register_buffer("variance", variance)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.latent_dim = int(mean.numel())
        self.confidence_scale = float(confidence_scale)
        self.max_relative_update = float(max_relative_update)
        # The reverse process normalizes the evidence gradient direction and
        # scales it relative to the denoiser prediction.
        self.guidance_scale_mode = "relative_eps"
        self.guidance_kind = "target_diagonal_gaussian"

    def _noisy_moments(self, t: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if t.ndim == 0:
            t = t.reshape(1)
        alpha_bar = self.alphas_cumprod.to(device=device, dtype=dtype)[t.long()].reshape(-1, 1)
        mean_t = torch.sqrt(torch.clamp(alpha_bar, min=0.0)) * self.mean.to(device=device, dtype=dtype).reshape(1, -1)
        variance_t = (
            alpha_bar * self.variance.to(device=device, dtype=dtype).reshape(1, -1)
            + (1.0 - alpha_bar)
        ).clamp_min(1e-8)
        return mean_t, variance_t

    def log_target_evidence(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return mean per-dimension log evidence under the target moments."""

        if z.ndim != 2 or int(z.shape[1]) != self.latent_dim:
            raise V5ProtocolError(
                f"target evidence expects [B,{self.latent_dim}], got {tuple(z.shape)}"
            )
        if t.ndim == 0:
            t = t.expand(z.shape[0])
        if t.ndim != 1 or int(t.shape[0]) != int(z.shape[0]):
            raise V5ProtocolError("target evidence timestep must be [B]")
        mean_t, variance_t = self._noisy_moments(t, dtype=z.dtype, device=z.device)
        standardized = (z - mean_t).pow(2) / variance_t
        return -0.5 * (standardized + torch.log(variance_t)).mean(dim=-1)

    def guidance_confidence(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Increase guidance only for states outside typical target support."""

        mean_t, variance_t = self._noisy_moments(t, dtype=z.dtype, device=z.device)
        energy = ((z - mean_t).pow(2) / variance_t).mean(dim=-1)
        scale = max(float(self.confidence_scale), 1e-6)
        return torch.sigmoid((energy - 1.0) * scale)


class TargetMomentGuidancePair(nn.Module):
    """Node/neighbor target evidence pair with no trainable parameters."""

    def __init__(
        self,
        *,
        node_mean: torch.Tensor,
        node_variance: torch.Tensor,
        neighbor_mean: torch.Tensor,
        neighbor_variance: torch.Tensor,
        node_alphas_cumprod: torch.Tensor,
        neighbor_alphas_cumprod: torch.Tensor,
        confidence_scale: float,
        max_relative_update: float,
    ):
        super().__init__()
        self.node = DiagonalGaussianTargetEvidence(
            mean=node_mean,
            variance=node_variance,
            alphas_cumprod=node_alphas_cumprod,
            confidence_scale=confidence_scale,
            max_relative_update=max_relative_update,
        )
        self.neighbor = DiagonalGaussianTargetEvidence(
            mean=neighbor_mean,
            variance=neighbor_variance,
            alphas_cumprod=neighbor_alphas_cumprod,
            confidence_scale=confidence_scale,
            max_relative_update=max_relative_update,
        )
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise V5ProtocolError("target moment guidance must not own trainable parameters")


class _RunningDiagonalMoments:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.sum = torch.zeros(self.dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(self.dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        x = values.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 2 or int(x.shape[1]) != self.dim:
            raise V5ProtocolError("running target moment shape mismatch")
        self.count += int(x.shape[0])
        self.sum += x.sum(dim=0)
        self.sum_sq += x.pow(2).sum(dim=0)

    def finalize(self, *, variance_floor: float, variance_shrinkage: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0:
            raise V5ProtocolError("cannot fit target evidence from zero nodes")
        mean = self.sum / float(self.count)
        variance = self.sum_sq / float(self.count) - mean.pow(2)
        variance = variance.clamp_min(float(variance_floor))
        shrink = min(max(float(variance_shrinkage), 0.0), 1.0)
        if shrink > 0.0:
            scalar = variance.mean()
            variance = (1.0 - shrink) * variance + shrink * scalar
        return mean.float(), variance.float().clamp_min(float(variance_floor))


@torch.no_grad()
def fit_target_moment_guidance_pair(
    *,
    model: nn.Module,
    global_token_seqs: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8192,
    variance_floor: float = 1e-4,
    variance_shrinkage: float = 0.05,
    confidence_scale: float = 2.0,
    max_relative_update: float = 0.05,
) -> tuple[TargetMomentGuidancePair, TargetEvidenceFitReport]:
    """Fit target moments once from raw unlabeled target representations."""

    if int(chunk_size) <= 0:
        raise V5ProtocolError("target evidence chunk_size must be positive")
    if global_token_seqs.ndim != 3:
        raise V5ProtocolError("target evidence tokens must be [N,hops+1,D]")
    if not (0.0 <= float(variance_shrinkage) <= 1.0):
        raise V5ProtocolError("target evidence variance_shrinkage must be in [0,1]")
    if float(variance_floor) <= 0.0:
        raise V5ProtocolError("target evidence variance_floor must be positive")

    was_training = bool(model.training)
    model.eval()
    hidden_dim = int(getattr(model, "hidden_dim"))
    node_moments = _RunningDiagonalMoments(hidden_dim)
    neighbor_moments = _RunningDiagonalMoments(hidden_dim)
    n = int(global_token_seqs.shape[0])
    try:
        for start in range(0, n, int(chunk_size)):
            end = min(n, start + int(chunk_size))
            node, neighbor, _ = model.encode_raw(global_token_seqs[start:end].to(device))
            node_moments.update(node)
            neighbor_moments.update(neighbor)
    finally:
        if was_training:
            model.train()

    node_mean, node_variance = node_moments.finalize(
        variance_floor=variance_floor,
        variance_shrinkage=variance_shrinkage,
    )
    neighbor_mean, neighbor_variance = neighbor_moments.finalize(
        variance_floor=variance_floor,
        variance_shrinkage=variance_shrinkage,
    )
    pair = TargetMomentGuidancePair(
        node_mean=node_mean,
        node_variance=node_variance,
        neighbor_mean=neighbor_mean,
        neighbor_variance=neighbor_variance,
        node_alphas_cumprod=model.diffusion_node.alphas_cumprod,
        neighbor_alphas_cumprod=model.diffusion_neighbor.alphas_cumprod,
        confidence_scale=confidence_scale,
        max_relative_update=max_relative_update,
    ).to(device)
    report = TargetEvidenceFitReport(
        protocol="strict-one-for-all-target-moment-evidence-v1",
        num_nodes=n,
        latent_dim=hidden_dim,
        fit_chunk_size=int(chunk_size),
        variance_floor=float(variance_floor),
        variance_shrinkage=float(variance_shrinkage),
        node_mean_norm=float(node_mean.norm().item()),
        neighbor_mean_norm=float(neighbor_mean.norm().item()),
        node_variance_mean=float(node_variance.mean().item()),
        neighbor_variance_mean=float(neighbor_variance.mean().item()),
    )
    return pair, report
