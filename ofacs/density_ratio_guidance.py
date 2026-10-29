"""Target-unlabeled density-ratio guidance for V5 OFA-CS.

This module contains two complementary pieces:

* ``compute_source_target_alignment_loss``: a conservative MMD/CORAL proxy kept
  for backward compatibility and diagnostics. It is **not** the full DiffGDA
  reverse guidance.
* ``DensityRatioGuidancePair``: a target-unlabeled domain-ratio estimator used
  by reverse denoising.  It trains discriminators D(z_t,t) to distinguish source
  vs target noisy embeddings and uses their logits as an estimate of
  ``log q_T(z_t) - log p_S(z_t)``. During fixed inference, reverse diffusion can
  add ``grad_z log q_T/p_S`` to the source denoising score, implementing the
  DiffGDA-style density-ratio guided reverse trajectory in representation space.

No target labels or target ground-truth communities are read or required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .errors import V5ProtocolError


@dataclass(frozen=True)
class AlignmentLossReport:
    source_nodes: int
    target_nodes: int
    mmd_linear: float
    coral: float
    total: float
    target_labels_used: bool = False
    target_gt_used: bool = False
    source_only_model: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sample_rows(x: torch.Tensor, max_nodes: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    if x.ndim != 2:
        raise V5ProtocolError("alignment/guidance tensors must be [N,D]")
    n = int(x.shape[0])
    if n <= int(max_nodes):
        return x
    idx = torch.randperm(n, device=x.device, generator=generator)[: int(max_nodes)]
    return x.index_select(0, idx)


def _sample_equal_rows(
    source: torch.Tensor,
    target: torch.Tensor,
    max_nodes: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 2 or target.ndim != 2:
        raise V5ProtocolError("density-ratio guidance tensors must be [N,D]")
    if int(source.shape[1]) != int(target.shape[1]):
        raise V5ProtocolError("source/target guidance dim mismatch")
    k = min(int(source.shape[0]), int(target.shape[0]), int(max_nodes))
    if k <= 0:
        raise V5ProtocolError("density-ratio guidance requires non-empty source and target embeddings")
    s_idx = torch.randperm(int(source.shape[0]), device=source.device, generator=generator)[:k]
    t_idx = torch.randperm(int(target.shape[0]), device=target.device, generator=generator)[:k]
    return source.index_select(0, s_idx), target.index_select(0, t_idx)


def _cov(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 1:
        return torch.zeros((x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.t() @ xc) / float(x.shape[0] - 1)


def compute_source_target_alignment_loss(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    max_nodes: int = 4096,
    coral_weight: float = 0.1,
) -> tuple[torch.Tensor, AlignmentLossReport]:
    """Compute light MMD/CORAL alignment without target labels.

    This is retained as a small optional auxiliary alignment proxy. It is not the
    gradient-based density-ratio reverse guidance used in ``guided_refined``.
    """

    if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise V5ProtocolError("alignment embeddings must be [N,D]")
    if int(source_embeddings.shape[1]) != int(target_embeddings.shape[1]):
        raise V5ProtocolError("source/target alignment dim mismatch")
    src = F.normalize(_sample_rows(source_embeddings.float(), max_nodes), p=2, dim=-1, eps=1e-12)
    tgt = F.normalize(_sample_rows(target_embeddings.float(), max_nodes), p=2, dim=-1, eps=1e-12)
    mmd = (src.mean(dim=0) - tgt.mean(dim=0)).pow(2).mean()
    coral = (_cov(src) - _cov(tgt)).pow(2).mean()
    total = mmd + float(coral_weight) * coral
    report = AlignmentLossReport(
        source_nodes=int(src.shape[0]),
        target_nodes=int(tgt.shape[0]),
        mmd_linear=float(mmd.detach().cpu().item()),
        coral=float(coral.detach().cpu().item()),
        total=float(total.detach().cpu().item()),
    )
    return total, report


class RatioTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(1, self.dim // 2)
        device = t.device
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / float(half))
        emb = t.unsqueeze(-1).float() * freq.unsqueeze(0)
        out = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if out.shape[1] < self.dim:
            out = F.pad(out, (0, self.dim - out.shape[1]))
        return out[:, : self.dim]


class DensityRatioMLP(nn.Module):
    """Log-density-ratio estimator log q_target(z_t) - log p_source(z_t).

    The network is trained as a balanced binary domain classifier with target=1
    and source=0. With equal class priors, the classifier logit estimates
    ``log q_T(z_t) / p_S(z_t)`` up to a constant, whose gradient is unchanged.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        feature_mode: str = "unit_direction",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.feature_mode = str(feature_mode)
        if self.latent_dim <= 0 or self.hidden_dim <= 0 or self.time_emb_dim <= 0:
            raise V5ProtocolError("DensityRatioMLP dims must be positive")
        if self.feature_mode not in {"unit_direction", "rms_preserve"}:
            raise V5ProtocolError(f"unsupported density-ratio feature_mode={self.feature_mode!r}")
        self.register_buffer("input_rms", torch.ones((), dtype=torch.float32))
        self.guidance_scale_mode = "legacy_raw"
        self.guidance_confidence_scale = 1.0
        self.max_relative_update = 0.10
        self.time_emb = RatioTimeEmbedding(self.time_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + self.time_emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise V5ProtocolError("density ratio input z must be [B,D]")
        if int(z.shape[1]) != self.latent_dim:
            raise V5ProtocolError(f"density ratio latent dim mismatch: got {z.shape[1]}, expected {self.latent_dim}")
        if t.ndim == 0:
            t = t.expand(z.shape[0])
        if t.ndim != 1 or int(t.shape[0]) != int(z.shape[0]):
            raise V5ProtocolError("density ratio timestep t must be [B]")
        z_float = z.float()
        if self.feature_mode == "unit_direction":
            z_norm = F.normalize(z_float, p=2, dim=-1, eps=1e-12)
        else:
            z_norm = z_float / torch.clamp(self.input_rms.to(device=z.device, dtype=z_float.dtype), min=1e-6)
        te = self.time_emb(t.to(device=z.device))
        return self.net(torch.cat([z_norm, te.to(dtype=z_norm.dtype)], dim=-1)).squeeze(-1)

    def log_density_ratio(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(z, t)

    @torch.no_grad()
    def set_input_rms(self, value: torch.Tensor | float) -> None:
        scalar = torch.as_tensor(value, dtype=self.input_rms.dtype, device=self.input_rms.device).reshape(())
        if not torch.isfinite(scalar) or float(scalar.item()) <= 0.0:
            raise V5ProtocolError("density-ratio input RMS must be finite and positive")
        self.input_rms.copy_(scalar)


@dataclass(frozen=True)
class DensityRatioTrainReport:
    loss: float
    node_loss: float
    neighbor_loss: float
    node_acc: float
    neighbor_acc: float
    samples_per_domain: int
    target_labels_used: bool = False
    target_gt_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DensityRatioGuidancePair(nn.Module):
    """Node/neighbor density-ratio guidance pair for OFA-CS embeddings."""

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int = 256,
        time_emb_dim: int = 64,
        feature_mode: str = "unit_direction",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_emb_dim = int(time_emb_dim)
        self.feature_mode = str(feature_mode)
        self.node = DensityRatioMLP(
            latent_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim, feature_mode=feature_mode
        )
        self.neighbor = DensityRatioMLP(
            latent_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim, feature_mode=feature_mode
        )

    @torch.no_grad()
    def fit_frozen_rms(
        self,
        *,
        source_node: torch.Tensor,
        source_neighbor: torch.Tensor,
        target_node: torch.Tensor,
        target_neighbor: torch.Tensor,
    ) -> dict[str, float]:
        """Freeze pooled RMS scales without deleting radial domain shift."""

        def pooled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            total = a.detach().float().pow(2).sum() + b.detach().float().pow(2).sum()
            count = int(a.numel()) + int(b.numel())
            return torch.sqrt(total / max(1, count)).clamp_min(1e-6)

        node_rms = pooled(source_node, target_node)
        neighbor_rms = pooled(source_neighbor, target_neighbor)
        self.node.set_input_rms(node_rms)
        self.neighbor.set_input_rms(neighbor_rms)
        return {"node_input_rms": float(node_rms.item()), "neighbor_input_rms": float(neighbor_rms.item())}

    def train_step(
        self,
        *,
        source_node: torch.Tensor,
        source_neighbor: torch.Tensor,
        target_node: torch.Tensor,
        target_neighbor: torch.Tensor,
        refiner_node: object,
        refiner_neighbor: object,
        optimizer: torch.optim.Optimizer,
        max_nodes: int = 4096,
        max_timestep: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, DensityRatioTrainReport]:
        """One balanced BCE step on noisy source/target embeddings.

        Embeddings are detached on purpose: the guidance discriminator learns the
        source/target density ratio; it is not an adversarial loss on the main
        encoder. This preserves the SeniorExact source-training mainline.
        """

        device = source_node.device
        s_node, t_node = _sample_equal_rows(
            source_node.detach(), target_node.detach(), max_nodes, generator=generator
        )
        s_nei, t_nei = _sample_equal_rows(
            source_neighbor.detach(), target_neighbor.detach(), max_nodes, generator=generator
        )
        k = min(int(s_node.shape[0]), int(s_nei.shape[0]), int(t_node.shape[0]), int(t_nei.shape[0]))
        s_node, t_node, s_nei, t_nei = s_node[:k], t_node[:k], s_nei[:k], t_nei[:k]

        optimizer.zero_grad(set_to_none=True)
        node_z0 = torch.cat([s_node, t_node], dim=0).to(device)
        nei_z0 = torch.cat([s_nei, t_nei], dim=0).to(device)
        y = torch.cat([torch.zeros(k, device=device), torch.ones(k, device=device)], dim=0)
        node_high = int(refiner_node.T)
        neighbor_high = int(refiner_neighbor.T)
        if max_timestep is not None:
            node_high = min(node_high, int(max_timestep) + 1)
            neighbor_high = min(neighbor_high, int(max_timestep) + 1)
        if node_high <= 0 or neighbor_high <= 0:
            raise V5ProtocolError("density-ratio training timestep range must be non-empty")
        t_node_step = torch.randint(0, node_high, (2 * k,), device=device, dtype=torch.long, generator=generator)
        t_nei_step = torch.randint(0, neighbor_high, (2 * k,), device=device, dtype=torch.long, generator=generator)
        node_noise = torch.randn(node_z0.shape, device=device, dtype=node_z0.dtype, generator=generator)
        nei_noise = torch.randn(nei_z0.shape, device=device, dtype=nei_z0.dtype, generator=generator)
        node_zt, _ = refiner_node.q_sample(node_z0, t_node_step, noise=node_noise)
        nei_zt, _ = refiner_neighbor.q_sample(nei_z0, t_nei_step, noise=nei_noise)
        node_logits = self.node(node_zt, t_node_step)
        nei_logits = self.neighbor(nei_zt, t_nei_step)
        node_loss = F.binary_cross_entropy_with_logits(node_logits, y)
        nei_loss = F.binary_cross_entropy_with_logits(nei_logits, y)
        loss = node_loss + nei_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            node_pred = (torch.sigmoid(node_logits) >= 0.5).float()
            nei_pred = (torch.sigmoid(nei_logits) >= 0.5).float()
            node_acc = float((node_pred == y).float().mean().detach().cpu().item())
            nei_acc = float((nei_pred == y).float().mean().detach().cpu().item())
        report = DensityRatioTrainReport(
            loss=float(loss.detach().cpu().item()),
            node_loss=float(node_loss.detach().cpu().item()),
            neighbor_loss=float(nei_loss.detach().cpu().item()),
            node_acc=node_acc,
            neighbor_acc=nei_acc,
            samples_per_domain=int(k),
        )
        return loss.detach(), report

    def export_state(
        self,
        *,
        guidance_weight: float,
        guidance_clip: float,
        guidance_scale_mode: str = "legacy_raw",
        guidance_confidence_scale: float = 1.0,
        max_relative_update: float = 0.10,
        target_graph: str | None = None,
        source_graphs: tuple[str, ...] | list[str] | None = None,
        max_timestep: int | None = None,
        backbone_sha256: str | None = None,
        fit_report: Mapping[str, object] | None = None,
        loader_audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if guidance_scale_mode not in {"legacy_raw", "relative_eps"}:
            raise V5ProtocolError(f"unsupported guidance_scale_mode={guidance_scale_mode!r}")
        return {
            "protocol": "DiffGDA-density-ratio-guided-reverse-denoise-v2-target-specific",
            "latent_dim": int(self.latent_dim),
            "hidden_dim": int(self.hidden_dim),
            "time_emb_dim": int(self.time_emb_dim),
            "feature_mode": str(self.feature_mode),
            "guidance_weight": float(guidance_weight),
            "guidance_clip": float(guidance_clip),
            "guidance_scale_mode": str(guidance_scale_mode),
            "guidance_confidence_scale": float(guidance_confidence_scale),
            "max_relative_update": float(max_relative_update),
            "target_graph": None if target_graph is None else str(target_graph).lower(),
            "source_graphs": [str(x).lower() for x in (source_graphs or [])],
            "max_timestep": None if max_timestep is None else int(max_timestep),
            "backbone_sha256": backbone_sha256,
            "fit_report": dict(fit_report or {}),
            "loader_audit": dict(loader_audit or {}),
            "node_state_dict": self.node.state_dict(),
            "neighbor_state_dict": self.neighbor.state_dict(),
            "target_labels_used": False,
            "target_gt_used": False,
        }


def load_density_ratio_guidance_pair(state: Mapping[str, object], *, map_location: str | torch.device = "cpu") -> tuple[DensityRatioGuidancePair, dict[str, object]]:
    if not isinstance(state, Mapping):
        raise V5ProtocolError("density_ratio_guidance state must be a mapping")
    latent_dim = int(state.get("latent_dim", 0))
    hidden_dim = int(state.get("hidden_dim", 256))
    time_emb_dim = int(state.get("time_emb_dim", 64))
    feature_mode = str(state.get("feature_mode", "unit_direction"))
    if latent_dim <= 0:
        raise V5ProtocolError("density_ratio_guidance state missing latent_dim")
    pair = DensityRatioGuidancePair(
        latent_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim, feature_mode=feature_mode
    )
    for estimator, key in ((pair.node, "node_state_dict"), (pair.neighbor, "neighbor_state_dict")):
        result = estimator.load_state_dict(state[key], strict=False)
        unexpected = set(result.unexpected_keys)
        missing = set(result.missing_keys) - {"input_rms"}
        if unexpected or missing:
            raise V5ProtocolError(
                f"invalid density-ratio estimator state {key}: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
    pair.to(map_location if isinstance(map_location, torch.device) else torch.device(map_location))
    scale_mode = str(state.get("guidance_scale_mode", "legacy_raw"))
    confidence_scale = float(state.get("guidance_confidence_scale", 1.0))
    max_relative_update = float(state.get("max_relative_update", 0.10))
    for estimator in (pair.node, pair.neighbor):
        estimator.guidance_scale_mode = scale_mode
        estimator.guidance_confidence_scale = confidence_scale
        estimator.max_relative_update = max_relative_update
    pair.eval()
    report = {
        "protocol": str(state.get("protocol", "unknown")),
        "guidance_weight": float(state.get("guidance_weight", 0.0)),
        "guidance_clip": float(state.get("guidance_clip", 0.0)),
        "guidance_scale_mode": scale_mode,
        "guidance_confidence_scale": confidence_scale,
        "max_relative_update": max_relative_update,
        "feature_mode": feature_mode,
        "target_graph": state.get("target_graph"),
        "source_graphs": list(state.get("source_graphs", [])),
        "max_timestep": state.get("max_timestep"),
        "backbone_sha256": state.get("backbone_sha256"),
        "fit_report": dict(state.get("fit_report", {})),
        "loader_audit": dict(state.get("loader_audit", {})),
        "target_labels_used": bool(state.get("target_labels_used", False)),
        "target_gt_used": bool(state.get("target_gt_used", False)),
    }
    return pair, report
