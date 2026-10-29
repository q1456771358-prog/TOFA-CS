"""Target-unlabeled domain guidance for V5 diffusion.

This module ports the useful DiffGDA idea, not its dense adjacency generator.
It trains a small source/target domain discriminator from label-free
structure-semantic node statistics and exposes the hidden condition vector to
``GuidedDiffusionRefiner``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .errors import V5DataLeakageError, V5ProtocolError


DOMAIN_GUIDANCE_VERSION = "v5-domain-guidance-v1"


@dataclass(frozen=True)
class DomainGuidanceConfig:
    condition_dim: int = 16
    hidden_dim: int = 64
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_nodes_per_graph: int = 4096
    random_seed: int = 42

    def validate(self) -> "DomainGuidanceConfig":
        for name in ("condition_dim", "hidden_dim", "epochs", "batch_size", "max_nodes_per_graph"):
            if int(getattr(self, name)) <= 0:
                raise V5ProtocolError(f"{name} must be positive")
        if float(self.lr) <= 0 or float(self.weight_decay) < 0:
            raise V5ProtocolError("lr must be positive and weight_decay non-negative")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DomainGuidanceReport:
    enabled: bool
    source_graph_name: str
    target_graph_name: str
    feature_dim: int
    condition_dim: int
    epochs_run: int
    sample_count: int
    final_loss: float
    final_domain_accuracy: float
    source_condition_norm: float
    target_condition_norm: float
    task_labels_used: bool = False
    target_ground_truth_used: bool = False
    metrics_computed: bool = False
    checkpoint_selection_from_unseen: bool = False
    target_unlabeled_used: bool = True

    def __post_init__(self) -> None:
        if self.task_labels_used or self.target_ground_truth_used or self.metrics_computed or self.checkpoint_selection_from_unseen:
            raise V5DataLeakageError("unsafe domain guidance report flags")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DomainGuidanceNet(nn.Module):
    def __init__(self, in_dim: int, config: DomainGuidanceConfig):
        super().__init__()
        cfg = config.validate()
        self.in_dim = int(in_dim)
        self.config = cfg
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(int(cfg.hidden_dim), 2)
        self.condition = nn.Linear(int(cfg.hidden_dim) + 2, int(cfg.condition_dim))

    def forward(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if stats.ndim != 2 or int(stats.shape[1]) != self.in_dim:
            raise V5ProtocolError(f"stats must be [N,{self.in_dim}], got {tuple(stats.shape)}")
        hidden = self.backbone(stats.float())
        logits = self.classifier(hidden)
        probs = F.softmax(logits, dim=-1)
        cond = torch.tanh(self.condition(torch.cat([hidden, probs], dim=-1)))
        return logits, cond


def node_domain_statistics(global_token_seqs: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
    """Build label-free node statistics from hop tokens and sparse edge_index."""

    if global_token_seqs.ndim != 3:
        raise V5ProtocolError(f"global_token_seqs must be [N,K,D], got {tuple(global_token_seqs.shape)}")
    tokens = global_token_seqs.detach().cpu().float()
    n, k, _ = int(tokens.shape[0]), int(tokens.shape[1]), int(tokens.shape[2])
    if n <= 0 or k <= 0:
        raise V5ProtocolError("global_token_seqs cannot be empty")
    x0 = tokens[:, 0, :]
    x_last = tokens[:, -1, :]
    hop_norm = tokens.norm(dim=-1)
    if k > 1:
        delta = tokens[:, 1:, :] - tokens[:, :-1, :]
        delta_norm = delta.norm(dim=-1)
        mean_delta = delta_norm.mean(dim=1)
        max_delta = delta_norm.max(dim=1).values
        first_last_cos = F.cosine_similarity(x0, x_last, dim=-1, eps=1e-12)
    else:
        mean_delta = torch.zeros(n)
        max_delta = torch.zeros(n)
        first_last_cos = torch.ones(n)
    degree = torch.zeros(n, dtype=torch.float32)
    if edge_index is not None:
        ei = edge_index.detach().cpu().long()
        if ei.ndim != 2 or int(ei.shape[0]) != 2:
            raise V5ProtocolError("edge_index must be [2,E]")
        if ei.numel() > 0:
            if int(ei.min().item()) < 0 or int(ei.max().item()) >= n:
                raise V5ProtocolError("edge_index contains node id outside stats range")
            degree.scatter_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.float32))
            degree.scatter_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.float32))
    degree = torch.log1p(degree)
    stats = torch.stack(
        [
            x0.norm(dim=-1),
            x_last.norm(dim=-1),
            hop_norm.mean(dim=1),
            hop_norm.std(dim=1, unbiased=False),
            mean_delta,
            max_delta,
            x0.mean(dim=-1),
            x0.std(dim=-1, unbiased=False),
            x0.abs().mean(dim=-1),
            first_last_cos,
            degree,
        ],
        dim=1,
    )
    if not torch.isfinite(stats).all():
        raise V5ProtocolError("domain guidance stats contain NaN/Inf")
    return stats


def fit_domain_guidance(
    *,
    source_global_token_seqs: torch.Tensor,
    target_global_token_seqs: torch.Tensor,
    source_edge_index: torch.Tensor | None,
    target_edge_index: torch.Tensor | None,
    source_graph_name: str,
    target_graph_name: str,
    config: DomainGuidanceConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[DomainGuidanceNet, torch.Tensor, torch.Tensor, DomainGuidanceReport]:
    """Train source/target discriminator from unlabeled node statistics."""

    cfg = (config or DomainGuidanceConfig()).validate()
    src_stats = node_domain_statistics(source_global_token_seqs, source_edge_index)
    tgt_stats = node_domain_statistics(target_global_token_seqs, target_edge_index)
    src_idx = _sample_indices(src_stats.shape[0], cfg.max_nodes_per_graph, cfg.random_seed)
    tgt_idx = _sample_indices(tgt_stats.shape[0], cfg.max_nodes_per_graph, cfg.random_seed + 1)
    train_x = torch.cat([src_stats[src_idx], tgt_stats[tgt_idx]], dim=0)
    train_y = torch.cat(
        [
            torch.zeros(len(src_idx), dtype=torch.long),
            torch.ones(len(tgt_idx), dtype=torch.long),
        ],
        dim=0,
    )
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    model = DomainGuidanceNet(in_dim=int(train_x.shape[1]), config=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.random_seed))
    final_loss = 0.0
    final_acc = 0.0
    for _ in range(int(cfg.epochs)):
        order = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, int(order.numel()), int(cfg.batch_size)):
            batch_idx = order[start : start + int(cfg.batch_size)]
            xb = train_x[batch_idx].to(device)
            yb = train_y[batch_idx].to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits, _ = model(train_x.to(device))
            loss = F.cross_entropy(logits, train_y.to(device))
            pred = logits.argmax(dim=-1).cpu()
            final_loss = float(loss.detach().cpu().item())
            final_acc = float((pred == train_y).float().mean().item())
    with torch.no_grad():
        src_cond = make_structure_condition(model, src_stats, mean=mean, std=std, device=device)
        tgt_cond = make_structure_condition(model, tgt_stats, mean=mean, std=std, device=device)
    report = DomainGuidanceReport(
        enabled=True,
        source_graph_name=str(source_graph_name),
        target_graph_name=str(target_graph_name),
        feature_dim=int(train_x.shape[1]),
        condition_dim=int(cfg.condition_dim),
        epochs_run=int(cfg.epochs),
        sample_count=int(train_x.shape[0]),
        final_loss=final_loss,
        final_domain_accuracy=final_acc,
        source_condition_norm=float(src_cond.norm(dim=-1).mean().item()),
        target_condition_norm=float(tgt_cond.norm(dim=-1).mean().item()),
    )
    return model.cpu(), mean.cpu(), std.cpu(), report


@torch.no_grad()
def make_structure_condition(
    model: DomainGuidanceNet,
    stats: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    model = model.to(device)
    was_training = bool(model.training)
    model.eval()
    x = (stats.detach().cpu().float() - mean.cpu().float()) / std.cpu().float().clamp_min(1e-6)
    _, cond = model(x.to(device))
    out = cond.detach().cpu()
    if was_training:
        model.train()
    return out


def export_domain_guidance_state(
    *,
    model: DomainGuidanceNet,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: DomainGuidanceConfig,
    report: DomainGuidanceReport,
) -> dict[str, object]:
    return {
        "version": DOMAIN_GUIDANCE_VERSION,
        "model_state_dict": model.cpu().state_dict(),
        "normalization_mean": mean.detach().cpu(),
        "normalization_std": std.detach().cpu(),
        "config": config.to_dict(),
        "report": report.to_dict(),
        "task_labels_used": False,
        "target_ground_truth_used": False,
        "metrics_computed": False,
        "checkpoint_selection_from_unseen": False,
    }


def load_domain_guidance_state(payload: Mapping[str, object]) -> tuple[DomainGuidanceNet, torch.Tensor, torch.Tensor, DomainGuidanceConfig, dict[str, object]]:
    if payload.get("version") != DOMAIN_GUIDANCE_VERSION:
        raise V5ProtocolError("unsupported domain guidance state version")
    if bool(payload.get("task_labels_used", False)) or bool(payload.get("target_ground_truth_used", False)) or bool(payload.get("metrics_computed", False)):
        raise V5DataLeakageError("domain guidance state contains unsafe supervision flags")
    cfg = DomainGuidanceConfig(**dict(payload.get("config", {}))).validate()
    mean = payload.get("normalization_mean")
    std = payload.get("normalization_std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise V5ProtocolError("domain guidance state missing normalization tensors")
    model = DomainGuidanceNet(in_dim=int(mean.numel()), config=cfg)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise V5ProtocolError("domain guidance state missing model_state_dict")
    model.load_state_dict(dict(state))
    return model, mean.float().view(1, -1), std.float().view(1, -1), cfg, dict(payload.get("report", {}))


def _sample_indices(n: int, max_count: int, seed: int) -> torch.Tensor:
    if int(n) <= 0:
        raise V5ProtocolError("cannot sample from empty stats")
    count = min(int(n), int(max_count))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if count == int(n):
        return torch.arange(int(n), dtype=torch.long)
    return torch.randperm(int(n), generator=generator)[:count]
