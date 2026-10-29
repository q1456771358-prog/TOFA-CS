"""Label-free structure-semantic condition vectors for V5 guided diffusion.

These conditions are intentionally computed from X/A/query seeds/current scores
only.  They may be used during unseen inference because they do not read labels,
.gt files, metrics, or oracle boundary information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .errors import V5DataLeakageError, V5ProtocolError


@dataclass(frozen=True)
class StructureSemanticConditionConfig:
    condition_dim: int = 16
    max_edges_for_feature_stats: int = 200_000
    eps: float = 1e-12
    include_frequency_placeholder: bool = False  # Stage 3 only; default off.

    def validate(self) -> "StructureSemanticConditionConfig":
        if int(self.condition_dim) < 16:
            raise V5ProtocolError("condition_dim must be at least 16")
        if int(self.max_edges_for_feature_stats) <= 0:
            raise V5ProtocolError("max_edges_for_feature_stats must be positive")
        if float(self.eps) <= 0:
            raise V5ProtocolError("eps must be positive")
        return self


@dataclass(frozen=True)
class StructureSemanticConditionResult:
    graph_condition: torch.Tensor  # [D]
    query_condition: torch.Tensor  # [Q,D]
    condition_dim: int
    report: dict[str, object]
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_used: bool = False
    unseen_training_used: bool = False

    def __post_init__(self) -> None:
        if self.labels_used or self.ground_truth_used or self.metrics_used or self.unseen_training_used:
            raise V5DataLeakageError("structure-semantic condition must be label-free")
        if self.graph_condition.ndim != 1:
            raise V5ProtocolError("graph_condition must be [D]")
        if self.query_condition.ndim != 2:
            raise V5ProtocolError("query_condition must be [Q,D]")
        if self.graph_condition.shape[0] != self.condition_dim or self.query_condition.shape[1] != self.condition_dim:
            raise V5ProtocolError("condition dim mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_dim": self.condition_dim,
            "graph_condition_shape": tuple(self.graph_condition.shape),
            "query_condition_shape": tuple(self.query_condition.shape),
            "report": dict(self.report),
            "labels_used": self.labels_used,
            "ground_truth_used": self.ground_truth_used,
            "metrics_used": self.metrics_used,
            "unseen_training_used": self.unseen_training_used,
        }


def _edge_index(edge_index: torch.Tensor | Sequence[Sequence[int]] | None, *, num_nodes: int) -> torch.Tensor | None:
    if edge_index is None:
        return None
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().long()
    else:
        ei = torch.tensor(edge_index, dtype=torch.long)
    if ei.numel() == 0:
        return None
    if ei.ndim != 2:
        raise V5ProtocolError("edge_index must be [2,E] or [E,2]")
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.t().contiguous()
    if ei.shape[0] != 2:
        raise V5ProtocolError("edge_index must have shape [2,E]")
    if int(ei.min().item()) < 0 or int(ei.max().item()) >= int(num_nodes):
        raise V5ProtocolError("edge_index contains out-of-range ids")
    return ei


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    return x.float().std(unbiased=False) if x.numel() > 0 else torch.tensor(0.0)


def _score_entropy(scores: torch.Tensor | None, eps: float) -> float:
    if scores is None or scores.numel() == 0:
        return 0.0
    s = scores.detach().cpu().float().flatten()
    p = torch.softmax(s, dim=0)
    ent = -(p * torch.log(p + eps)).sum() / torch.log(torch.tensor(float(max(2, s.numel()))))
    return float(ent.item())


def build_structure_semantic_condition(
    *,
    node_features: torch.Tensor,
    query_node_sets: Sequence[Sequence[int] | torch.Tensor],
    edge_index: torch.Tensor | Sequence[Sequence[int]] | None = None,
    current_scores: torch.Tensor | None = None,
    config: StructureSemanticConditionConfig | None = None,
    split: str = "source_or_unseen_forward",
) -> StructureSemanticConditionResult:
    """Build graph/query condition vectors from label-free information only."""

    cfg = (config or StructureSemanticConditionConfig()).validate()
    if any(x in str(split).lower() for x in ("metric", "gt", "label")):
        raise V5DataLeakageError("split name suggests forbidden label/metric use")
    if node_features.ndim != 2:
        raise V5ProtocolError("node_features must be [N,F]")
    x = node_features.detach().cpu().float()
    n, f = int(x.shape[0]), int(x.shape[1])
    if n <= 0 or f <= 0:
        raise V5ProtocolError("node_features must have positive shape")
    ei = _edge_index(edge_index, num_nodes=n)
    device = node_features.device

    if ei is None:
        degrees = torch.zeros(n, dtype=torch.float32)
        edge_count = 0
        edge_sim_mean = torch.tensor(0.0)
        edge_sim_std = torch.tensor(0.0)
    else:
        row, col = ei[0], ei[1]
        edge_count = int(row.numel())
        degrees = torch.bincount(row, minlength=n).float()
        # Bounded edge feature similarity stats.
        if row.numel() > cfg.max_edges_for_feature_stats:
            idx = torch.linspace(0, row.numel() - 1, steps=cfg.max_edges_for_feature_stats).long()
            row_s, col_s = row[idx], col[idx]
        else:
            row_s, col_s = row, col
        x_norm = F.normalize(x, p=2, dim=-1, eps=cfg.eps)
        edge_sims = (x_norm.index_select(0, row_s) * x_norm.index_select(0, col_s)).sum(dim=-1) if row_s.numel() > 0 else torch.tensor([])
        edge_sim_mean = edge_sims.mean() if edge_sims.numel() > 0 else torch.tensor(0.0)
        edge_sim_std = _safe_std(edge_sims)

    density = float(edge_count) / float(max(1, n * max(1, n - 1)))
    graph_values = [
        torch.log1p(torch.tensor(float(n))) / 12.0,
        torch.log1p(torch.tensor(float(edge_count))) / 16.0,
        torch.tensor(density),
        degrees.mean() / max(1.0, float(n)),
        _safe_std(degrees) / max(1.0, float(n)),
        degrees.max() / max(1.0, float(n)),
        edge_sim_mean,
        edge_sim_std,
        torch.tensor(_score_entropy(current_scores.flatten() if current_scores is not None and current_scores.ndim == 1 else None, cfg.eps)),
    ]
    graph_vec = torch.stack([v.float() for v in graph_values])

    qsets: list[tuple[int, ...]] = []
    for q in query_node_sets:
        ids = tuple(int(z) for z in (q.detach().cpu().flatten().tolist() if isinstance(q, torch.Tensor) else q))
        if not ids:
            raise V5ProtocolError("empty query set in condition builder")
        if min(ids) < 0 or max(ids) >= n:
            raise V5ProtocolError("query id outside graph range in condition builder")
        qsets.append(tuple(dict.fromkeys(ids)))

    q_rows: list[torch.Tensor] = []
    x_norm = F.normalize(x, p=2, dim=-1, eps=cfg.eps)
    for qi, qset in enumerate(qsets):
        q_ids = torch.tensor(qset, dtype=torch.long)
        q_deg = degrees.index_select(0, q_ids)
        q_feat = x_norm.index_select(0, q_ids)
        if q_feat.shape[0] > 1:
            q_centroid = F.normalize(q_feat.mean(dim=0, keepdim=True), p=2, dim=-1, eps=cfg.eps)
            intra_sim = (q_feat * q_centroid).sum(dim=-1).mean()
        else:
            intra_sim = torch.tensor(1.0)
        if current_scores is not None and current_scores.ndim == 2 and qi < current_scores.shape[0]:
            ent = _score_entropy(current_scores[qi], cfg.eps)
        else:
            ent = 0.0
        q_values = [
            torch.tensor(float(len(qset)) / max(1.0, float(n))),
            q_deg.mean() / max(1.0, float(n)),
            _safe_std(q_deg) / max(1.0, float(n)),
            q_deg.max() / max(1.0, float(n)),
            intra_sim,
            torch.tensor(ent),
        ]
        row = torch.cat([graph_vec, torch.stack([v.float() for v in q_values])], dim=0)
        q_rows.append(row)

    def _pad(vec: torch.Tensor, dim: int) -> torch.Tensor:
        if vec.numel() > dim:
            return vec[:dim]
        if vec.numel() < dim:
            return torch.cat([vec, torch.zeros(dim - vec.numel(), dtype=vec.dtype)], dim=0)
        return vec

    graph_condition = _pad(graph_vec, cfg.condition_dim).to(device=device)
    query_condition = torch.stack([_pad(v, cfg.condition_dim) for v in q_rows], dim=0).to(device=device)
    return StructureSemanticConditionResult(
        graph_condition=graph_condition,
        query_condition=query_condition,
        condition_dim=int(cfg.condition_dim),
        report={
            "num_nodes": n,
            "num_edges": edge_count,
            "density": density,
            "edge_feature_similarity_mean": float(edge_sim_mean.item()),
            "edge_feature_similarity_std": float(edge_sim_std.item()),
            "query_count": len(qsets),
            "condition_source": "label_free_X_A_query_scores_only",
            "frequency_placeholder_enabled": bool(cfg.include_frequency_placeholder),
        },
        labels_used=False,
        ground_truth_used=False,
        metrics_used=False,
        unseen_training_used=False,
    )
