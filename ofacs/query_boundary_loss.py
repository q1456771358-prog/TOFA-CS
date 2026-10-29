"""Source-only query-aware and community-boundary-aware losses for V5.

This module intentionally avoids full positive-negative Cartesian products.  For
each source query it computes bounded paired ranking terms, boundary mean-margin
terms, and an optional hard-negative softmax term over sampled nodes only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .errors import V5ProtocolError, V5DataLeakageError


@dataclass(frozen=True)
class QueryBoundaryLossConfig:
    rank_margin: float = 0.20
    boundary_margin: float = 0.20
    hard_negative_temperature: float = 0.20
    old_loss_weight: float = 0.05
    rank_weight: float = 1.0
    boundary_weight: float = 1.0
    hard_negative_weight: float = 0.25
    reg_weight: float = 0.0
    eps: float = 1e-12

    def validate(self) -> "QueryBoundaryLossConfig":
        for name in ("rank_margin", "boundary_margin", "hard_negative_temperature", "old_loss_weight", "rank_weight", "boundary_weight", "hard_negative_weight", "reg_weight", "eps"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)):
                raise V5ProtocolError(f"{name} must be finite")
        if self.rank_margin < 0 or self.boundary_margin < 0:
            raise V5ProtocolError("margins must be non-negative")
        if self.hard_negative_temperature <= 0 or self.eps <= 0:
            raise V5ProtocolError("temperature and eps must be positive")
        for name in ("old_loss_weight", "rank_weight", "boundary_weight", "hard_negative_weight", "reg_weight"):
            if float(getattr(self, name)) < 0:
                raise V5ProtocolError(f"{name} must be non-negative")
        return self


@dataclass(frozen=True)
class QueryBoundarySample:
    """One bounded source query supervision sample.

    All IDs are source graph-local node IDs.  These samples may be constructed
    from source labels only.  They must never be constructed from unseen labels
    or unseen ground truth.
    """

    query_local_ids: tuple[int, ...]
    positive_local_ids: tuple[int, ...]
    negative_local_ids: tuple[int, ...]
    boundary_positive_local_ids: tuple[int, ...] = ()
    boundary_negative_local_ids: tuple[int, ...] = ()
    high_score_negative_local_ids: tuple[int, ...] = ()
    low_score_positive_local_ids: tuple[int, ...] = ()
    graph_name: str = "source"
    source_only: bool = True
    sample_reason_stats: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.source_only:
            raise V5DataLeakageError("QueryBoundarySample must be source_only=True")
        for field_name in (
            "query_local_ids", "positive_local_ids", "negative_local_ids",
            "boundary_positive_local_ids", "boundary_negative_local_ids",
            "high_score_negative_local_ids", "low_score_positive_local_ids",
        ):
            values = tuple(dict.fromkeys(int(x) for x in getattr(self, field_name)))
            object.__setattr__(self, field_name, values)
            if any(x < 0 for x in values):
                raise V5ProtocolError(f"{field_name} contains negative local id")
        if not self.query_local_ids:
            raise V5ProtocolError("query_local_ids cannot be empty")
        if not self.positive_local_ids:
            raise V5ProtocolError("positive_local_ids cannot be empty")
        if not self.negative_local_ids:
            raise V5ProtocolError("negative_local_ids cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueryBoundaryLossReport:
    sample_count: int
    rank_loss: float
    boundary_loss: float
    hard_negative_loss: float
    reg_loss: float
    old_aux_loss: float
    total_loss: float
    avg_positive_score: float
    avg_negative_score: float
    avg_boundary_positive_score: float
    avg_boundary_negative_score: float
    avg_pairs_per_query: float
    full_cartesian_used: bool
    source_only: bool
    labels_used: bool
    ground_truth_used: bool
    unseen_training_used: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueryBoundaryLossResult:
    total_loss: torch.Tensor
    rank_loss: torch.Tensor
    boundary_loss: torch.Tensor
    hard_negative_loss: torch.Tensor
    reg_loss: torch.Tensor
    old_aux_loss: torch.Tensor
    report: QueryBoundaryLossReport


def _ids_to_tensor(ids: Sequence[int], *, device: torch.device, num_nodes: int) -> torch.Tensor:
    t = torch.tensor([int(x) for x in ids], dtype=torch.long, device=device)
    if t.numel() == 0:
        raise V5ProtocolError("empty id tensor")
    if int(t.min().item()) < 0 or int(t.max().item()) >= int(num_nodes):
        raise V5ProtocolError(f"sample id outside [0,{num_nodes})")
    return t


def _paired_scores(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded pair construction without full Cartesian products."""

    pair_count = max(int(pos_scores.numel()), int(neg_scores.numel()))
    if pair_count <= 0:
        raise V5ProtocolError("cannot pair empty scores")
    pos_idx = torch.arange(pair_count, device=pos_scores.device) % pos_scores.numel()
    neg_idx = torch.arange(pair_count, device=neg_scores.device) % neg_scores.numel()
    return pos_scores[pos_idx], neg_scores[neg_idx]


def compute_query_boundary_loss(
    embeddings: torch.Tensor,
    samples: Sequence[QueryBoundarySample],
    *,
    config: QueryBoundaryLossConfig | None = None,
    old_aux_loss: torch.Tensor | None = None,
) -> QueryBoundaryLossResult:
    """Compute V5 source-only query-boundary loss over bounded samples."""

    cfg = (config or QueryBoundaryLossConfig()).validate()
    if embeddings.ndim != 2:
        raise V5ProtocolError(f"embeddings must be [N,H], got {tuple(embeddings.shape)}")
    if not samples:
        raise V5ProtocolError("at least one QueryBoundarySample is required")
    n, _ = int(embeddings.shape[0]), int(embeddings.shape[1])
    device = embeddings.device
    emb = F.normalize(embeddings.float(), p=2, dim=-1, eps=cfg.eps)

    rank_losses: list[torch.Tensor] = []
    boundary_losses: list[torch.Tensor] = []
    hn_losses: list[torch.Tensor] = []
    pos_score_log: list[torch.Tensor] = []
    neg_score_log: list[torch.Tensor] = []
    bd_pos_score_log: list[torch.Tensor] = []
    bd_neg_score_log: list[torch.Tensor] = []
    pair_counts: list[int] = []

    for sample in samples:
        if not sample.source_only:
            raise V5DataLeakageError("query-boundary loss received a non-source sample")
        q_ids = _ids_to_tensor(sample.query_local_ids, device=device, num_nodes=n)
        pos_ids = _ids_to_tensor(sample.positive_local_ids, device=device, num_nodes=n)
        neg_ids = _ids_to_tensor(sample.negative_local_ids, device=device, num_nodes=n)

        q_emb = F.normalize(emb.index_select(0, q_ids).mean(dim=0, keepdim=True), p=2, dim=-1, eps=cfg.eps)
        pos_scores = (emb.index_select(0, pos_ids) * q_emb).sum(dim=-1)
        neg_scores = (emb.index_select(0, neg_ids) * q_emb).sum(dim=-1)
        p_pair, n_pair = _paired_scores(pos_scores, neg_scores)
        rank_losses.append(F.relu(float(cfg.rank_margin) - p_pair + n_pair).mean())
        pair_counts.append(int(p_pair.numel()))
        pos_score_log.append(pos_scores.detach())
        neg_score_log.append(neg_scores.detach())

        bd_pos_ids = sample.boundary_positive_local_ids or sample.low_score_positive_local_ids or sample.positive_local_ids
        bd_neg_ids = sample.boundary_negative_local_ids or sample.high_score_negative_local_ids or sample.negative_local_ids
        bd_pos_t = _ids_to_tensor(bd_pos_ids, device=device, num_nodes=n)
        bd_neg_t = _ids_to_tensor(bd_neg_ids, device=device, num_nodes=n)
        bd_pos_scores = (emb.index_select(0, bd_pos_t) * q_emb).sum(dim=-1)
        bd_neg_scores = (emb.index_select(0, bd_neg_t) * q_emb).sum(dim=-1)
        boundary_losses.append(F.relu(float(cfg.boundary_margin) - bd_pos_scores.mean() + bd_neg_scores.mean()))
        bd_pos_score_log.append(bd_pos_scores.detach())
        bd_neg_score_log.append(bd_neg_scores.detach())

        hard_neg_ids = sample.high_score_negative_local_ids or sample.boundary_negative_local_ids or sample.negative_local_ids
        hard_neg_t = _ids_to_tensor(hard_neg_ids, device=device, num_nodes=n)
        hard_neg_scores = (emb.index_select(0, hard_neg_t) * q_emb).sum(dim=-1)
        logits_pos = pos_scores / float(cfg.hard_negative_temperature)
        logits_neg = hard_neg_scores / float(cfg.hard_negative_temperature)
        numerator = torch.logsumexp(logits_pos, dim=0)
        denominator = torch.logsumexp(torch.cat([logits_pos, logits_neg], dim=0), dim=0)
        hn_losses.append(-(numerator - denominator))

    zero = embeddings.new_tensor(0.0)
    rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero
    boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else zero
    hard_negative_loss = torch.stack(hn_losses).mean() if hn_losses else zero
    reg_loss = embeddings.pow(2).mean() if cfg.reg_weight > 0 else zero
    old_loss = old_aux_loss.to(device=embeddings.device, dtype=embeddings.dtype) if old_aux_loss is not None else zero
    total = (
        float(cfg.rank_weight) * rank_loss
        + float(cfg.boundary_weight) * boundary_loss
        + float(cfg.hard_negative_weight) * hard_negative_loss
        + float(cfg.reg_weight) * reg_loss
        + float(cfg.old_loss_weight) * old_loss
    )

    def _mean_or_zero(chunks: list[torch.Tensor]) -> float:
        if not chunks:
            return 0.0
        return float(torch.cat([x.flatten().cpu() for x in chunks]).mean().item())

    report = QueryBoundaryLossReport(
        sample_count=len(samples),
        rank_loss=float(rank_loss.detach().cpu().item()),
        boundary_loss=float(boundary_loss.detach().cpu().item()),
        hard_negative_loss=float(hard_negative_loss.detach().cpu().item()),
        reg_loss=float(reg_loss.detach().cpu().item()),
        old_aux_loss=float(old_loss.detach().cpu().item()),
        total_loss=float(total.detach().cpu().item()),
        avg_positive_score=_mean_or_zero(pos_score_log),
        avg_negative_score=_mean_or_zero(neg_score_log),
        avg_boundary_positive_score=_mean_or_zero(bd_pos_score_log),
        avg_boundary_negative_score=_mean_or_zero(bd_neg_score_log),
        avg_pairs_per_query=float(sum(pair_counts) / max(1, len(pair_counts))),
        full_cartesian_used=False,
        source_only=True,
        labels_used=True,
        ground_truth_used=False,
        unseen_training_used=False,
    )
    return QueryBoundaryLossResult(
        total_loss=total,
        rank_loss=rank_loss,
        boundary_loss=boundary_loss,
        hard_negative_loss=hard_negative_loss,
        reg_loss=reg_loss,
        old_aux_loss=old_loss,
        report=report,
    )
