"""Source-only query-boundary sampler for V5.

The sampler replaces V4's weak random 10-positive/10-negative query supervision
with bounded hard-negative and boundary-aware samples.  It uses source labels only
and never consumes unseen labels, gt, metrics, or oracle diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .errors import V5DataLeakageError, V5ProtocolError
from .query_boundary_loss import QueryBoundarySample


@dataclass(frozen=True)
class SourceQueryBoundarySamplerConfig:
    max_queries: int = 128
    query_seed_count: int = 3
    positive_count: int = 32
    negative_count: int = 32
    boundary_positive_count: int = 16
    boundary_negative_count: int = 16
    hard_negative_count: int = 32
    low_score_positive_count: int = 16
    random_seed: int = 42
    max_boundary_pool: int = 4096
    max_feature_pool: int = 4096
    source_only: bool = True

    def validate(self) -> "SourceQueryBoundarySamplerConfig":
        if not self.source_only:
            raise V5DataLeakageError("source query-boundary sampler must be source_only=True")
        for name in (
            "max_queries", "query_seed_count", "positive_count", "negative_count",
            "boundary_positive_count", "boundary_negative_count", "hard_negative_count",
            "low_score_positive_count", "max_boundary_pool", "max_feature_pool",
        ):
            if int(getattr(self, name)) <= 0:
                raise V5ProtocolError(f"{name} must be positive")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceQueryBoundaryPlan:
    graph_name: str
    samples: tuple[QueryBoundarySample, ...]
    config: SourceQueryBoundarySamplerConfig
    report: dict[str, object]
    source_only: bool = True
    labels_used: bool = True
    unseen_labels_used: bool = False
    ground_truth_used: bool = False
    metrics_used: bool = False
    full_cartesian_used: bool = False

    def __post_init__(self) -> None:
        if not self.source_only or self.unseen_labels_used or self.ground_truth_used or self.metrics_used:
            raise V5DataLeakageError("unsafe source query-boundary plan flags")
        if not self.samples:
            raise V5ProtocolError("SourceQueryBoundaryPlan requires at least one sample")

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["samples"] = [s.to_dict() for s in self.samples]
        return d


def _labels_to_cpu(labels: torch.Tensor | Sequence[int]) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        out = labels.detach().cpu().long().flatten()
    else:
        out = torch.tensor([int(x) for x in labels], dtype=torch.long)
    if out.numel() <= 0:
        raise V5ProtocolError("labels cannot be empty for source sampler")
    return out


def _edge_index_to_pairs(edge_index: torch.Tensor | Sequence[Sequence[int]] | None, *, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor] | None:
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
        raise V5ProtocolError("edge_index contains node id outside graph range")
    return ei[0], ei[1]


def _choice(pool: torch.Tensor, count: int, *, generator: torch.Generator, replace_if_needed: bool = True) -> tuple[int, ...]:
    pool = pool.detach().cpu().long().flatten().unique(sorted=False)
    if pool.numel() == 0:
        return tuple()
    count = int(count)
    if pool.numel() >= count:
        perm = torch.randperm(pool.numel(), generator=generator)[:count]
        return tuple(int(x) for x in pool[perm].tolist())
    if not replace_if_needed:
        return tuple(int(x) for x in pool.tolist())
    idx = torch.randint(0, pool.numel(), (count,), generator=generator)
    return tuple(int(x) for x in pool[idx].tolist())


def _topk_from_scores(ids: torch.Tensor, scores: torch.Tensor, count: int, *, largest: bool) -> tuple[int, ...]:
    ids = ids.detach().cpu().long().flatten().unique(sorted=False)
    if ids.numel() == 0:
        return tuple()
    score_cpu = scores.detach().cpu().float().flatten()
    local_scores = score_cpu.index_select(0, ids)
    k = min(int(count), int(ids.numel()))
    order = torch.topk(local_scores, k=k, largest=largest, sorted=True).indices
    return tuple(int(x) for x in ids[order].tolist())


def _feature_similar_negatives(
    *,
    node_features: torch.Tensor,
    query_ids: tuple[int, ...],
    negative_pool: torch.Tensor,
    count: int,
    max_pool: int,
    generator: torch.Generator,
) -> tuple[int, ...]:
    if node_features is None or negative_pool.numel() == 0:
        return tuple()
    pool = negative_pool.detach().cpu().long().flatten().unique(sorted=False)
    if pool.numel() > int(max_pool):
        pool = pool[torch.randperm(pool.numel(), generator=generator)[: int(max_pool)]]
    feat = node_features.detach().cpu().float()
    q = torch.tensor(query_ids, dtype=torch.long)
    q_emb = F.normalize(feat.index_select(0, q).mean(dim=0, keepdim=True), p=2, dim=-1)
    cand = F.normalize(feat.index_select(0, pool), p=2, dim=-1)
    sims = (cand * q_emb).sum(dim=-1)
    k = min(int(count), int(pool.numel()))
    order = torch.topk(sims, k=k, largest=True, sorted=True).indices
    return tuple(int(x) for x in pool[order].tolist())


def _boundary_nodes(labels: torch.Tensor, edge_index: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Return per-label boundary positives and adjacent boundary negatives."""

    labels_cpu = labels.detach().cpu().long()
    boundary_pos: dict[int, set[int]] = {int(c): set() for c in torch.unique(labels_cpu).tolist()}
    boundary_neg: dict[int, set[int]] = {int(c): set() for c in torch.unique(labels_cpu).tolist()}
    if edge_index is None:
        return boundary_pos, boundary_neg
    row, col = edge_index
    for u, v in zip(row.tolist(), col.tolist()):
        lu, lv = int(labels_cpu[u].item()), int(labels_cpu[v].item())
        if lu != lv:
            boundary_pos[lu].add(int(u))
            boundary_neg[lu].add(int(v))
            boundary_pos[lv].add(int(v))
            boundary_neg[lv].add(int(u))
    return boundary_pos, boundary_neg


def _make_query_sets_from_labels(labels: torch.Tensor, *, cfg: SourceQueryBoundarySamplerConfig, generator: torch.Generator) -> tuple[tuple[int, ...], ...]:
    qsets: list[tuple[int, ...]] = []
    classes = torch.unique(labels).tolist()
    class_order = torch.tensor(classes, dtype=torch.long)
    if class_order.numel() > 1:
        class_order = class_order[torch.randperm(class_order.numel(), generator=generator)]
    cursor = 0
    while len(qsets) < int(cfg.max_queries) and cursor < max(int(cfg.max_queries) * 4, len(classes) * 2):
        c = int(class_order[cursor % class_order.numel()].item())
        pool = (labels == c).nonzero(as_tuple=False).flatten()
        if pool.numel() > 0:
            qsets.append(_choice(pool, int(cfg.query_seed_count), generator=generator, replace_if_needed=True))
        cursor += 1
    return tuple(qsets[: int(cfg.max_queries)])


def build_source_query_boundary_plan(
    *,
    labels: torch.Tensor | Sequence[int],
    graph_name: str = "source",
    query_node_sets: Sequence[Sequence[int] | torch.Tensor] | None = None,
    edge_index: torch.Tensor | Sequence[Sequence[int]] | None = None,
    node_features: torch.Tensor | None = None,
    current_scores: Mapping[int, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor | None = None,
    config: SourceQueryBoundarySamplerConfig | None = None,
    split: str = "source_train",
) -> SourceQueryBoundaryPlan:
    """Build bounded source-only query-boundary samples.

    ``current_scores`` is optional and may contain source-only query score vectors
    from the current model.  It is used only to pick low-score positives and
    high-score negatives.
    """

    cfg = (config or SourceQueryBoundarySamplerConfig()).validate()
    if not str(split).startswith("source"):
        raise V5DataLeakageError("query-boundary plan can only be built for source splits")
    label_t = _labels_to_cpu(labels)
    n = int(label_t.numel())
    edge_pairs = _edge_index_to_pairs(edge_index, num_nodes=n)
    boundary_pos_by_label, boundary_neg_by_label = _boundary_nodes(label_t, edge_pairs)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.random_seed))
    if query_node_sets is None:
        qsets = _make_query_sets_from_labels(label_t, cfg=cfg, generator=generator)
    else:
        qsets = tuple(
            tuple(dict.fromkeys(int(x) for x in (q.detach().cpu().flatten().tolist() if isinstance(q, torch.Tensor) else q)))
            for q in query_node_sets
        )[: int(cfg.max_queries)]
    if not qsets:
        raise V5ProtocolError("no query sets available for source query-boundary plan")

    def _score_for(qi: int) -> torch.Tensor | None:
        if current_scores is None:
            return None
        if isinstance(current_scores, torch.Tensor):
            if current_scores.ndim == 1:
                return current_scores.detach().cpu().float()
            return current_scores[qi].detach().cpu().float()
        if isinstance(current_scores, Mapping):
            score = current_scores.get(qi)
            return None if score is None else score.detach().cpu().float()
        if qi < len(current_scores):
            return current_scores[qi].detach().cpu().float()
        return None

    samples: list[QueryBoundarySample] = []
    stats = {
        "query_count_requested": len(qsets),
        "samples_built": 0,
        "used_score_aware_sampling": 0,
        "used_boundary_edges": 0,
        "used_feature_similar_negatives": 0,
    }

    for qi, qset in enumerate(qsets):
        if not qset:
            continue
        if min(qset) < 0 or max(qset) >= n:
            raise V5ProtocolError(f"query set outside [0,{n}): {qset}")
        q_label = int(label_t[int(qset[0])].item())
        # Require all query seeds to share the same source community label.
        if any(int(label_t[int(x)].item()) != q_label for x in qset):
            continue
        same_pool = (label_t == q_label).nonzero(as_tuple=False).flatten()
        diff_pool = (label_t != q_label).nonzero(as_tuple=False).flatten()
        if same_pool.numel() == 0 or diff_pool.numel() == 0:
            continue

        q_tensor = torch.tensor(qset, dtype=torch.long)
        positive_pool = same_pool[~torch.isin(same_pool, q_tensor)]
        if positive_pool.numel() == 0:
            positive_pool = same_pool
        negative_pool = diff_pool

        score = _score_for(qi)
        if score is not None:
            if score.numel() != n:
                raise V5ProtocolError("current score vector length must equal num_nodes")
            low_pos = _topk_from_scores(positive_pool, score, cfg.low_score_positive_count, largest=False)
            high_neg = _topk_from_scores(negative_pool, score, cfg.hard_negative_count, largest=True)
            stats["used_score_aware_sampling"] += 1
        else:
            low_pos = tuple()
            high_neg = tuple()

        bd_pos_pool = torch.tensor(sorted(boundary_pos_by_label.get(q_label, set())), dtype=torch.long)
        if bd_pos_pool.numel() > 0:
            bd_pos_pool = bd_pos_pool[torch.isin(bd_pos_pool, same_pool)]
        bd_neg_pool = torch.tensor(sorted(boundary_neg_by_label.get(q_label, set())), dtype=torch.long)
        if bd_neg_pool.numel() > 0:
            bd_neg_pool = bd_neg_pool[torch.isin(bd_neg_pool, diff_pool)]
        if bd_pos_pool.numel() > 0 or bd_neg_pool.numel() > 0:
            stats["used_boundary_edges"] += 1

        bd_pos = _choice(bd_pos_pool if bd_pos_pool.numel() > 0 else positive_pool, cfg.boundary_positive_count, generator=generator)
        bd_neg = _choice(bd_neg_pool if bd_neg_pool.numel() > 0 else negative_pool, cfg.boundary_negative_count, generator=generator)
        feat_neg = _feature_similar_negatives(
            node_features=node_features,
            query_ids=qset,
            negative_pool=negative_pool,
            count=cfg.hard_negative_count,
            max_pool=cfg.max_feature_pool,
            generator=generator,
        ) if node_features is not None else tuple()
        if feat_neg:
            stats["used_feature_similar_negatives"] += 1

        positives = tuple(dict.fromkeys(low_pos + bd_pos + _choice(positive_pool, cfg.positive_count, generator=generator)))
        negatives = tuple(dict.fromkeys(high_neg + bd_neg + feat_neg + _choice(negative_pool, cfg.negative_count, generator=generator)))
        if not positives or not negatives:
            continue
        samples.append(
            QueryBoundarySample(
                query_local_ids=qset,
                positive_local_ids=positives[: max(cfg.positive_count, cfg.low_score_positive_count, cfg.boundary_positive_count) * 3],
                negative_local_ids=negatives[: max(cfg.negative_count, cfg.hard_negative_count, cfg.boundary_negative_count) * 4],
                boundary_positive_local_ids=bd_pos,
                boundary_negative_local_ids=bd_neg,
                high_score_negative_local_ids=high_neg,
                low_score_positive_local_ids=low_pos,
                graph_name=graph_name,
                source_only=True,
                sample_reason_stats={
                    "positive_pool_size": int(positive_pool.numel()),
                    "negative_pool_size": int(negative_pool.numel()),
                    "boundary_positive_pool_size": int(bd_pos_pool.numel()),
                    "boundary_negative_pool_size": int(bd_neg_pool.numel()),
                },
            )
        )
        stats["samples_built"] += 1

    if not samples:
        raise V5ProtocolError("failed to build any source query-boundary samples")

    return SourceQueryBoundaryPlan(
        graph_name=graph_name,
        samples=tuple(samples),
        config=cfg,
        report=stats,
        source_only=True,
        labels_used=True,
        unseen_labels_used=False,
        ground_truth_used=False,
        metrics_used=False,
        full_cartesian_used=False,
    )
