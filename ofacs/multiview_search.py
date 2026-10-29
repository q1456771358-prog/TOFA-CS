"""Label-free multi-view GlobalSearch candidate selection with diversity diagnostics.

This module is deliberately target-label-free: it reads query nodes, predicted
candidate sets, score ranks, and optional target adjacency only.  It never reads
``*.gt`` files, target metrics, or oracle labels.  The goal is to make the
existing multi-view idea *actually multi-view*: candidate profiles must produce
measurably different communities, no-op duplicates are flagged, and per-query
normalised quality prevents one canonical view from winning only because of
raw score scale or tie ordering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .chunked_scoring import ChunkedScoreResult
from .community_output import CommunityPredictionRecord
from .errors import V5ProtocolError
from .senior_compat import senior_global_search_from_ranked_scores, subgraph_density_controlled

_SEARCH_PROFILES = {"none", "center", "zscore", "local_center", "temperature", "precision_profile", "recall_profile"}
_AFFINE_NOOP_PROFILES = {"center", "zscore"}


def _safe_float(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _jaccard(a: Sequence[int] | set[int], b: Sequence[int] | set[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _top_overlap(a: Sequence[int], b: Sequence[int], k: int) -> float:
    kk = max(1, int(k))
    sa = set(int(x) for x in list(a)[:kk])
    sb = set(int(x) for x in list(b)[:kk])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _entropy(values: Sequence[float], eps: float = 1e-12) -> float:
    if not values:
        return 0.0
    xs = torch.tensor([float(v) for v in values], dtype=torch.float64)
    xs = xs - xs.max()
    p = torch.softmax(xs, dim=0)
    ent = -(p * torch.log(p.clamp_min(float(eps)))).sum().item()
    return float(ent / math.log(max(2, len(values))))


def _minmax_norm(values: Sequence[float], *, higher_is_better: bool = True, eps: float = 1e-12) -> list[float]:
    vals = [_safe_float(v) for v in values]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo <= eps:
        return [0.5 for _ in vals]
    out = [(v - lo) / (hi - lo) for v in vals]
    if not higher_is_better:
        out = [1.0 - x for x in out]
    return [float(x) for x in out]


@dataclass(frozen=True)
class GraphStructure:
    """Small adjacency helper for label-free candidate structural quality."""

    adjacency: tuple[frozenset[int], ...]

    @property
    def num_nodes(self) -> int:
        return len(self.adjacency)

    @classmethod
    def from_edge_index(cls, edge_index: torch.Tensor | Sequence[Sequence[int]] | None, *, num_nodes: int) -> "GraphStructure | None":
        if edge_index is None:
            return None
        if isinstance(edge_index, torch.Tensor):
            ei = edge_index.detach().cpu().long()
            if ei.numel() == 0:
                return cls(tuple(frozenset() for _ in range(int(num_nodes))))
            if ei.ndim != 2:
                raise V5ProtocolError("edge_index must be [2,E] or [E,2]")
            if int(ei.shape[0]) == 2:
                src = ei[0].tolist()
                dst = ei[1].tolist()
            elif int(ei.shape[1]) == 2:
                src = ei[:, 0].tolist()
                dst = ei[:, 1].tolist()
            else:
                raise V5ProtocolError("edge_index must be [2,E] or [E,2]")
        else:
            rows = list(edge_index)
            if not rows:
                return cls(tuple(frozenset() for _ in range(int(num_nodes))))
            if len(rows) == 2 and all(not isinstance(x, int) for x in rows):
                src = [int(x) for x in rows[0]]
                dst = [int(x) for x in rows[1]]
            else:
                src = [int(x[0]) for x in rows]
                dst = [int(x[1]) for x in rows]
        adj: list[set[int]] = [set() for _ in range(int(num_nodes))]
        for u, v in zip(src, dst):
            u_i, v_i = int(u), int(v)
            if 0 <= u_i < int(num_nodes) and 0 <= v_i < int(num_nodes) and u_i != v_i:
                adj[u_i].add(v_i)
                adj[v_i].add(u_i)
        return cls(tuple(frozenset(x) for x in adj))

    def candidate_features(self, nodes: Sequence[int], query_nodes: Sequence[int]) -> dict[str, float]:
        cand = set(int(x) for x in nodes if 0 <= int(x) < self.num_nodes)
        if not cand:
            return {
                "internal_edge_density": 0.0,
                "conductance_penalty": 1.0,
                "boundary_cut_ratio": 1.0,
                "largest_component_ratio": 0.0,
                "query_connected_ratio": 0.0,
                "structural_quality": 0.0,
            }
        internal_twice = 0
        boundary = 0
        degree_sum = 0
        for u in cand:
            neigh = self.adjacency[u]
            degree_sum += len(neigh)
            inside = len(neigh & cand)
            internal_twice += inside
            boundary += len(neigh) - inside
        internal_edges = internal_twice // 2
        max_edges = max(1.0, float(len(cand) * (len(cand) - 1) / 2.0))
        internal_density = float(internal_edges) / max_edges
        denom = float(2 * internal_edges + boundary)
        conductance = float(boundary) / denom if denom > 0 else 1.0
        boundary_cut_ratio = float(boundary) / max(1.0, float(degree_sum))
        # Largest connected component and query-connected coverage.
        unseen = set(cand)
        lcc = 0
        query_connected: set[int] = set()
        qset = set(int(x) for x in query_nodes)
        while unseen:
            root = next(iter(unseen))
            stack = [root]
            comp = {root}
            unseen.remove(root)
            while stack:
                u = stack.pop()
                for v in self.adjacency[u] & cand:
                    if v in unseen:
                        unseen.remove(v)
                        comp.add(v)
                        stack.append(v)
            if len(comp) > lcc:
                lcc = len(comp)
            if comp & qset:
                query_connected |= comp
        lcc_ratio = float(lcc) / float(len(cand))
        query_connected_ratio = float(len(query_connected)) / float(len(cand))
        structural_quality = (
            0.30 * internal_density
            + 0.30 * lcc_ratio
            + 0.25 * query_connected_ratio
            + 0.15 * (1.0 - conductance)
        )
        return {
            "internal_edge_density": float(internal_density),
            "conductance_penalty": float(conductance),
            "boundary_cut_ratio": float(boundary_cut_ratio),
            "largest_component_ratio": float(lcc_ratio),
            "query_connected_ratio": float(query_connected_ratio),
            "structural_quality": float(structural_quality),
        }


@dataclass
class MultiViewCandidate:
    view_name: str
    base_mode: str
    search_profile: str
    query_index: int
    query_nodes: tuple[int, ...]
    predicted_nodes: tuple[int, ...]
    prefix_rank_count: int
    pred_size: int
    native_senior_density: float
    native_mean_selected: float
    native_boundary_gap: float
    native_size_penalty: float
    legacy_quality: float
    canonical_compactness: float = 0.0
    canonical_boundary_gap: float = 0.0
    canonical_density: float = 0.0
    size_penalty: float = 0.0
    two_sided_size_penalty: float = 0.0
    raw_guided_agreement: float = 0.0
    top_score_entropy: float = 0.0
    structural_features: dict[str, float] = field(default_factory=dict)
    duplicate_of_canonical: bool = False
    duplicate_reason: str = ""
    jaccard_with_canonical: float = 0.0
    jaccard_with_selected: float = 0.0
    topk_overlap_with_canonical: float = 0.0
    normalized_components: dict[str, float] = field(default_factory=dict)
    quality: float = 0.0
    selection_margin: float = 0.0
    selected: bool = False
    selected_reason: str = ""
    reject_reason: str = ""

    def to_safe_metadata(self) -> dict[str, object]:
        return {
            "multi_view_search": True,
            "selected_view": self.view_name,
            "mv_quality": float(self.quality),
            "mv_native_density": float(self.native_senior_density),
            "mv_canonical_density": float(self.canonical_density),
            "mv_compactness": float(self.canonical_compactness),
            "mv_boundary_gap": float(self.canonical_boundary_gap),
            "mv_size_penalty": float(self.size_penalty),
            "mv_two_sided_size_penalty": float(self.two_sided_size_penalty),
            "mv_structural_quality": float(self.structural_features.get("structural_quality", 0.0)),
            "mv_lcc_ratio": float(self.structural_features.get("largest_component_ratio", 0.0)),
            "mv_query_connected_ratio": float(self.structural_features.get("query_connected_ratio", 0.0)),
            "mv_raw_agreement": float(self.raw_guided_agreement),
            "mv_prefix_rank_count": int(self.prefix_rank_count),
            "mv_pred_size": int(self.pred_size),
            "mv_selection_margin": float(self.selection_margin),
            "mv_selected_reason": str(self.selected_reason),
            "labels_used": False,
            "ground_truth_used": False,
            "metrics_computed": False,
        }

    def to_diagnostic_row(self) -> dict[str, object]:
        row = {
            "query_index": int(self.query_index),
            "view_name": self.view_name,
            "base_mode": self.base_mode,
            "search_profile": self.search_profile,
            "selected": bool(self.selected),
            "selected_view": self.view_name if self.selected else "",
            "candidate_size": int(self.pred_size),
            "candidate_nodes_local": [int(x) for x in self.predicted_nodes],
            "query_nodes_local": [int(x) for x in self.query_nodes],
            "prefix_rank_count": int(self.prefix_rank_count),
            "quality": float(self.quality),
            "legacy_quality": float(self.legacy_quality),
            "selection_margin": float(self.selection_margin),
            "native_senior_density": float(self.native_senior_density),
            "native_mean_selected": float(self.native_mean_selected),
            "native_boundary_gap": float(self.native_boundary_gap),
            "canonical_density": float(self.canonical_density),
            "canonical_compactness": float(self.canonical_compactness),
            "canonical_boundary_gap": float(self.canonical_boundary_gap),
            "size_penalty": float(self.size_penalty),
            "two_sided_size_penalty": float(self.two_sided_size_penalty),
            "raw_guided_agreement": float(self.raw_guided_agreement),
            "top_score_entropy": float(self.top_score_entropy),
            "duplicate_of_canonical": bool(self.duplicate_of_canonical),
            "duplicate_reason": self.duplicate_reason,
            "jaccard_with_canonical": float(self.jaccard_with_canonical),
            "jaccard_with_selected": float(self.jaccard_with_selected),
            "topk_overlap_with_canonical": float(self.topk_overlap_with_canonical),
            "selected_reason": self.selected_reason,
            "reject_reason": self.reject_reason,
            "labels_used": False,
            "ground_truth_used": False,
            "metrics_computed": False,
        }
        for k, v in self.structural_features.items():
            row[str(k)] = float(v)
        for k, v in self.normalized_components.items():
            row[f"norm_{k}"] = float(v)
        return row


@dataclass(frozen=True)
class MultiViewSelectionConfig:
    selection: str = "diversity_hybrid"  # senior_density | hybrid | diversity_hybrid | structure_quality | selector-safe variants
    compactness_weight: float = 0.12
    boundary_gap_weight: float = 0.08
    size_penalty_weight: float = 0.12
    structural_quality_weight: float = 0.18
    lcc_weight: float = 0.08
    conductance_weight: float = 0.08
    raw_agreement_weight: float = 0.05
    diversity_bonus_weight: float = 0.05
    size_reference_power: float = 0.50
    min_size_reference: float = 20.0
    duplicate_jaccard_threshold: float = 0.98
    topk_overlap_k: int = 100
    canonical_view: str = "auto"
    normalize_quality_terms: bool = True
    canonical_dominance_guard: bool = True
    dominance_warning_threshold: float = 0.95
    temperature_value: float = 0.20
    local_center_top_fraction: float = 0.10
    precision_size_strength: float = 0.08
    recall_size_bonus: float = 0.04
    canonical_switch_margin: float = 0.08
    alternative_min_size_ratio: float = 0.85
    temperature_min_size_ratio: float = 0.85
    temperature_entropy_threshold: float = 0.82
    balanced_size_low: float = 0.85
    balanced_size_high: float = 1.25
    under_size_penalty_weight: float = 0.30
    temperature_penalty_weight: float = 0.35
    recall_view_bonus: float = 0.08
    eps: float = 1e-8

    def validate(self) -> "MultiViewSelectionConfig":
        if self.selection not in {"senior_density", "hybrid", "diversity_hybrid", "structure_quality", "canonical_guard", "no_temperature", "temperature_guard", "recall_safe", "balanced_size", "citeseer_safe_unsupervised", "structure_secondary"}:
            raise V5ProtocolError(f"unsupported multi-view selection={self.selection!r}")
        if self.eps <= 0:
            raise V5ProtocolError("multi-view eps must be positive")
        if self.min_size_reference <= 0:
            raise V5ProtocolError("multi-view min_size_reference must be positive")
        if self.size_reference_power < 0:
            raise V5ProtocolError("multi-view size_reference_power must be non-negative")
        if not (0.0 <= self.duplicate_jaccard_threshold <= 1.0):
            raise V5ProtocolError("duplicate_jaccard_threshold must be in [0,1]")
        if int(self.topk_overlap_k) <= 0:
            raise V5ProtocolError("topk_overlap_k must be positive")
        if float(self.temperature_value) <= 0:
            raise V5ProtocolError("temperature_value must be positive")
        for field_name in ("alternative_min_size_ratio", "temperature_min_size_ratio", "balanced_size_low", "balanced_size_high"):
            value = float(getattr(self, field_name))
            if value <= 0:
                raise V5ProtocolError(f"{field_name} must be positive")
        if float(self.balanced_size_low) > float(self.balanced_size_high):
            raise V5ProtocolError("balanced_size_low must be <= balanced_size_high")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate_prefix_count(pred: Sequence[int], query_nodes: Sequence[int], ranked_nodes: Sequence[int]) -> int:
    pred_set = set(int(x) for x in pred)
    current = set(int(x) for x in query_nodes)
    if pred_set <= current:
        return 0
    for k, node in enumerate(ranked_nodes, start=1):
        current.add(int(node))
        if pred_set <= current:
            return k
    return len(ranked_nodes)


def _profiled_scores(
    ranked_scores: list[float],
    q_scores: list[float],
    *,
    profile: str,
    full_score_sum: float,
    num_nodes: int,
    cfg: MultiViewSelectionConfig,
) -> tuple[list[float], list[float], float]:
    profile = str(profile or "none")
    if profile in {"none", "center", "zscore", "precision_profile", "recall_profile"}:
        return list(ranked_scores), list(q_scores), float(full_score_sum)
    if not ranked_scores:
        return list(ranked_scores), list(q_scores), float(full_score_sum)
    local_count = max(8, int(len(ranked_scores) * float(cfg.local_center_top_fraction)))
    local_count = min(len(ranked_scores), local_count)
    top = torch.tensor(ranked_scores[:local_count], dtype=torch.float64)
    mu = float(top.mean().item())
    std = float(top.std(unbiased=False).clamp_min(1e-6).item())
    if profile == "local_center":
        def f(x: float) -> float:
            z = (float(x) - mu) / std
            return z if z >= 0.0 else 1.50 * z
        rs = [f(x) for x in ranked_scores]
        qs = [f(x) for x in q_scores]
        full_mean = float(full_score_sum) / float(max(1, num_nodes))
        fs = f(full_mean) * float(num_nodes)
        return rs, qs, fs
    if profile == "temperature":
        temp = max(1e-4, float(cfg.temperature_value))
        def f(x: float) -> float:
            z = (float(x) - mu) / temp
            z = max(-40.0, min(40.0, z))
            return 1.0 / (1.0 + math.exp(-z))
        rs = [f(x) for x in ranked_scores]
        qs = [f(x) for x in q_scores]
        full_mean = float(full_score_sum) / float(max(1, num_nodes))
        fs = f(full_mean) * float(num_nodes)
        return rs, qs, fs
    raise V5ProtocolError(f"unsupported multi-view search profile={profile!r}")


def _choose_profile_prefix(
    *,
    q_scores: list[float],
    ranked_scores: list[float],
    full_score_sum: float,
    num_nodes: int,
    query_size: int,
    profile: str,
    cfg: MultiViewSelectionConfig,
) -> int:
    endpoint = min(len(ranked_scores), max(0, int(0.50 * int(num_nodes))), 10_000)
    if endpoint <= 0:
        return 0
    q_sum = float(sum(q_scores))
    prefix = [0.0]
    acc = 0.0
    for s in ranked_scores[:endpoint]:
        acc += float(s)
        prefix.append(acc)
    ref_size = max(float(cfg.min_size_reference), float(num_nodes) ** float(cfg.size_reference_power))
    best_k = 0
    best_val = float("-inf")
    for k in range(0, endpoint + 1):
        size = max(1, int(query_size) + int(k))
        density = subgraph_density_controlled(q_sum + prefix[k], size, full_score_sum=full_score_sum, num_nodes=num_nodes)
        if k > 0 and k < endpoint:
            gap = float(ranked_scores[k - 1] - ranked_scores[k])
        else:
            gap = 0.0
        over = max(0.0, math.log((float(size) + cfg.eps) / (ref_size + cfg.eps)))
        under = max(0.0, math.log((ref_size + cfg.eps) / (float(size) + cfg.eps)))
        value = density
        if profile == "precision_profile":
            value = density + 0.02 * gap - float(cfg.precision_size_strength) * over - 0.02 * math.log1p(k)
        elif profile == "recall_profile":
            value = density + float(cfg.recall_size_bonus) * math.log1p(k) - 0.03 * under - 0.01 * over
        # local_center/temperature use transformed density only.
        if value > best_val:
            best_val = value
            best_k = k
    return int(best_k)


def _candidate_from_result(
    *,
    view_name: str,
    base_mode: str,
    search_profile: str,
    score_result: ChunkedScoreResult,
    query_index: int,
    cfg: MultiViewSelectionConfig,
) -> MultiViewCandidate:
    qi = int(query_index)
    profile = str(search_profile or "none")
    if profile not in _SEARCH_PROFILES:
        raise V5ProtocolError(f"unsupported search_profile={profile!r}")
    qset = tuple(int(x) for x in score_result.query_node_sets[qi])
    ranked_nodes = [int(x) for x in score_result.top_nodes[qi].tolist()]
    ranked_scores_native = [float(x) for x in score_result.top_scores[qi].tolist()]
    q_scores_native = [float(score_result.query_seed_scores[qi][int(seed)]) for seed in qset]
    full_sum_native = float(score_result.full_score_sums[qi].item())
    n = int(score_result.num_nodes)

    ranked_scores, q_scores, full_sum = _profiled_scores(
        ranked_scores_native,
        q_scores_native,
        profile=profile,
        full_score_sum=full_sum_native,
        num_nodes=n,
        cfg=cfg,
    )
    if profile in {"none", "center", "zscore"}:
        pred = senior_global_search_from_ranked_scores(
            query_nodes=qset,
            query_scores=q_scores,
            ranked_nodes=ranked_nodes,
            ranked_scores=ranked_scores,
            num_nodes=n,
            full_score_sum=full_sum,
        )
        prefix_k = _candidate_prefix_count(pred, qset, ranked_nodes)
    else:
        prefix_k = _choose_profile_prefix(
            q_scores=q_scores,
            ranked_scores=ranked_scores,
            full_score_sum=full_sum,
            num_nodes=n,
            query_size=len(qset),
            profile=profile,
            cfg=cfg,
        )
        pred = tuple(dict.fromkeys(list(qset) + ranked_nodes[:prefix_k]))

    prefix_k = min(int(prefix_k), len(ranked_scores_native))
    cand_size = max(1, len(pred))
    # Native components are always computed in this view's native score space for diagnostics.
    native_sum = float(sum(q_scores_native) + sum(ranked_scores_native[:prefix_k]))
    native_density = subgraph_density_controlled(native_sum, cand_size, full_score_sum=full_sum_native, num_nodes=n)
    native_mean = native_sum / float(cand_size)
    if prefix_k > 0 and prefix_k < len(ranked_scores_native):
        native_gap = float(ranked_scores_native[prefix_k - 1] - ranked_scores_native[prefix_k])
    else:
        native_gap = 0.0
    ref_size = max(float(cfg.min_size_reference), float(n) ** float(cfg.size_reference_power))
    size_penalty = max(0.0, math.log((float(cand_size) + cfg.eps) / (ref_size + cfg.eps)))
    legacy_quality = native_density if cfg.selection == "senior_density" else (
        native_density
        + float(cfg.compactness_weight) * native_mean
        + float(cfg.boundary_gap_weight) * native_gap
        - float(cfg.size_penalty_weight) * size_penalty
    )
    return MultiViewCandidate(
        view_name=str(view_name),
        base_mode=str(base_mode),
        search_profile=profile,
        query_index=qi,
        query_nodes=qset,
        predicted_nodes=tuple(int(x) for x in pred),
        prefix_rank_count=int(prefix_k),
        pred_size=int(cand_size),
        native_senior_density=float(native_density),
        native_mean_selected=float(native_mean),
        native_boundary_gap=float(native_gap),
        native_size_penalty=float(size_penalty),
        legacy_quality=float(legacy_quality),
        top_score_entropy=_entropy(ranked_scores_native[: min(256, len(ranked_scores_native))]),
    )


def _score_lookup(result: ChunkedScoreResult, qi: int) -> tuple[dict[int, float], float, list[int], list[float]]:
    ranked_nodes = [int(x) for x in result.top_nodes[int(qi)].tolist()]
    ranked_scores = [float(x) for x in result.top_scores[int(qi)].tolist()]
    lookup = {int(n): float(s) for n, s in zip(ranked_nodes, ranked_scores)}
    for seed, value in result.query_seed_scores[int(qi)].items():
        lookup[int(seed)] = float(value)
    tail = ranked_scores[-1] if ranked_scores else 0.0
    return lookup, float(tail), ranked_nodes, ranked_scores


def _canonical_components(candidate: MultiViewCandidate, canonical_result: ChunkedScoreResult, *, cfg: MultiViewSelectionConfig) -> tuple[float, float, float]:
    qi = candidate.query_index
    lookup, tail, canonical_ranked, canonical_scores = _score_lookup(canonical_result, qi)
    q_scores = [lookup.get(int(seed), tail) for seed in candidate.query_nodes]
    cand_scores = [lookup.get(int(node), tail) for node in candidate.predicted_nodes if int(node) not in set(candidate.query_nodes)]
    total = float(sum(q_scores) + sum(cand_scores))
    size = max(1, int(candidate.pred_size))
    compactness = total / float(size)
    density = subgraph_density_controlled(
        total,
        size,
        full_score_sum=float(canonical_result.full_score_sums[qi].item()),
        num_nodes=int(canonical_result.num_nodes),
    )
    # Boundary gap under canonical rank at the first node outside candidate after scanning.
    cand_set = set(int(x) for x in candidate.predicted_nodes)
    last_in = None
    first_out = None
    for node, sc in zip(canonical_ranked, canonical_scores):
        if int(node) in cand_set:
            last_in = float(sc)
        elif last_in is not None:
            first_out = float(sc)
            break
    gap = float(last_in - first_out) if last_in is not None and first_out is not None else 0.0
    return float(compactness), float(gap), float(density)


def _select_canonical_name(items: list[tuple[str, ChunkedScoreResult]], cfg: MultiViewSelectionConfig) -> str:
    names = [name for name, _ in items]
    requested = str(cfg.canonical_view or "auto")
    if requested != "auto":
        if requested not in names:
            raise V5ProtocolError(f"requested canonical view {requested!r} not found among {names}")
        return requested
    for preferred in ("guided_refined__none", "refined__none", "raw__none"):
        if preferred in names:
            return preferred
    for name in names:
        if name.endswith("__none"):
            return name
    return names[0]


def _assign_quality(
    candidates: list[MultiViewCandidate],
    *,
    canonical: MultiViewCandidate,
    canonical_result: ChunkedScoreResult,
    rawplus_candidate: MultiViewCandidate | None,
    graph_structure: GraphStructure | None,
    cfg: MultiViewSelectionConfig,
) -> None:
    sizes = [c.pred_size for c in candidates]
    consensus = float(statistics.median(sizes)) if sizes else 1.0
    ref_global = max(float(cfg.min_size_reference), float(canonical_result.num_nodes) ** float(cfg.size_reference_power))
    ref_size = max(1.0, 0.5 * consensus + 0.5 * ref_global)
    for c in candidates:
        c.jaccard_with_canonical = _jaccard(c.predicted_nodes, canonical.predicted_nodes)
        c.duplicate_of_canonical = (c.view_name != canonical.view_name and c.jaccard_with_canonical >= float(cfg.duplicate_jaccard_threshold) and c.pred_size == canonical.pred_size)
        if c.duplicate_of_canonical:
            if c.search_profile in _AFFINE_NOOP_PROFILES:
                c.duplicate_reason = f"{c.search_profile}_affine_noop_same_candidate"
            else:
                c.duplicate_reason = "same_as_canonical_candidate"
        c.canonical_compactness, c.canonical_boundary_gap, c.canonical_density = _canonical_components(c, canonical_result, cfg=cfg)
        c.size_penalty = abs(math.log((float(c.pred_size) + cfg.eps) / (ref_size + cfg.eps)))
        c.two_sided_size_penalty = c.size_penalty
        if rawplus_candidate is not None:
            c.raw_guided_agreement = _jaccard(c.predicted_nodes, rawplus_candidate.predicted_nodes)
        else:
            c.raw_guided_agreement = 0.0
        if graph_structure is not None:
            c.structural_features = graph_structure.candidate_features(c.predicted_nodes, c.query_nodes)
        else:
            c.structural_features = {
                "internal_edge_density": 0.0,
                "conductance_penalty": 0.5,
                "boundary_cut_ratio": 0.5,
                "largest_component_ratio": 0.0,
                "query_connected_ratio": 0.0,
                "structural_quality": 0.0,
            }
        _, _, canon_ranked, _ = _score_lookup(canonical_result, c.query_index)
        # Compare view top-K against canonical top-K when possible.  The view-specific
        # top nodes are not stored in the candidate, so this is filled later by caller
        # through candidate-level metadata if available; default to candidate set overlap.
        c.topk_overlap_with_canonical = _jaccard(c.predicted_nodes, canonical.predicted_nodes)

    if cfg.selection == "senior_density":
        vals = [c.native_senior_density for c in candidates]
        for c, q in zip(candidates, vals):
            c.quality = float(q)
            c.normalized_components = {"native_density": float(q)}
        return
    if cfg.selection == "hybrid":
        for c in candidates:
            c.quality = float(c.legacy_quality)
            c.normalized_components = {"legacy_quality": float(c.legacy_quality)}
        return

    components = {
        "canonical_density": _minmax_norm([c.canonical_density for c in candidates], higher_is_better=True),
        "compactness": _minmax_norm([c.canonical_compactness for c in candidates], higher_is_better=True),
        "boundary_gap": _minmax_norm([c.canonical_boundary_gap for c in candidates], higher_is_better=True),
        "size_quality": _minmax_norm([c.two_sided_size_penalty for c in candidates], higher_is_better=False),
        "structural_quality": _minmax_norm([c.structural_features.get("structural_quality", 0.0) for c in candidates], higher_is_better=True),
        "lcc": _minmax_norm([c.structural_features.get("largest_component_ratio", 0.0) for c in candidates], higher_is_better=True),
        "conductance_quality": _minmax_norm([c.structural_features.get("conductance_penalty", 0.5) for c in candidates], higher_is_better=False),
        "raw_agreement": _minmax_norm([c.raw_guided_agreement for c in candidates], higher_is_better=True),
        "diversity": [0.0 if c.duplicate_of_canonical else max(0.0, 1.0 - c.jaccard_with_canonical) for c in candidates],
    }
    for idx, c in enumerate(candidates):
        c.normalized_components = {k: float(v[idx]) for k, v in components.items()}
        if cfg.selection == "structure_quality":
            q = (
                float(cfg.structural_quality_weight) * c.normalized_components["structural_quality"]
                + float(cfg.lcc_weight) * c.normalized_components["lcc"]
                + float(cfg.conductance_weight) * c.normalized_components["conductance_quality"]
                + float(cfg.size_penalty_weight) * c.normalized_components["size_quality"]
                + float(cfg.diversity_bonus_weight) * c.normalized_components["diversity"]
            )
        elif cfg.selection == "structure_secondary":
            q = (
                0.18 * c.normalized_components["canonical_density"]
                + 0.10 * c.normalized_components["compactness"]
                + 0.10 * c.normalized_components["boundary_gap"]
                + 0.22 * c.normalized_components["size_quality"]
                + 0.08 * c.normalized_components["raw_agreement"]
                + 0.04 * c.normalized_components["diversity"]
                + 0.03 * c.normalized_components["structural_quality"]
            )
            c.normalized_components["structure_secondary_note"] = 1.0
        else:
            q = (
                0.12 * c.normalized_components["canonical_density"]
                + float(cfg.compactness_weight) * c.normalized_components["compactness"]
                + float(cfg.boundary_gap_weight) * c.normalized_components["boundary_gap"]
                + float(cfg.size_penalty_weight) * c.normalized_components["size_quality"]
                + float(cfg.structural_quality_weight) * c.normalized_components["structural_quality"]
                + float(cfg.lcc_weight) * c.normalized_components["lcc"]
                + float(cfg.conductance_weight) * c.normalized_components["conductance_quality"]
                + float(cfg.raw_agreement_weight) * c.normalized_components["raw_agreement"]
                + float(cfg.diversity_bonus_weight) * c.normalized_components["diversity"]
            )
            # Do not let canonical re-score one-vote veto a genuinely different
            # alternative.  A non-duplicate candidate with better structure/size
            # gets a small acceptance bonus, recorded in diagnostics.
            if cfg.canonical_dominance_guard and c.view_name != canonical.view_name and not c.duplicate_of_canonical:
                if c.normalized_components["structural_quality"] >= 0.75 or c.normalized_components["size_quality"] >= 0.75:
                    q += 0.03
                    c.normalized_components["canonical_dominance_guard_bonus"] = 0.03
        # Recall-safe selector variants.  These are label-free safeguards against
        # the EXP02 failure mode where small/sharp temperature candidates won on
        # compactness while harming Citeseer-style hard-positive recall.
        size_ratio = float(c.pred_size) / float(max(1, canonical.pred_size))
        c.normalized_components["size_ratio_to_canonical"] = float(size_ratio)
        is_temperature = c.search_profile == "temperature" or "__temperature" in c.view_name
        is_raw_recall = ("raw_plus" in c.view_name) or c.search_profile == "recall_profile" or c.view_name.endswith("__recall_profile")
        high_entropy_or_weak_gap = (float(c.top_score_entropy) >= float(cfg.temperature_entropy_threshold)) or (float(canonical.canonical_boundary_gap) <= 0.0)
        if cfg.selection == "no_temperature" and is_temperature:
            q -= 10.0
            c.normalized_components["temperature_disabled"] = 1.0
        if cfg.selection in {"temperature_guard", "citeseer_safe_unsupervised"} and is_temperature:
            min_ratio = float(cfg.temperature_min_size_ratio)
            if size_ratio < min_ratio:
                q -= float(cfg.temperature_penalty_weight) * (min_ratio - size_ratio + 1.0)
                c.normalized_components["temperature_under_size_penalty"] = float(min_ratio - size_ratio)
            if cfg.selection == "citeseer_safe_unsupervised" and high_entropy_or_weak_gap:
                q -= 0.15
                c.normalized_components["temperature_high_entropy_penalty"] = 0.15
        if cfg.selection in {"recall_safe", "citeseer_safe_unsupervised"}:
            if size_ratio < float(cfg.alternative_min_size_ratio):
                q -= float(cfg.under_size_penalty_weight) * (float(cfg.alternative_min_size_ratio) - size_ratio + 1.0)
                c.normalized_components["under_size_recall_penalty"] = float(float(cfg.alternative_min_size_ratio) - size_ratio)
            if is_raw_recall:
                q += float(cfg.recall_view_bonus)
                c.normalized_components["recall_view_bonus"] = float(cfg.recall_view_bonus)
        if cfg.selection == "balanced_size":
            lo, hi = float(cfg.balanced_size_low), float(cfg.balanced_size_high)
            if size_ratio < lo:
                q -= float(cfg.under_size_penalty_weight) * (lo - size_ratio + 1.0)
                c.normalized_components["balanced_under_size_penalty"] = float(lo - size_ratio)
            elif size_ratio > hi:
                q -= 0.15 * (size_ratio - hi + 1.0)
                c.normalized_components["balanced_over_size_penalty"] = float(size_ratio - hi)
            else:
                q += 0.05
                c.normalized_components["balanced_size_band_bonus"] = 0.05
        c.quality = float(q)


def _select_candidate(candidates: list[MultiViewCandidate], canonical: MultiViewCandidate, *, cfg: MultiViewSelectionConfig) -> MultiViewCandidate:
    ranked = sorted(
        candidates,
        key=lambda c: (
            c.quality,
            0 if c.duplicate_of_canonical else 1,
            c.structural_features.get("structural_quality", 0.0),
            -abs(c.pred_size - canonical.pred_size),
            -c.pred_size,
        ),
        reverse=True,
    )
    proposed = ranked[0]
    guard_reason = ""
    if cfg.selection == "canonical_guard" and proposed.view_name != canonical.view_name:
        size_ratio = float(proposed.pred_size) / float(max(1, canonical.pred_size))
        quality_margin = float(proposed.quality - canonical.quality)
        if quality_margin < float(cfg.canonical_switch_margin):
            guard_reason = f"canonical_guard_reject_margin_{quality_margin:.4f}_lt_{float(cfg.canonical_switch_margin):.4f}"
            proposed = canonical
        elif size_ratio < float(cfg.alternative_min_size_ratio):
            guard_reason = f"canonical_guard_reject_size_ratio_{size_ratio:.4f}_lt_{float(cfg.alternative_min_size_ratio):.4f}"
            proposed = canonical
    best = proposed
    second = next((c for c in ranked if c is not best), None)
    margin = float(best.quality - (second.quality if second is not None else 0.0))
    if guard_reason:
        best.selected_reason = guard_reason
    elif second is not None and abs(best.quality - second.quality) <= 1e-10:
        best.selected_reason = "tie_broken_by_nonduplicate_structure_size"
    elif best.duplicate_of_canonical and best.view_name != canonical.view_name:
        best.selected_reason = "selected_duplicate_unexpected"
    elif best.view_name == canonical.view_name:
        if all(c.duplicate_of_canonical or c.view_name == canonical.view_name for c in candidates):
            best.selected_reason = "canonical_selected_all_alternatives_duplicate"
        else:
            best.selected_reason = "canonical_selected_guard_or_best_quality" if cfg.selection == "canonical_guard" else "canonical_selected_best_normalized_quality"
    else:
        best.selected_reason = "alternative_selected_best_recall_safe_quality" if cfg.selection in {"no_temperature", "temperature_guard", "recall_safe", "balanced_size", "citeseer_safe_unsupervised", "structure_secondary"} else "alternative_selected_best_normalized_quality"
    for c in candidates:
        c.selected = c is best
        c.selection_margin = margin if c is best else float(best.quality - c.quality)
        c.jaccard_with_selected = _jaccard(c.predicted_nodes, best.predicted_nodes)
        if c is not best:
            if guard_reason and c is ranked[0]:
                c.reject_reason = guard_reason
            elif c.duplicate_of_canonical:
                c.reject_reason = c.duplicate_reason or "duplicate_candidate"
            elif c.quality < best.quality:
                c.reject_reason = "lower_recall_safe_quality" if cfg.selection in {"no_temperature", "temperature_guard", "recall_safe", "balanced_size", "citeseer_safe_unsupervised", "structure_secondary"} else "lower_normalized_quality"
            else:
                c.reject_reason = "tie_break_lost"
    return best


def _write_candidate_diagnostics(path: str | Path, by_query: Sequence[Sequence[MultiViewCandidate]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for candidates in by_query:
            selected = next((c for c in candidates if c.selected), None)
            row = {
                "query_index": int(candidates[0].query_index) if candidates else -1,
                "selected_view": selected.view_name if selected is not None else "",
                "selected_reason": selected.selected_reason if selected is not None else "",
                "selection_margin": float(selected.selection_margin) if selected is not None else 0.0,
                "labels_used": False,
                "ground_truth_used": False,
                "metrics_computed": False,
                "candidates": [c.to_diagnostic_row() for c in candidates],
            }
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def _write_summary(path: str | Path, by_query: Sequence[Sequence[MultiViewCandidate]], *, cfg: MultiViewSelectionConfig) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_counts: dict[str, int] = {}
    sizes: dict[str, list[int]] = {}
    qualities: dict[str, list[float]] = {}
    all_pair_j: list[float] = []
    distinct_flags: list[float] = []
    for candidates in by_query:
        selected = next((c for c in candidates if c.selected), None)
        if selected is not None:
            selected_counts[selected.view_name] = selected_counts.get(selected.view_name, 0) + 1
        for c in candidates:
            sizes.setdefault(c.view_name, []).append(int(c.pred_size))
            qualities.setdefault(c.view_name, []).append(float(c.quality))
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                jj = _jaccard(candidates[i].predicted_nodes, candidates[j].predicted_nodes)
                all_pair_j.append(jj)
                distinct_flags.append(1.0 if jj < 0.95 else 0.0)
    q_count = max(1, len(by_query))
    selected_distribution = {k: float(v) / float(q_count) for k, v in sorted(selected_counts.items())}
    warning = None
    if selected_distribution:
        top_view, top_frac = max(selected_distribution.items(), key=lambda kv: kv[1])
        if top_frac >= float(cfg.dominance_warning_threshold):
            warning = f"dominance_warning: {top_view} selected {top_frac:.3f}; multi-view may be degenerate unless alternatives are duplicates"
    payload = {
        "selected_view_distribution": selected_distribution,
        "pairwise_candidate_jaccard_mean": float(statistics.mean(all_pair_j)) if all_pair_j else None,
        "pairwise_candidate_jaccard_median": float(statistics.median(all_pair_j)) if all_pair_j else None,
        "distinct_candidate_ratio_jaccard_lt_0_95": float(statistics.mean(distinct_flags)) if distinct_flags else None,
        "view_avg_candidate_size": {k: float(statistics.mean(v)) for k, v in sorted(sizes.items())},
        "view_avg_quality": {k: float(statistics.mean(v)) for k, v in sorted(qualities.items())},
        "dominance_warning": warning,
        "labels_used": False,
        "ground_truth_used": False,
        "metrics_computed": False,
        "selection_config": cfg.to_dict(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if warning:
        print(warning)
    return out_path


def build_prediction_records_from_multiview_scores(
    *,
    graph_name: str,
    score_results: Mapping[str, ChunkedScoreResult],
    global_node_ids: Sequence[int] | None = None,
    config: MultiViewSelectionConfig | None = None,
    view_profiles: Mapping[str, tuple[str, str]] | None = None,
    edge_index: torch.Tensor | Sequence[Sequence[int]] | None = None,
    candidate_diagnostics_path: str | Path | None = None,
    candidate_summary_path: str | Path | None = None,
) -> tuple[CommunityPredictionRecord, ...]:
    cfg = (config or MultiViewSelectionConfig()).validate()
    if not score_results:
        raise V5ProtocolError("multi-view search requires at least one score result")
    items = list(score_results.items())
    first = items[0][1]
    q_count = len(first.query_node_sets)
    n = int(first.num_nodes)
    for name, result in items:
        if int(result.num_nodes) != n:
            raise V5ProtocolError(f"multi-view score result {name!r} has num_nodes mismatch")
        if tuple(result.query_node_sets) != tuple(first.query_node_sets):
            raise V5ProtocolError(f"multi-view score result {name!r} has query mismatch")
    canonical_name = _select_canonical_name(items, cfg)
    canonical_result = score_results[canonical_name]
    graph_structure = GraphStructure.from_edge_index(edge_index, num_nodes=n) if edge_index is not None else None

    profiles = dict(view_profiles or {})
    records: list[CommunityPredictionRecord] = []
    diagnostics: list[list[MultiViewCandidate]] = []
    for qi in range(q_count):
        candidates: list[MultiViewCandidate] = []
        for name, result in items:
            base_mode, profile = profiles.get(name, (name.split("__", 1)[0], name.split("__", 1)[1] if "__" in name else "none"))
            candidates.append(
                _candidate_from_result(
                    view_name=name,
                    base_mode=base_mode,
                    search_profile=profile,
                    score_result=result,
                    query_index=qi,
                    cfg=cfg,
                )
            )
        canonical = next((c for c in candidates if c.view_name == canonical_name), candidates[0])
        rawplus = next((c for c in candidates if "raw_plus" in c.view_name or "adaptive_raw_plus" in c.view_name), None)
        _assign_quality(
            candidates,
            canonical=canonical,
            canonical_result=canonical_result,
            rawplus_candidate=rawplus,
            graph_structure=graph_structure,
            cfg=cfg,
        )
        # Fill top-k overlap with canonical using each view's ranked nodes.
        canonical_nodes = [int(x) for x in canonical_result.top_nodes[qi].tolist()]
        for c in candidates:
            result = score_results.get(c.view_name)
            if result is not None:
                nodes = [int(x) for x in result.top_nodes[qi].tolist()]
                c.topk_overlap_with_canonical = _top_overlap(nodes, canonical_nodes, int(cfg.topk_overlap_k))
        best = _select_candidate(candidates, canonical, cfg=cfg)
        query_global = None
        pred_global = None
        if global_node_ids is not None:
            query_global = tuple(int(global_node_ids[int(i)]) for i in best.query_nodes)
            pred_global = tuple(int(global_node_ids[int(i)]) for i in best.predicted_nodes)
        metadata = best.to_safe_metadata()
        metadata["mv_candidate_count"] = len(candidates)
        metadata["mv_candidate_views"] = [c.view_name for c in candidates]
        metadata["mv_selection"] = str(cfg.selection)
        metadata["mv_canonical_view"] = canonical_name
        metadata["mv_duplicate_alt_count"] = sum(1 for c in candidates if c.duplicate_of_canonical)
        records.append(
            CommunityPredictionRecord(
                graph_name=graph_name,
                query_index=int(qi),
                query_nodes_local=best.query_nodes,
                predicted_nodes_local=best.predicted_nodes,
                query_nodes_global=query_global,
                predicted_nodes_global=pred_global,
                metadata=metadata,
            )
        )
        diagnostics.append(candidates)
    if candidate_diagnostics_path is not None:
        _write_candidate_diagnostics(candidate_diagnostics_path, diagnostics)
    if candidate_summary_path is not None:
        _write_summary(candidate_summary_path, diagnostics, cfg=cfg)
    return tuple(records)
