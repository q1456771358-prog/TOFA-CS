"""Controlled source-supervised community error-state synthesis (CDSS).

This module is source-only. It uses labeled source communities to construct
controlled retrieval-error states without consulting target labels or target
community ground truth.

Node selection is frozen from a detached pre-optimization reference embedding.
The expensive selection plan can be built once per source graph and reused in
all training steps. Current embeddings are used only for differentiable
prototype/geometry computation, separating error-state augmentation from moving
hard-mining dynamics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

_STATE_ORDER = ("under", "moderate", "boundary", "stable")


def _as_sequence(value) -> list:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    return list(value)


def _stable_offset(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % modulo


def _finite_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    return result if math.isfinite(result) else 0.0


@dataclass(frozen=True)
class ControlledStateSynthesisConfig:
    temperature: float = 0.10
    core_topk: int = 64
    boundary_pos_topk: int = 64
    hard_neg_topk: int = 64
    under_recall: float = 0.40
    moderate_recall: float = 0.75
    boundary_precision: float = 0.60
    stable_recall: float = 0.95
    state_schedule: str = "balanced_cycle"
    enabled_states: tuple[str, ...] = _STATE_ORDER
    aggregation: str = "query_mean"
    balance_min: float = 0.50
    balance_max: float = 2.00
    distance_mode: str = "euclidean"
    separation_mode: str = "absolute"
    absolute_margin: float = 0.05
    absolute_separation_cap: float = 5.0
    relative_margin: float = 0.10
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        for name in ("core_topk", "boundary_pos_topk", "hard_neg_topk"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("under_recall", "moderate_recall", "stable_recall", "boundary_precision"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not self.under_recall < self.moderate_recall <= self.stable_recall:
            raise ValueError("expected under_recall < moderate_recall <= stable_recall")
        if self.state_schedule != "balanced_cycle":
            raise ValueError(f"unsupported state_schedule={self.state_schedule!r}")
        if self.aggregation not in {"query_mean", "capped_state_balance"}:
            raise ValueError(f"unsupported aggregation={self.aggregation!r}")
        if self.distance_mode not in {"euclidean", "cosine"}:
            raise ValueError(f"unsupported distance_mode={self.distance_mode!r}")
        if self.separation_mode not in {"absolute", "relative"}:
            raise ValueError(f"unsupported separation_mode={self.separation_mode!r}")
        states = tuple(str(x) for x in self.enabled_states)
        if not states or any(x not in _STATE_ORDER for x in states):
            raise ValueError(f"enabled_states must be a non-empty subset of {_STATE_ORDER}")
        if self.balance_min <= 0 or self.balance_max < self.balance_min:
            raise ValueError("invalid balance clip")
        if self.absolute_separation_cap <= 0:
            raise ValueError("absolute_separation_cap must be > 0")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")


@dataclass
class ControlledStateSynthesisReport:
    valid_queries: int = 0
    skipped_queries: int = 0
    state_counts: dict[str, int] = field(default_factory=lambda: {x: 0 for x in _STATE_ORDER})
    state_mean_losses: dict[str, float] = field(default_factory=lambda: {x: 0.0 for x in _STATE_ORDER})
    state_balance_weights: dict[str, float] = field(default_factory=lambda: {x: 0.0 for x in _STATE_ORDER})
    state_mean_synthetic_recall: dict[str, float] = field(default_factory=lambda: {x: 0.0 for x in _STATE_ORDER})
    state_mean_synthetic_precision: dict[str, float] = field(default_factory=lambda: {x: 0.0 for x in _STATE_ORDER})
    state_mean_selected_positive_nodes: dict[str, float] = field(default_factory=lambda: {x: 0.0 for x in _STATE_ORDER})
    state_mean_selected_negative_nodes: dict[str, float] = field(default_factory=lambda: {x: 0.0 for x in _STATE_ORDER})
    mean_positive_distance: float = 0.0
    mean_negative_distance: float = 0.0
    mean_ratio: float = 0.0
    mean_separation: float = 0.0
    separation_active_rate: float = 0.0
    mean_core_nodes: float = 0.0
    mean_selected_positive_nodes: float = 0.0
    mean_selected_negative_nodes: float = 0.0
    reference_requires_grad: bool = False
    source_only: bool = True
    target_labels_used: bool = False
    target_gt_used: bool = False
    selection_plan_reused: bool = False
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _state_for_query(*, query_index: int, schedule_step: int, graph_name: str, states: Sequence[str]) -> str:
    offset = _stable_offset(graph_name, len(states))
    return states[(int(query_index) + int(schedule_step) + offset) % len(states)]


def _rank_reference_nodes(normalized_reference: torch.Tensor, query_nodes: Sequence[int]) -> torch.Tensor:
    if query_nodes:
        q_idx = torch.as_tensor(list(query_nodes), device=normalized_reference.device, dtype=torch.long)
        query_vec = F.normalize(
            normalized_reference[q_idx].mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-12
        )
    else:
        query_vec = F.normalize(
            normalized_reference.mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-12
        )
    return (normalized_reference * query_vec).sum(dim=-1).detach()


def _positive_subset_for_state(
    state: str,
    positive_ranked_desc: torch.Tensor,
    *,
    config: ControlledStateSynthesisConfig,
) -> torch.Tensor:
    count = int(positive_ranked_desc.numel())
    # A valid prototype-boundary pair needs at least one positive kept for the
    # core and one disjoint positive used as the recovery/boundary example.
    if count < 2:
        return positive_ranked_desc[:0]
    if state == "under":
        retained = min(count - 1, max(1, int(round(count * config.under_recall))))
        selected = positive_ranked_desc[retained:]
    elif state == "moderate":
        retained = min(count - 1, max(1, int(round(count * config.moderate_recall))))
        selected = positive_ranked_desc[retained:]
    elif state == "stable":
        retained = min(count - 1, max(1, int(round(count * config.stable_recall))))
        selected = positive_ranked_desc[retained:]
    else:
        selected_count = min(count - 1, int(config.boundary_pos_topk))
        selected = positive_ranked_desc[count - selected_count :]
    if selected.numel() == 0:
        selected = positive_ranked_desc[-1:]
    return selected[: int(config.boundary_pos_topk)]


def _desired_negative_count(state: str, *, gt_size: int, config: ControlledStateSynthesisConfig) -> int:
    if state == "boundary":
        desired = float(gt_size) * (1.0 - float(config.boundary_precision)) / float(config.boundary_precision)
        return max(1, int(math.ceil(desired - 1e-12)))
    if state == "moderate":
        return max(1, int(config.hard_neg_topk) // 2)
    return max(1, int(config.hard_neg_topk) // 4)


def _negative_subset_for_state(
    state: str,
    negative_ranked_desc: torch.Tensor,
    *,
    gt_size: int,
    config: ControlledStateSynthesisConfig,
) -> torch.Tensor:
    desired = _desired_negative_count(state, gt_size=gt_size, config=config)
    count = max(1, min(int(config.hard_neg_topk), desired, int(negative_ranked_desc.numel())))
    return negative_ranked_desc[:count]


def _synthetic_metrics(
    state: str,
    *,
    gt_size: int,
    positive_count: int,
    negative_count: int,
    config: ControlledStateSynthesisConfig,
) -> tuple[float, float]:
    if state in {"under", "moderate", "stable"}:
        retained = max(1, int(gt_size) - int(positive_count))
        return float(retained) / float(gt_size), 1.0
    # Hard-negative indices are a representative optimization subset.  The
    # synthesized boundary state's semantic precision is controlled directly,
    # rather than being redefined by the sampling cap (which would otherwise
    # make large DBLP/Photo communities look artificially stable).
    return 1.0, float(config.boundary_precision)


def build_controlled_state_selection_plan(
    reference_embedding: torch.Tensor,
    query_sets,
    gt_sets,
    *,
    config: ControlledStateSynthesisConfig,
    graph_name: str = "source",
) -> dict:
    """Precompute frozen core/boundary selections for one source graph."""
    config.validate()
    if reference_embedding.ndim != 2:
        raise ValueError("reference_embedding must be [N,D]")
    if reference_embedding.requires_grad:
        raise ValueError("reference_embedding must be detached/frozen")
    queries = _as_sequence(query_sets)
    communities = _as_sequence(gt_sets)
    if len(queries) != len(communities):
        raise ValueError(f"query/gt count mismatch: {len(queries)} vs {len(communities)}")

    n_nodes = int(reference_embedding.shape[0])
    normalized_reference = F.normalize(reference_embedding.detach(), p=2, dim=-1, eps=1e-12)
    entries: list[dict | None] = []
    skipped = 0
    gt_sizes: list[int] = []
    plan_state_stats = {
        state: {
            "queries": 0,
            "positive_nodes": 0,
            "negative_nodes": 0,
            "desired_negative_nodes": 0,
            "synthetic_recall": 0.0,
            "synthetic_precision": 0.0,
            "representative_precision": 0.0,
            "core_positive_overlap": 0,
        }
        for state in _STATE_ORDER
    }
    for query_index, (query_nodes, gt_nodes) in enumerate(zip(queries, communities)):
        query = [int(x) for x in (query_nodes or []) if 0 <= int(x) < n_nodes]
        gt = sorted({int(x) for x in (gt_nodes or []) if 0 <= int(x) < n_nodes})
        if len(gt) < 2 or len(gt) >= n_nodes:
            entries.append(None)
            skipped += 1
            continue
        gt_sizes.append(len(gt))
        reference_scores = _rank_reference_nodes(normalized_reference, query)
        gt_tensor = torch.as_tensor(gt, device=reference_embedding.device, dtype=torch.long)
        positive_ranked = gt_tensor[torch.argsort(reference_scores[gt_tensor], descending=True)]
        negative_mask = torch.ones(n_nodes, device=reference_embedding.device, dtype=torch.bool)
        negative_mask[gt_tensor] = False
        negative_indices = torch.where(negative_mask)[0]
        if negative_indices.numel() == 0:
            entries.append(None)
            skipped += 1
            continue
        max_candidates = min(int(config.hard_neg_topk), int(negative_indices.numel()))
        top_indices = torch.topk(reference_scores[negative_indices], k=max_candidates, largest=True).indices
        negative_ranked = negative_indices[top_indices]
        state_entries = {}
        selected_positive_union: set[int] = set()
        for state in config.enabled_states:
            pos_idx = _positive_subset_for_state(state, positive_ranked, config=config)
            selected_positive_union.update(int(x) for x in pos_idx.detach().cpu().tolist())
            neg_idx = _negative_subset_for_state(state, negative_ranked, gt_size=len(gt), config=config)
            desired_negatives = _desired_negative_count(state, gt_size=len(gt), config=config)
            recall, precision = _synthetic_metrics(
                state,
                gt_size=len(gt),
                positive_count=int(pos_idx.numel()),
                negative_count=int(neg_idx.numel()),
                config=config,
            )
            representative_precision = float(len(gt)) / float(len(gt) + int(neg_idx.numel()))
            state_entries[state] = {
                "positive_indices": pos_idx.detach(),
                "negative_indices": neg_idx.detach(),
                "synthetic_recall": float(recall),
                "synthetic_precision": float(precision),
                "desired_negative_count": int(desired_negatives),
                "representative_precision": float(representative_precision),
            }
        core_candidates = [int(x) for x in positive_ranked.detach().cpu().tolist() if int(x) not in selected_positive_union]
        if not core_candidates:
            entries.append(None)
            skipped += 1
            continue
        core_count = min(int(config.core_topk), len(core_candidates))
        core_indices = torch.as_tensor(core_candidates[:core_count], device=reference_embedding.device, dtype=torch.long)
        core_weights = torch.softmax(
            reference_scores[core_indices] / max(float(config.temperature), 1e-6), dim=0
        ).detach()
        core_set = set(int(x) for x in core_indices.detach().cpu().tolist())
        for state in config.enabled_states:
            pos_idx = state_entries[state]["positive_indices"]
            overlap = len(core_set & set(int(x) for x in pos_idx.detach().cpu().tolist()))
            if overlap:
                raise RuntimeError("CDSS invariant violated: core and selected positive sets overlap")
            stat = plan_state_stats[state]
            stat["queries"] += 1
            stat["positive_nodes"] += int(pos_idx.numel())
            stat["negative_nodes"] += int(state_entries[state]["negative_indices"].numel())
            stat["desired_negative_nodes"] += int(state_entries[state]["desired_negative_count"])
            stat["synthetic_recall"] += float(state_entries[state]["synthetic_recall"])
            stat["synthetic_precision"] += float(state_entries[state]["synthetic_precision"])
            stat["representative_precision"] += float(state_entries[state]["representative_precision"])
            stat["core_positive_overlap"] += int(overlap)
        entries.append({
            "query_index": int(query_index),
            "core_indices": core_indices.detach(),
            "core_weights": core_weights,
            "states": state_entries,
        })
    state_plan_summary = {}
    for state, stat in plan_state_stats.items():
        count = int(stat["queries"])
        state_plan_summary[state] = {
            "queries": count,
            "mean_selected_positive_nodes": float(stat["positive_nodes"]) / max(1, count),
            "mean_selected_negative_nodes": float(stat["negative_nodes"]) / max(1, count),
            "mean_desired_negative_nodes": float(stat["desired_negative_nodes"]) / max(1, count),
            "mean_synthetic_recall": float(stat["synthetic_recall"]) / max(1, count),
            "mean_synthetic_precision": float(stat["synthetic_precision"]) / max(1, count),
            "mean_representative_precision": float(stat["representative_precision"]) / max(1, count),
            "core_positive_overlap_count": int(stat["core_positive_overlap"]),
            "negative_sampling_is_representative": True,
        }
    return {
        "graph_name": str(graph_name),
        "num_nodes": n_nodes,
        "embedding_dim": int(reference_embedding.shape[1]),
        "query_count": len(queries),
        "valid_query_count": len(queries) - skipped,
        "skipped_query_count": skipped,
        "mean_gt_size": (sum(gt_sizes) / len(gt_sizes)) if gt_sizes else 0.0,
        "state_plan_summary": state_plan_summary,
        "boundary_state_semantics": "target_precision_with_representative_hard_negative_subset",
        "config": asdict(config),
        "entries": entries,
        "source_only": True,
        "target_labels_used": False,
        "target_gt_used": False,
    }


def _soft_prototype(
    embedding: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (weights.to(embedding.device)[:, None] * embedding[indices.to(embedding.device)]).sum(dim=0)


def _mean_distance(nodes: torch.Tensor, prototype: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cosine":
        node_norm = F.normalize(nodes, p=2, dim=-1, eps=1e-12)
        proto_norm = F.normalize(prototype[None, :], p=2, dim=-1, eps=1e-12)
        similarity = (node_norm * proto_norm).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - similarity).mean()
    return ((nodes - prototype) ** 2).sum(dim=-1).mean()


def _smooth_positive_cap(value: torch.Tensor, cap: float) -> torch.Tensor:
    """Bound a non-negative penalty while retaining a smooth gradient."""
    cap_value = float(cap)
    return cap_value * torch.tanh(value / cap_value)


def _state_weights(state: str) -> tuple[float, float]:
    # Mild emphasis only. Controlled state severity is already encoded by the
    # selected positive/negative sets, preventing a repeat of the large loss
    # scale shift observed in the failed GlobalSearch state branch.
    return {
        "under": (1.20, 0.40),
        "moderate": (1.00, 0.60),
        "boundary": (0.60, 1.00),
        "stable": (0.50, 0.50),
    }[state]


def _aggregate_losses(
    losses_by_state: dict[str, list[torch.Tensor]],
    *,
    config: ControlledStateSynthesisConfig,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    non_empty = {state: values for state, values in losses_by_state.items() if values}
    if not non_empty:
        raise ValueError("cannot aggregate an empty loss collection")
    state_means = {state: torch.stack(values).mean() for state, values in non_empty.items()}
    mean_loss_values = {state: _finite_float(value) for state, value in state_means.items()}
    if config.aggregation == "query_mean":
        all_losses = [loss for values in non_empty.values() for loss in values]
        return torch.stack(all_losses).mean(), mean_loss_values, {state: 1.0 for state in non_empty}

    if config.aggregation == "capped_state_balance":
        counts = {state: len(values) for state, values in non_empty.items()}
        mean_count = sum(counts.values()) / float(len(counts))
        weights = {
            state: max(float(config.balance_min), min(float(config.balance_max), mean_count / float(count)))
            for state, count in counts.items()
        }
    weighted = [state_means[state] * float(weights[state]) for state in state_means]
    normalizer = sum(float(weights[state]) for state in state_means)
    return torch.stack(weighted).sum() / max(normalizer, 1e-12), mean_loss_values, weights


def compute_controlled_community_error_state_loss(
    embedding: torch.Tensor,
    reference_embedding: torch.Tensor,
    query_sets,
    gt_sets,
    *,
    config: ControlledStateSynthesisConfig,
    schedule_step: int = 0,
    graph_name: str = "source",
    selection_plan: dict | None = None,
) -> tuple[torch.Tensor, ControlledStateSynthesisReport]:
    """Compute the differentiable controlled community geometry loss."""
    config.validate()
    if embedding.ndim != 2 or reference_embedding.ndim != 2:
        raise ValueError("embedding and reference_embedding must be [N,D]")
    if tuple(embedding.shape) != tuple(reference_embedding.shape):
        raise ValueError("current/reference embedding shapes must match")
    if reference_embedding.requires_grad:
        raise ValueError("reference_embedding must be detached/frozen")
    plan_was_provided = selection_plan is not None
    if selection_plan is None:
        selection_plan = build_controlled_state_selection_plan(
            reference_embedding, query_sets, gt_sets, config=config, graph_name=graph_name
        )
    if int(selection_plan.get("num_nodes", -1)) != int(embedding.shape[0]):
        raise ValueError("selection plan node count mismatch")
    if selection_plan.get("config") != asdict(config):
        raise ValueError("selection plan config mismatch")

    states = tuple(str(x) for x in config.enabled_states)
    losses_by_state: dict[str, list[torch.Tensor]] = {state: [] for state in _STATE_ORDER}
    state_recalls: dict[str, list[float]] = {state: [] for state in _STATE_ORDER}
    state_precisions: dict[str, list[float]] = {state: [] for state in _STATE_ORDER}
    state_positive_counts: dict[str, list[int]] = {state: [] for state in _STATE_ORDER}
    state_negative_counts: dict[str, list[int]] = {state: [] for state in _STATE_ORDER}
    positive_distances: list[float] = []
    negative_distances: list[float] = []
    ratios: list[float] = []
    separations: list[float] = []
    core_counts: list[int] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    active = 0
    skipped = 0

    for query_index, entry in enumerate(selection_plan["entries"]):
        if entry is None:
            skipped += 1
            continue
        state = _state_for_query(
            query_index=query_index,
            schedule_step=int(schedule_step),
            graph_name=str(graph_name),
            states=states,
        )
        state_entry = entry["states"][state]
        core_indices = entry["core_indices"].to(embedding.device)
        core_weights = entry["core_weights"].to(embedding.device)
        positive_indices = state_entry["positive_indices"].to(embedding.device)
        negative_indices = state_entry["negative_indices"].to(embedding.device)
        if core_indices.numel() == 0 or positive_indices.numel() == 0 or negative_indices.numel() == 0:
            skipped += 1
            continue
        prototype = _soft_prototype(embedding, core_indices, core_weights)
        positive_distance = _mean_distance(embedding[positive_indices], prototype, config.distance_mode)
        negative_distance = _mean_distance(embedding[negative_indices], prototype, config.distance_mode)
        denominator = (positive_distance + negative_distance).clamp_min(float(config.epsilon))
        ratio = positive_distance / denominator
        if config.separation_mode == "relative":
            gap = (negative_distance - positive_distance) / denominator
            separation = F.relu(float(config.relative_margin) - gap)
        else:
            raw_separation = F.relu(float(config.absolute_margin) + positive_distance - negative_distance)
            cap = float(config.absolute_separation_cap)
            # Smooth cap limits extreme loss magnitude without creating the
            # zero-gradient plateau caused by clamp_max.
            separation = _smooth_positive_cap(raw_separation, cap)
        if not bool(torch.isfinite(ratio).all()) or not bool(torch.isfinite(separation).all()):
            raise FloatingPointError("non-finite CDSS ratio/separation")
        compact_weight, separation_weight = _state_weights(state)
        loss = float(compact_weight) * ratio + float(separation_weight) * 0.5 * separation
        losses_by_state[state].append(loss)
        state_recalls[state].append(float(state_entry["synthetic_recall"]))
        state_precisions[state].append(float(state_entry["synthetic_precision"]))
        state_positive_counts[state].append(int(positive_indices.numel()))
        state_negative_counts[state].append(int(negative_indices.numel()))
        positive_distances.append(_finite_float(positive_distance))
        negative_distances.append(_finite_float(negative_distance))
        ratios.append(_finite_float(ratio))
        separations.append(_finite_float(separation))
        active += int(_finite_float(separation) > 0.0)
        core_counts.append(int(core_indices.numel()))
        positive_counts.append(int(positive_indices.numel()))
        negative_counts.append(int(negative_indices.numel()))

    valid = sum(len(values) for values in losses_by_state.values())
    report = ControlledStateSynthesisReport(
        valid_queries=int(valid),
        skipped_queries=int(skipped),
        state_counts={state: len(losses_by_state[state]) for state in _STATE_ORDER},
        reference_requires_grad=bool(reference_embedding.requires_grad),
        selection_plan_reused=bool(plan_was_provided),
        config=asdict(config),
    )
    if valid == 0:
        return embedding.sum() * 0.0, report

    aggregate, state_mean_losses, state_balance_weights = _aggregate_losses(losses_by_state, config=config)
    report.state_mean_losses.update(state_mean_losses)
    report.state_balance_weights.update(state_balance_weights)
    report.state_mean_synthetic_recall.update({
        state: (sum(values) / len(values) if values else 0.0) for state, values in state_recalls.items()
    })
    report.state_mean_synthetic_precision.update({
        state: (sum(values) / len(values) if values else 0.0) for state, values in state_precisions.items()
    })
    report.state_mean_selected_positive_nodes.update({
        state: (sum(values) / len(values) if values else 0.0) for state, values in state_positive_counts.items()
    })
    report.state_mean_selected_negative_nodes.update({
        state: (sum(values) / len(values) if values else 0.0) for state, values in state_negative_counts.items()
    })
    report.mean_positive_distance = sum(positive_distances) / len(positive_distances)
    report.mean_negative_distance = sum(negative_distances) / len(negative_distances)
    report.mean_ratio = sum(ratios) / len(ratios)
    report.mean_separation = sum(separations) / len(separations)
    report.separation_active_rate = float(active) / float(valid)
    report.mean_core_nodes = sum(core_counts) / len(core_counts)
    report.mean_selected_positive_nodes = sum(positive_counts) / len(positive_counts)
    report.mean_selected_negative_nodes = sum(negative_counts) / len(negative_counts)
    return aggregate, report
