"""Offline multi-view candidate re-selection utilities.

These helpers are intentionally label-free during selection.  Ground truth is
allowed only by caller-side diagnostic/evaluation tools after a candidate has
already been selected.  They operate on ``*.mv_candidates.jsonl`` rows written by
``ofacs.multiview_search``.  Old candidate files that do not contain candidate node
sets are rejected with a clear error because selector-only re-evaluation would be
ill-posed without the alternative communities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import V5ProtocolError

SELECTOR_NAMES = (
    "original_selector",
    "canonical_guard",
    "no_temperature",
    "temperature_guard_085",
    "temperature_guard_095",
    "recall_safe",
    "balanced_size",
    "citeseer_safe_unsupervised",
    "structure_secondary",
)


def jaccard(a: Sequence[int] | set[int], b: Sequence[int] | set[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def f1_jaccard(pred: Sequence[int], gt: Sequence[int]) -> tuple[float, float, int, int, int]:
    ps = set(int(x) for x in pred)
    gs = set(int(x) for x in gt)
    tp = len(ps & gs)
    prec = float(tp) / float(len(ps)) if ps else 0.0
    rec = float(tp) / float(len(gs)) if gs else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    union = len(ps | gs)
    jac = float(tp) / float(union) if union else 0.0
    return float(f1), float(jac), int(tp), int(len(ps)), int(len(gs))


def senior_micro_metrics(preds: Sequence[Sequence[int]], gts: Sequence[Sequence[int]]) -> dict[str, float]:
    if len(preds) != len(gts):
        raise V5ProtocolError(f"prediction/gt query count mismatch: {len(preds)} vs {len(gts)}")
    tp = pred_sum = gt_sum = union_sum = 0
    f1s: list[float] = []
    jacs: list[float] = []
    sizes: list[int] = []
    for pred, gt in zip(preds, gts):
        f1, jac, hit, ps, gs = f1_jaccard(pred, gt)
        tp += hit
        pred_sum += ps
        gt_sum += gs
        union_sum += len(set(int(x) for x in pred) | set(int(x) for x in gt))
        f1s.append(f1)
        jacs.append(jac)
        sizes.append(ps)
    precision = float(tp) / float(pred_sum + 1e-9)
    recall = float(tp) / float(gt_sum + 1e-9)
    micro_f1 = (2.0 * precision * recall / (precision + recall + 1e-9)) if (precision + recall) > 0 else 0.0
    micro_jac = float(tp) / float(union_sum + 1e-9)
    return {
        "senior_micro_precision": float(precision),
        "senior_micro_recall": float(recall),
        "senior_micro_f1": float(micro_f1),
        "senior_jaccard": float(micro_jac),
        "macro_f1": float(sum(f1s) / max(1, len(f1s))),
        "macro_jaccard": float(sum(jacs) / max(1, len(jacs))),
        "avg_pred_size": float(sum(sizes) / max(1, len(sizes))),
    }


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _minmax(vals: Sequence[float], higher: bool = True) -> list[float]:
    xs = [_safe_float(v) for v in vals]
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo <= 1e-12:
        out = [0.5] * len(xs)
    else:
        out = [(x - lo) / (hi - lo) for x in xs]
    if not higher:
        out = [1.0 - x for x in out]
    return [float(x) for x in out]


@dataclass
class CandidateChoice:
    query_index: int
    selector: str
    selected_view: str
    selected_nodes: tuple[int, ...]
    canonical_view: str
    canonical_nodes: tuple[int, ...]
    selected_size: int
    canonical_size: int
    size_ratio_to_canonical: float
    reason: str
    selected_quality: float
    canonical_quality: float
    original_selected_view: str | None = None


def load_mv_candidate_entries(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "candidates" not in obj:
                raise V5ProtocolError(f"{p}:{line_no} is not an mv_candidates row")
            rows.append(obj)
    return rows


def _candidate_nodes(c: Mapping[str, Any]) -> tuple[int, ...]:
    nodes = c.get("candidate_nodes_local")
    if nodes is None:
        nodes = c.get("predicted_nodes_local")
    if nodes is None:
        nodes = c.get("candidate_nodes")
    if nodes is None:
        raise V5ProtocolError(
            "mv_candidates row does not contain candidate_nodes_local; "
            "this file was likely produced before selector-diagnostic support and cannot be reselected"
        )
    return tuple(dict.fromkeys(int(x) for x in nodes))


def canonical_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for c in candidates:
        if str(c.get("view_name", "")) == "guided_refined__none":
            return c
    for c in candidates:
        if str(c.get("view_name", "")).endswith("__none"):
            return c
    if not candidates:
        raise V5ProtocolError("empty candidate list")
    return candidates[0]


def _feature_matrix(candidates: Sequence[Mapping[str, Any]], canonical_size: int) -> list[dict[str, float]]:
    raw = []
    for c in candidates:
        size = int(c.get("candidate_size", len(_candidate_nodes(c))))
        ratio = float(size) / float(max(1, canonical_size))
        raw.append({
            "quality_raw": _safe_float(c.get("quality", c.get("legacy_quality", 0.0))),
            "canonical_density": _safe_float(c.get("canonical_density", c.get("native_senior_density", 0.0))),
            "compactness": _safe_float(c.get("canonical_compactness", c.get("native_mean_selected", 0.0))),
            "boundary_gap": _safe_float(c.get("canonical_boundary_gap", c.get("native_boundary_gap", 0.0))),
            "structural_quality": _safe_float(c.get("structural_quality", 0.0)),
            "lcc": _safe_float(c.get("largest_component_ratio", 0.0)),
            "conductance_quality": _safe_float(c.get("conductance_penalty", 0.5)),
            "raw_agreement": _safe_float(c.get("raw_guided_agreement", 0.0)),
            "entropy": _safe_float(c.get("top_score_entropy", 0.0)),
            "size_ratio": ratio,
            "size_band_error": max(0.0, 0.85 - ratio) + max(0.0, ratio - 1.25),
            "under_size": max(0.0, 0.85 - ratio),
        })
    norms = {
        "quality_raw": _minmax([x["quality_raw"] for x in raw], True),
        "canonical_density": _minmax([x["canonical_density"] for x in raw], True),
        "compactness": _minmax([x["compactness"] for x in raw], True),
        "boundary_gap": _minmax([x["boundary_gap"] for x in raw], True),
        "structural_quality": _minmax([x["structural_quality"] for x in raw], True),
        "lcc": _minmax([x["lcc"] for x in raw], True),
        "conductance_quality": _minmax([x["conductance_quality"] for x in raw], False),
        "raw_agreement": _minmax([x["raw_agreement"] for x in raw], True),
        "size_band": _minmax([x["size_band_error"] for x in raw], False),
        "under_size": _minmax([x["under_size"] for x in raw], False),
    }
    out: list[dict[str, float]] = []
    for i, r in enumerate(raw):
        row = dict(r)
        for k, values in norms.items():
            row[f"norm_{k}"] = values[i]
        out.append(row)
    return out


def _base_score(c: Mapping[str, Any], f: Mapping[str, float]) -> float:
    return float(
        0.14 * f["norm_canonical_density"]
        + 0.08 * f["norm_compactness"]
        + 0.10 * f["norm_boundary_gap"]
        + 0.18 * f["norm_size_band"]
        + 0.08 * f["norm_raw_agreement"]
        + 0.04 * f["norm_structural_quality"]
        + 0.02 * f["norm_lcc"]
        + 0.02 * f["norm_conductance_quality"]
    )


def choose_candidate(entry: Mapping[str, Any], selector: str) -> CandidateChoice:
    selector = str(selector)
    if selector not in SELECTOR_NAMES:
        raise V5ProtocolError(f"unsupported selector={selector!r}; choices={SELECTOR_NAMES}")
    candidates = list(entry.get("candidates", []))
    if not candidates:
        raise V5ProtocolError("mv_candidates entry has no candidates")
    original = next((c for c in candidates if bool(c.get("selected", False))), None)
    can = canonical_candidate(candidates)
    can_nodes = _candidate_nodes(can)
    can_size = int(can.get("candidate_size", len(can_nodes)))
    can_view = str(can.get("view_name", "canonical"))
    feats = _feature_matrix(candidates, can_size)

    scores: list[float] = []
    reasons: list[str] = []
    high_entropy = max((f["entropy"] for f in feats), default=0.0) >= 0.82
    weak_gap = _safe_float(can.get("canonical_boundary_gap", can.get("native_boundary_gap", 0.0))) <= 0.0
    for c, f in zip(candidates, feats):
        view = str(c.get("view_name", ""))
        profile = str(c.get("search_profile", ""))
        is_temp = ("temperature" in view) or profile == "temperature"
        is_recall = ("raw_plus" in view) or ("recall_profile" in view) or profile == "recall_profile"
        size_ratio = f["size_ratio"]
        q = _base_score(c, f)
        reason = "base_balanced_score"
        if selector == "original_selector":
            q = 1.0 if bool(c.get("selected", False)) else 0.0
            reason = "original_selected_flag" if q == 1.0 else "original_not_selected"
        elif selector == "no_temperature":
            if is_temp:
                q -= 100.0
                reason = "temperature_disabled"
        elif selector.startswith("temperature_guard"):
            min_ratio = 0.95 if selector.endswith("095") else 0.85
            if is_temp and size_ratio < min_ratio:
                q -= 100.0
                reason = f"temperature_rejected_size_ratio_{size_ratio:.3f}_lt_{min_ratio:.2f}"
            elif is_temp and high_entropy:
                q -= 0.20
                reason = "temperature_soft_penalty_high_entropy"
        elif selector == "recall_safe":
            if size_ratio < 0.90:
                q -= 0.50 * (0.90 - size_ratio + 1.0)
                reason = f"under_size_recall_penalty_{size_ratio:.3f}"
            if is_recall:
                q += 0.12
                reason = "recall_or_raw_plus_bonus"
            if is_temp and size_ratio < 1.0:
                q -= 0.25 * (1.0 - size_ratio + 1.0)
                reason = f"temperature_recall_safe_penalty_{size_ratio:.3f}"
        elif selector == "balanced_size":
            if 0.85 <= size_ratio <= 1.25:
                q += 0.08
                reason = "balanced_size_band_bonus"
            elif size_ratio < 0.85:
                q -= 0.50 * (0.85 - size_ratio + 1.0)
                reason = f"balanced_reject_under_size_{size_ratio:.3f}"
            else:
                q -= 0.15 * (size_ratio - 1.25 + 1.0)
                reason = f"balanced_penalty_over_size_{size_ratio:.3f}"
        elif selector == "citeseer_safe_unsupervised":
            # Name kept for the experiment design, but rule is target-name-free.
            if (high_entropy or weak_gap) and is_temp and size_ratio < 0.98:
                q -= 100.0
                reason = f"unsupervised_recall_protection_temperature_size_{size_ratio:.3f}"
            if (high_entropy or weak_gap) and is_recall:
                q += 0.12
                reason = "unsupervised_recall_protection_raw_recall_bonus"
            if size_ratio < 0.85:
                q -= 0.35 * (0.85 - size_ratio + 1.0)
        elif selector == "structure_secondary":
            q = float(
                0.18 * f["norm_canonical_density"]
                + 0.10 * f["norm_boundary_gap"]
                + 0.22 * f["norm_size_band"]
                + 0.08 * f["norm_raw_agreement"]
                + 0.02 * f["norm_structural_quality"]
            )
            reason = "structure_secondary_not_primary"
        elif selector == "canonical_guard":
            # Score first, guard after provisional best is known.
            reason = "canonical_guard_candidate_score"
        scores.append(float(q))
        reasons.append(reason)

    best_idx = max(range(len(candidates)), key=lambda i: (scores[i], -abs(feats[i]["size_ratio"] - 1.0), 0 if candidates[i] is can else 1))
    if selector == "canonical_guard":
        can_idx = candidates.index(can)
        if best_idx != can_idx:
            margin = scores[best_idx] - scores[can_idx]
            ratio = feats[best_idx]["size_ratio"]
            if margin < 0.08:
                reasons[best_idx] = f"canonical_guard_rejected_margin_{margin:.4f}_lt_0.08"
                best_idx = can_idx
            elif ratio < 0.85:
                reasons[best_idx] = f"canonical_guard_rejected_size_ratio_{ratio:.3f}_lt_0.85"
                best_idx = can_idx
            else:
                reasons[best_idx] = f"canonical_guard_allowed_margin_{margin:.4f}_size_ratio_{ratio:.3f}"
    chosen = candidates[best_idx]
    nodes = _candidate_nodes(chosen)
    size = int(chosen.get("candidate_size", len(nodes)))
    return CandidateChoice(
        query_index=int(entry.get("query_index", chosen.get("query_index", 0))),
        selector=selector,
        selected_view=str(chosen.get("view_name", "")),
        selected_nodes=tuple(nodes),
        canonical_view=can_view,
        canonical_nodes=tuple(can_nodes),
        selected_size=int(size),
        canonical_size=int(can_size),
        size_ratio_to_canonical=float(size) / float(max(1, can_size)),
        reason=str(reasons[best_idx]),
        selected_quality=float(scores[best_idx]),
        canonical_quality=float(scores[candidates.index(can)]),
        original_selected_view=str(original.get("view_name", "")) if original is not None else None,
    )


def write_prediction_jsonl_from_choices(path: str | Path, graph_name: str, choices: Sequence[CandidateChoice]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ch in choices:
            row = {
                "graph_name": graph_name,
                "query_index": int(ch.query_index),
                "query_nodes_local": [],
                "predicted_nodes_local": [int(x) for x in ch.selected_nodes],
                "query_nodes_global": None,
                "predicted_nodes_global": None,
                "pred_size": int(ch.selected_size),
                "query_missing_count": 0,
                "metadata": {
                    "selector_only_reselection": True,
                    "selector": ch.selector,
                    "selected_view": ch.selected_view,
                    "canonical_view": ch.canonical_view,
                    "size_ratio_to_canonical": ch.size_ratio_to_canonical,
                    "selected_reason": ch.reason,
                    "labels_used": False,
                    "ground_truth_used": False,
                    "metrics_computed": False,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return out
