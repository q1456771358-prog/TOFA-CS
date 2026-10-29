"""Offline evaluation helpers aligned with the senior OFA-CS evaluator.

These helpers are intentionally kept out of fixed prediction export.  They may
read ``*.gt`` only in an explicit post-export offline evaluation command, never
inside training, checkpoint selection, or fixed export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, jaccard_score, normalized_mutual_info_score

from .errors import V5ProtocolError
from .senior_compat import parse_query_lines, senior_global_search_prefix_limit


@dataclass(frozen=True)
class SeniorAlignedOfflineMetrics:
    graph_name: str
    query_count: int
    num_nodes: int
    official_precision: float
    official_recall: float
    official_f1: float
    official_jaccard: float
    senior_micro_precision: float
    senior_micro_recall: float
    senior_micro_f1: float
    senior_nmi: float
    senior_ari: float
    senior_jaccard: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_jaccard: float
    avg_pred_size: float
    avg_gt_size: float
    min_pred_size: int
    max_pred_size: int
    min_gt_size: int
    max_gt_size: int
    avg_pred_over_gt: float
    endpoint_cap: int
    endpoint_hit_rate: float
    near_endpoint_hit_rate: float
    query_missing_rows: int
    prediction_format: str
    official_metric_protocol: str = "senior_flatten_micro_v1"
    official_metric_source: str = "senior_micro_f1_and_flat_jaccard"
    offline_eval_only: bool = True
    used_for_checkpoint_selection: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_node_sets(path: str | Path) -> tuple[tuple[int, ...], ...]:
    return parse_query_lines(Path(path).read_text(encoding="utf-8").splitlines())


def load_predictions(path: str | Path) -> tuple[tuple[int, ...], ...]:
    p = Path(path)
    if not p.exists():
        raise V5ProtocolError(f"prediction file not found: {p}")
    if p.suffix == ".jsonl" or p.name.endswith(".pred.jsonl"):
        rows: list[tuple[int, ...]] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise V5ProtocolError(f"prediction jsonl line {line_no} must be an object")
                nodes = row.get("predicted_nodes_local")
                if nodes is None:
                    nodes = row.get("predicted_nodes")
                if nodes is None:
                    raise V5ProtocolError(f"prediction jsonl line {line_no} missing predicted_nodes_local")
                rows.append(tuple(dict.fromkeys(int(x) for x in nodes)))
        if not rows:
            raise V5ProtocolError(f"prediction jsonl is empty: {p}")
        return tuple(rows)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(tuple(dict.fromkeys(int(x) for x in stripped.replace(",", " ").split())))
    if not rows:
        raise V5ProtocolError(f"prediction file is empty: {p}")
    return tuple(rows)


def _sets_to_binary_matrix(node_sets: Sequence[Sequence[int]], *, num_nodes: int, field: str) -> torch.Tensor:
    y = torch.zeros((len(node_sets), int(num_nodes)), dtype=torch.int32)
    for qi, nodes in enumerate(node_sets):
        normalized = tuple(dict.fromkeys(int(x) for x in nodes))
        if not normalized:
            continue
        if min(normalized) < 0 or max(normalized) >= int(num_nodes):
            raise V5ProtocolError(f"{field} query {qi} contains node outside [0,{num_nodes})")
        y[qi, torch.tensor(normalized, dtype=torch.long)] = 1
    return y


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def senior_aligned_offline_evaluate(
    *,
    graph_name: str,
    prediction_path: str | Path,
    gt_path: str | Path,
    query_path: str | Path | None = None,
    num_nodes: int | None = None,
) -> SeniorAlignedOfflineMetrics:
    preds = load_predictions(prediction_path)
    gt_sets = load_node_sets(gt_path)
    if len(preds) != len(gt_sets):
        raise V5ProtocolError(f"prediction/gt query count mismatch: {len(preds)} vs {len(gt_sets)}")
    query_sets = load_node_sets(query_path) if query_path is not None else None
    if query_sets is not None and len(query_sets) != len(preds):
        raise V5ProtocolError(f"query/prediction count mismatch: {len(query_sets)} vs {len(preds)}")
    max_id = -1
    for collection in (preds, gt_sets, query_sets or ()):  # type: ignore[arg-type]
        for nodes in collection:
            if nodes:
                max_id = max(max_id, max(int(x) for x in nodes))
    if num_nodes is None:
        num_nodes = max_id + 1
    if int(num_nodes) <= 0:
        raise V5ProtocolError("num_nodes must be positive")

    y_pred = _sets_to_binary_matrix(preds, num_nodes=int(num_nodes), field="prediction")
    y_true = _sets_to_binary_matrix(gt_sets, num_nodes=int(num_nodes), field="ground_truth")

    tp = torch.sum(y_pred * y_true).double().item()
    pred_sum = torch.sum(y_pred).double().item()
    true_sum = torch.sum(y_true).double().item()
    precision = _safe_div(tp, pred_sum + 1e-9)
    recall = _safe_div(tp, true_sum + 1e-9)
    micro_f1 = _safe_div(2.0 * precision * recall, precision + recall + 1e-9)

    flat_pred = y_pred.reshape(-1).numpy()
    flat_true = y_true.reshape(-1).numpy()
    nmi = float(normalized_mutual_info_score(flat_true, flat_pred))
    ari = float(adjusted_rand_score(flat_true, flat_pred))
    jac = float(jaccard_score(flat_true, flat_pred, zero_division=0))

    per_p: list[float] = []
    per_r: list[float] = []
    per_f1: list[float] = []
    per_j: list[float] = []
    pred_sizes: list[int] = []
    gt_sizes: list[int] = []
    pred_over_gt: list[float] = []
    missing_rows = 0
    endpoint = senior_global_search_prefix_limit(int(num_nodes))
    endpoint_hits = 0
    near_endpoint_hits = 0
    for qi, (pred, gt) in enumerate(zip(preds, gt_sets)):
        ps = set(int(x) for x in pred)
        gs = set(int(x) for x in gt)
        hit = len(ps & gs)
        p = _safe_div(float(hit), float(len(ps)))
        r = _safe_div(float(hit), float(len(gs)))
        f1 = _safe_div(2.0 * p * r, p + r)
        union = len(ps | gs)
        jj = _safe_div(float(hit), float(union))
        per_p.append(p)
        per_r.append(r)
        per_f1.append(f1)
        per_j.append(jj)
        pred_sizes.append(len(ps))
        gt_sizes.append(len(gs))
        pred_over_gt.append(_safe_div(float(len(ps)), float(max(1, len(gs)))))
        if query_sets is not None and not set(query_sets[qi]).issubset(ps):
            missing_rows += 1
        query_extra = len(query_sets[qi]) if query_sets is not None else 0
        endpoint_plus_query = min(int(num_nodes), endpoint + query_extra)
        if len(ps) >= endpoint_plus_query:
            endpoint_hits += 1
        if len(ps) >= int(0.99 * max(1, endpoint_plus_query)):
            near_endpoint_hits += 1

    return SeniorAlignedOfflineMetrics(
        graph_name=str(graph_name),
        query_count=len(preds),
        num_nodes=int(num_nodes),
        official_precision=float(precision),
        official_recall=float(recall),
        official_f1=float(micro_f1),
        official_jaccard=float(jac),
        senior_micro_precision=float(precision),
        senior_micro_recall=float(recall),
        senior_micro_f1=float(micro_f1),
        senior_nmi=nmi,
        senior_ari=ari,
        senior_jaccard=jac,
        macro_precision=float(np.mean(per_p)),
        macro_recall=float(np.mean(per_r)),
        macro_f1=float(np.mean(per_f1)),
        macro_jaccard=float(np.mean(per_j)),
        avg_pred_size=float(np.mean(pred_sizes)),
        avg_gt_size=float(np.mean(gt_sizes)),
        min_pred_size=int(min(pred_sizes)),
        max_pred_size=int(max(pred_sizes)),
        min_gt_size=int(min(gt_sizes)),
        max_gt_size=int(max(gt_sizes)),
        avg_pred_over_gt=float(np.mean(pred_over_gt)),
        endpoint_cap=int(endpoint),
        endpoint_hit_rate=float(endpoint_hits / max(1, len(preds))),
        near_endpoint_hit_rate=float(near_endpoint_hits / max(1, len(preds))),
        query_missing_rows=int(missing_rows),
        prediction_format="jsonl" if str(prediction_path).endswith(".jsonl") else "plain_pred",
    )
