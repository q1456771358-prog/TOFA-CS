"""Prediction/community output helpers for V5 fixed-checkpoint inference.

Outputs are predictions only.  No labels, ground-truth communities, F1/Jaccard,
NMI/ARI, or oracle information may be stored in prediction metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

from .errors import V5DataLeakageError, V5ProtocolError
from .senior_compat import senior_global_search_from_ranked_scores
from .chunked_scoring import ChunkedScoreResult

_FORBIDDEN_OUTPUT_METADATA_KEYS = {
    "label", "labels", "ground_truth", "gt", "answer", "answers",
    "metric", "metrics", "score", "scores", "f1", "jaccard", "nmi", "ari",
    "unseen_metric", "unseen_score", "test_metric", "test_score", "oracle",
}


@dataclass(frozen=True)
class CommunityPredictionRecord:
    graph_name: str
    query_index: int
    query_nodes_local: tuple[int, ...]
    predicted_nodes_local: tuple[int, ...]
    query_nodes_global: tuple[int, ...] | None = None
    predicted_nodes_global: tuple[int, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_nodes_local:
            raise V5ProtocolError("prediction record requires query_nodes_local")
        if not self.predicted_nodes_local:
            raise V5ProtocolError("prediction record requires predicted_nodes_local")
        missing = sorted(set(self.query_nodes_local) - set(self.predicted_nodes_local))
        if missing:
            raise V5ProtocolError(f"query seeds missing from predicted community: {missing}")
        forbidden = sorted(set(str(k).lower() for k in self.metadata.keys()) & _FORBIDDEN_OUTPUT_METADATA_KEYS)
        if forbidden:
            raise V5DataLeakageError(f"forbidden prediction metadata keys: {forbidden}")

    @property
    def pred_size(self) -> int:
        return len(self.predicted_nodes_local)

    @property
    def query_missing_count(self) -> int:
        return len(set(self.query_nodes_local) - set(self.predicted_nodes_local))

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["pred_size"] = self.pred_size
        out["query_missing_count"] = self.query_missing_count
        return out


def _local_to_global(nodes: Sequence[int], global_node_ids: Sequence[int] | None) -> tuple[int, ...] | None:
    if global_node_ids is None:
        return None
    return tuple(int(global_node_ids[int(i)]) for i in nodes)


def build_prediction_records_from_chunked_scores(
    *,
    graph_name: str,
    score_result: ChunkedScoreResult,
    global_node_ids: Sequence[int] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[CommunityPredictionRecord, ...]:
    """Run senior GlobalSearch for every query and build JSONL-safe records."""

    records: list[CommunityPredictionRecord] = []
    for qi, qset in enumerate(score_result.query_node_sets):
        ranked_nodes = [int(x) for x in score_result.top_nodes[qi].tolist()]
        ranked_scores = [float(x) for x in score_result.top_scores[qi].tolist()]
        q_scores = [float(score_result.query_seed_scores[qi][int(seed)]) for seed in qset]
        pred = senior_global_search_from_ranked_scores(
            query_nodes=qset,
            query_scores=q_scores,
            ranked_nodes=ranked_nodes,
            ranked_scores=ranked_scores,
            num_nodes=score_result.num_nodes,
            full_score_sum=float(score_result.full_score_sums[qi].item()),
        )
        records.append(
            CommunityPredictionRecord(
                graph_name=graph_name,
                query_index=int(qi),
                query_nodes_local=tuple(int(x) for x in qset),
                predicted_nodes_local=tuple(int(x) for x in pred),
                query_nodes_global=_local_to_global(qset, global_node_ids),
                predicted_nodes_global=_local_to_global(pred, global_node_ids),
                metadata=dict(metadata or {}),
            )
        )
    return tuple(records)


def write_prediction_jsonl(records: Sequence[CommunityPredictionRecord], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def write_plain_pred(records: Sequence[CommunityPredictionRecord], path: str | Path, *, use_global_ids: bool = False) -> Path:
    """Write one predicted community per line, senior-style."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            nodes = rec.predicted_nodes_global if use_global_ids else rec.predicted_nodes_local
            if nodes is None:
                raise V5ProtocolError("global ids requested but record has no predicted_nodes_global")
            f.write(" ".join(str(int(x)) for x in nodes) + "\n")
    return out_path


def prediction_protocol_check(records: Sequence[CommunityPredictionRecord]) -> dict[str, object]:
    if not records:
        raise V5ProtocolError("no prediction records to check")
    sizes = [int(r.pred_size) for r in records]
    missing = [int(r.query_missing_count) for r in records]
    return {
        "record_count": len(records),
        "min_pred_size": min(sizes),
        "max_pred_size": max(sizes),
        "avg_pred_size": sum(sizes) / len(sizes),
        "all_len_20": all(s == 20 for s in sizes),
        "query_missing_rows": sum(1 for m in missing if m > 0),
        "query_missing_total": sum(missing),
        "labels_used": False,
        "metrics_computed": False,
        "ground_truth_loaded": False,
    }


def write_protocol_check(records: Sequence[CommunityPredictionRecord], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prediction_protocol_check(records), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
