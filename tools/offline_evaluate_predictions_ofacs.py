#!/usr/bin/env python3
"""Post-export offline evaluator aligned with the senior OFA-CS metrics.

This command is deliberately separate from fixed export.  It reads ``*.gt`` only
when the caller passes ``--allow-unseen-ground-truth-eval`` and writes metrics
that must not be fed back into training, checkpoint selection, or model routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ofacs.errors import V5DataLeakageError, V5ProtocolError
from ofacs.offline_eval import file_sha256, senior_aligned_offline_evaluate
from ofacs.senior_data import discover_senior_dataset_files, validate_dataset_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--graphs", required=True, help="comma-separated graph names")
    p.add_argument("--predictions-root", required=True, help="directory containing <graph>/<graph>.pred.jsonl or direct prediction files")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--allow-unseen-ground-truth-eval", action="store_true")
    p.add_argument("--used-for-checkpoint-selection", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _prediction_path(predictions_root: Path, graph: str) -> Path:
    candidates = [
        predictions_root / graph / f"{graph}.pred.jsonl",
        predictions_root / f"{graph}.pred.jsonl",
        predictions_root / graph / f"{graph}.pred",
        predictions_root / f"{graph}.pred",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise V5ProtocolError(f"no prediction file found for {graph} under {predictions_root}")


def main() -> None:
    args = parse_args()
    if not args.allow_unseen_ground_truth_eval:
        raise V5DataLeakageError("offline evaluation requires explicit --allow-unseen-ground-truth-eval")
    if args.used_for_checkpoint_selection:
        raise V5DataLeakageError("offline unseen metrics must not be used for checkpoint selection")
    graphs = [validate_dataset_name(x.strip()) for x in args.graphs.split(",") if x.strip()]
    if not graphs:
        raise V5ProtocolError("--graphs must contain at least one graph")
    pred_root = Path(args.predictions_root)
    records: list[dict[str, object]] = []
    for graph in graphs:
        files = discover_senior_dataset_files(args.dataset_root, graph)
        if not files.gt_exists or not files.gt_path_present_but_not_read:
            raise V5ProtocolError(f"ground-truth file not found for offline eval graph={graph}")
        if not files.query_exists or files.query_path is None:
            raise V5ProtocolError(f"query file not found for offline eval graph={graph}")
        pred_path = _prediction_path(pred_root, graph)
        metric = senior_aligned_offline_evaluate(
            graph_name=graph,
            prediction_path=pred_path,
            gt_path=files.gt_path_present_but_not_read,
            query_path=files.query_path,
        )
        row = metric.to_dict()
        row.update(
            {
                "metric_contract": "official_f1=senior_micro_f1; official_jac=senior_jaccard; macro fields are diagnostics only",
                "prediction_file": str(pred_path),
                "prediction_sha256": file_sha256(pred_path),
                "gt_file": str(files.gt_path_present_but_not_read),
                "gt_sha256": file_sha256(files.gt_path_present_but_not_read),
                "query_file": str(files.query_path),
                "query_sha256": file_sha256(files.query_path),
                "gt_file_read": True,
                "labels_used_for_training": False,
                "metrics_used_for_training": False,
                "checkpoint_selection_from_unseen": False,
                "used_for_checkpoint_selection": False,
                "fixed_export_read_gt": False,
            }
        )
        records.append(row)
    macro = _macro(records)
    payload = {
        "protocol": "OFA-CS-V5-official-senior-aligned-offline-eval-v1",
        "official_metric_contract": "official_f1=senior_micro_f1; official_jac=senior_jaccard; macro metrics are diagnostics only",
        "senior_aligned": True,
        "offline_eval_only": True,
        "fixed_export_read_gt": False,
        "checkpoint_selection_from_unseen": False,
        "used_for_checkpoint_selection": False,
        "records": records,
        "macro": macro,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out_csv, records, macro)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _macro(records: list[dict[str, object]]) -> dict[str, float]:
    fields = [
        "official_precision",
        "official_recall",
        "official_f1",
        "official_jaccard",
        "senior_micro_precision",
        "senior_micro_recall",
        "senior_micro_f1",
        "senior_nmi",
        "senior_ari",
        "senior_jaccard",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_jaccard",
        "avg_pred_size",
        "avg_gt_size",
        "avg_pred_over_gt",
        "endpoint_hit_rate",
        "near_endpoint_hit_rate",
    ]
    return {field: float(sum(float(r[field]) for r in records) / max(1, len(records))) for field in fields}

def _write_csv(path: Path, records: list[dict[str, object]], macro: dict[str, float]) -> None:
    fieldnames = [
        "graph_name",
        "query_count",
        "official_precision",
        "official_recall",
        "official_f1",
        "official_jaccard",
        "senior_micro_precision",
        "senior_micro_recall",
        "senior_micro_f1",
        "senior_nmi",
        "senior_ari",
        "senior_jaccard",
        "official_metric_protocol",
        "prediction_sha256",
        "query_sha256",
        "gt_sha256",
        "prediction_file",
        "query_file",
        "gt_file",
        "metric_contract",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_jaccard",
        "avg_pred_size",
        "avg_gt_size",
        "avg_pred_over_gt",
        "endpoint_hit_rate",
        "near_endpoint_hit_rate",
        "query_missing_rows",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in fieldnames})
        writer.writerow({"graph_name": "MACRO", **{k: macro.get(k) for k in fieldnames if k != "graph_name"}})


if __name__ == "__main__":
    main()
