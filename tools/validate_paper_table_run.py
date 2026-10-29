#!/usr/bin/env python3
"""Fail closed unless one paper-table run is complete and internally coherent."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _targets(value: str) -> set[str]:
    aliases = {"cs": "cocs", "cocs": "cocs"}
    return {aliases.get(item.strip().lower(), item.strip().lower()) for item in value.split(",") if item.strip()}


def _row_target(row: dict[str, str]) -> str:
    value = (row.get("target") or row.get("graph") or row.get("graph_name") or "").strip().lower()
    return "cocs" if value in {"cs", "cocs"} else value


def _parse_expected(raw: str):
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise ValueError(f"bad --expect value: {raw!r}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--targets", default="wisconsin,cora,citeseer,cs,physics")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval-before-training", action="store_true")
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--expect-json", default="", help="JSON object or @file whose fields must match final_resolved_args")
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    required = [
        "paper_table_status.json",
        "paper_table_best_by_f1.csv",
        "paper_table_best_by_jac.csv",
        "paper_table_epoch_metrics.csv",
        "paper_table_summary.json",
        "paper_table_run_config.json",
        "final_resolved_args.json",
        "joint_source_train/senior_exact_joint_training_summary.json",
    ]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise SystemExit(f"incomplete run; missing={missing}")

    status = json.loads((run / "paper_table_status.json").read_text(encoding="utf-8"))
    if int(status.get("return_code", -1)) != 0:
        raise SystemExit(f"nonzero paper-table status: {status}")

    expected_targets = _targets(args.targets)
    for filename in ("paper_table_best_by_f1.csv", "paper_table_best_by_jac.csv"):
        rows = _rows(run / filename)
        actual = {_row_target(row) for row in rows}
        if actual != expected_targets:
            raise SystemExit(f"{filename} target mismatch: expected={sorted(expected_targets)} actual={sorted(actual)}")
        for row in rows:
            for key in ("official_f1", "official_jaccard"):
                if key in row and row[key] not in (None, "") and not math.isfinite(float(row[key])):
                    raise SystemExit(f"non-finite {key} in {filename}")

    epochs = set(range(0 if args.eval_before_training else 1, int(args.epochs) + 1))
    metric_rows = _rows(run / "paper_table_epoch_metrics.csv")
    actual_pairs = {(int(row["epoch"]), _row_target(row)) for row in metric_rows}
    expected_pairs = {(epoch, target) for epoch in epochs for target in expected_targets}
    if actual_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - actual_pairs)[:10]
        extra_pairs = sorted(actual_pairs - expected_pairs)[:10]
        raise SystemExit(f"epoch metric grid mismatch: missing={missing_pairs} extra={extra_pairs}")

    resolved = json.loads((run / "final_resolved_args.json").read_text(encoding="utf-8"))
    expected_items = [_parse_expected(raw) for raw in args.expect]
    if args.expect_json:
        raw_json = args.expect_json
        if raw_json.startswith("@"):
            expected_payload = json.loads(Path(raw_json[1:]).read_text(encoding="utf-8"))
        else:
            expected_payload = json.loads(raw_json)
        if not isinstance(expected_payload, dict):
            raise SystemExit("--expect-json must resolve to a JSON object")
        expected_items.extend(expected_payload.items())
    for key, expected in expected_items:
        actual = resolved.get(key)
        if isinstance(expected, float) and isinstance(actual, (float, int)):
            matched = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            matched = actual == expected
        if not matched:
            raise SystemExit(f"resolved arg mismatch: {key} expected={expected!r} actual={actual!r}")
    print(json.dumps({"complete": True, "run_dir": str(run), "targets": sorted(expected_targets), "epochs": sorted(epochs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
