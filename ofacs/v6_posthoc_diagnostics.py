"""Prediction-frozen V6 diagnostics.

Every function in this module consumes already-written prediction artifacts.
It never runs a model, scores nodes, changes ranking, selects an endpoint, reads
labels/ground truth, or computes official metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from .hierarchical_target_conditioned_framework import (
    validate_protocol_report,
    validate_target_evidence_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prediction_rows(path: Path) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"empty prediction row: {path}:{line_number}")
            rows.append(tuple(int(token) for token in text.split()))
    if not rows:
        raise ValueError(f"prediction file has no rows: {path}")
    return tuple(rows)


def _ordered_hash(rows: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_semantic_hash(rows: tuple[tuple[int, ...], ...]) -> str:
    canonical = tuple(tuple(sorted(set(row))) for row in rows)
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _read_prediction_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"empty prediction JSONL row: {path}:{line_number}")
        payload = json.loads(line)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"prediction metadata is not a mapping: {path}:{line_number}")
        rows.append(metadata)
    return rows


def _summarize_prediction_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    state_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    endpoints: list[int] = []
    for row in rows:
        state = str(row.get("dual_branch_state_profile_id", "missing"))
        reason = str(row.get("dual_branch_search_reason", "missing"))
        state_counts[state] = state_counts.get(state, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        endpoint = row.get("dual_branch_final_prefix_k")
        if isinstance(endpoint, int):
            endpoints.append(endpoint)
    return {
        "available": True,
        "query_count": len(rows),
        "final_endpoint_prefix_positions": endpoints,
        "state_profile_counts": state_counts,
        "search_reason_counts": reason_counts,
        "mean_action_alpha": _mean_numeric(rows, "dual_branch_alpha"),
        "mean_stable_evidence_size": _mean_numeric(rows, "dual_branch_preserved_size"),
        "mean_target_conditioned_evidence_size": _mean_numeric(rows, "dual_branch_adapted_size"),
        "mean_evidence_jaccard": _mean_numeric(rows, "dual_branch_preserved_adapted_jaccard"),
        "mean_best_prefix": _mean_numeric(rows, "dual_branch_best_prefix_k"),
        "mean_final_prefix": _mean_numeric(rows, "dual_branch_final_prefix_k"),
        "mean_final_quality": _mean_numeric(rows, "dual_branch_final_quality"),
        "mean_final_density": _mean_numeric(rows, "dual_branch_final_density"),
        "mean_boundary_quality_margin_previous": _mean_numeric(
            rows, "dual_branch_marginal_quality_from_previous"
        ),
        "mean_boundary_quality_margin_next": _mean_numeric(
            rows, "dual_branch_marginal_quality_to_next"
        ),
        "mean_boundary_density_margin_previous": _mean_numeric(
            rows, "dual_branch_marginal_density_from_previous"
        ),
        "mean_boundary_density_margin_next": _mean_numeric(
            rows, "dual_branch_marginal_density_to_next"
        ),
    }


def build_posthoc_diagnostics(
    *, output_dir: str | Path, graph_name: str, interface_payload: dict[str, Any]
) -> dict[str, Any]:
    root = Path(output_dir)
    pred_path = root / f"{graph_name}.pred"
    resolved_path = root / "fixed_inference_resolved_args.json"
    evidence_path = root / "target_evidence_guidance_report.json"
    prediction_jsonl_path = root / f"{graph_name}.pred.jsonl"
    protocol_path = root / f"{graph_name}.protocol_check.json"
    for required in (pred_path, resolved_path, protocol_path, evidence_path):
        if not required.is_file():
            raise FileNotFoundError(f"prediction must be frozen before diagnostics: {required}")

    rows = _read_prediction_rows(pred_path)
    sizes = [len(row) for row in rows]
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    protocol_validation = validate_protocol_report(
        protocol, expected_record_count=len(rows)
    )
    contract_payload = dict(interface_payload.get("method_contract", {}))
    evidence_contract = dict(contract_payload.get("target_evidence", {}))
    expected_trust = float(evidence_contract.get("evidence_trust", float("nan")))
    validation_contract = SimpleNamespace(
        target_evidence=SimpleNamespace(evidence_trust=expected_trust)
    )
    evidence_validation = validate_target_evidence_report(
        evidence, method_contract=validation_contract
    )
    prediction_metadata = _read_prediction_metadata(prediction_jsonl_path)
    return {
        "protocol": "v58-posthoc-diagnostics-v3",
        "prediction_frozen_before_diagnostics": True,
        "model_forward_repeated": False,
        "scoring_repeated": False,
        "ranking_repeated": False,
        "endpoint_selection_repeated": False,
        "labels_used": protocol_validation["labels_used"],
        "ground_truth_used": protocol_validation["ground_truth_loaded"],
        "official_metrics_used": protocol_validation["metrics_computed"],
        "target_optimizer_steps": evidence_validation["target_optimizer_steps"],
        "target_parameter_updates": 0,
        "runtime_evidence_validation": {
            "protocol_report": protocol_validation,
            "target_evidence_report": evidence_validation,
        },
        "graph_name": graph_name,
        "query_count": len(rows),
        "community_size_min": min(sizes),
        "community_size_max": max(sizes),
        "community_size_mean": sum(sizes) / len(sizes),
        "final_community_sizes": sizes,
        "prediction_metadata_summary": _summarize_prediction_metadata(
            prediction_metadata
        ),
        "prediction_file_sha256": _sha256(pred_path),
        "prediction_ordered_sha256": _ordered_hash(rows),
        "prediction_set_semantic_sha256": _set_semantic_hash(rows),
        "target_evidence": {
            key: evidence.get(key)
            for key in (
                "protocol",
                "strict_one_for_all",
                "checkpoint_density_ratio_state_ignored",
                "guidance_weight",
                "guidance_clip",
                "node_mean_norm",
                "node_variance_mean",
                "neighbor_mean_norm",
                "neighbor_variance_mean",
                "variance_floor",
                "variance_shrinkage",
                "learned_target_estimator_loaded",
                "target_optimizer_steps",
                "trainable_parameter_count",
                "target_labels_used",
                "target_gt_used",
                "target_metrics_used",
            )
            if key in evidence
        },
        "fixed_kernel": {
            "v55_preset": resolved.get("v55_preset"),
            "preset_overrides_applied": resolved.get("preset_overrides_applied"),
            "checkpoint": resolved.get("checkpoint"),
            "dual_preserved_checkpoint": resolved.get("dual_preserved_checkpoint"),
        },
        "prediction_protocol": protocol,
        "paper_interface": interface_payload,
    }


def write_posthoc_diagnostics(
    *, output_dir: str | Path, graph_name: str, interface_payload: dict[str, Any]
) -> Path:
    payload = build_posthoc_diagnostics(
        output_dir=output_dir,
        graph_name=graph_name,
        interface_payload=interface_payload,
    )
    path = Path(output_dir) / "v58_posthoc_diagnostics.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
