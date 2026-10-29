"""Paper-facing OFA-CS extension framework backed by the immutable V575 kernel.

The public method is organized around two scientific problems:

1. target representation shift -> parameter-free target-evidence-guided diffusion;
2. target retrieval-decision shift -> hierarchical target-conditioned search.

This module is part of the formal execution path.  It validates the paper
contract, builds the *unchanged* V575 fixed-evaluation command, and extracts an
explanatory trace from already-frozen prediction artifacts.  It never
recomputes node scores, ranking, endpoints, communities, labels, ground truth,
or official metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

RELEASE_IDENTITY = "OFA-CS V6"
METHOD_IDENTITY = "V575 strict G003_RHO090_CORE125"
TRAINING_CHECKPOINT_SOURCE = "V575_NO_CDSS_CONTROL_SEED012 (A4 no-CDSS/no-SCPC)"
PUBLIC_METHOD_PRESET = "v6_target_evidence_hierarchical_search"
IMMUTABLE_V575_PRESET = "v575_strict_one_for_all_target_guidance"
METHOD_PROTOCOL = "v58-hierarchical-runtime-evidence-gated-v3"
TRACE_PROTOCOL = "v58-hierarchical-runtime-trace-v3"

# Exact validated formal overrides.  Values remain literals so the child V575
# resolved args are not changed by a recomputation or an alternative formula.
EXACT_V575_OVERRIDES: dict[str, float] = {
    "ratio_guidance_weight": 0.03,
    "dual_boundary_plateau_ceiling_ratio": 0.90,
    "dual_state_profile_core_weight_strong": 0.0,
    "dual_state_profile_core_weight_moderate": 0.275,
    "dual_state_profile_core_weight_boundary": 0.25,
    "dual_state_profile_core_weight_shift": 0.10,
    "dual_state_profile_core_weight_stable": 0.125,
}

_PROFILE_TO_SCORE_KEY = {
    "strong_recovery_profile": "dual_state_profile_score_mode_strong",
    "moderate_recall_profile": "dual_state_profile_score_mode_moderate",
    "boundary_preserve_profile": "dual_state_profile_score_mode_boundary",
    "boundary_plateau_precision_profile": "dual_state_profile_score_mode_boundary",
    "boundary_margin_sensitive_profile": "dual_state_profile_score_mode_boundary",
    "shift_adaptive_profile": "dual_state_profile_score_mode_shift",
    "stable_default_profile": "dual_state_profile_score_mode_stable",
}

_PROFILE_TO_CORE_SOURCE_KEY = {
    "strong_recovery_profile": "dual_state_profile_core_source_mode_strong",
    "moderate_recall_profile": "dual_state_profile_core_source_mode_moderate",
    "boundary_preserve_profile": "dual_state_profile_core_source_mode_boundary",
    "boundary_plateau_precision_profile": "dual_state_profile_core_source_mode_boundary",
    "boundary_margin_sensitive_profile": "dual_state_profile_core_source_mode_boundary",
    "shift_adaptive_profile": "dual_state_profile_core_source_mode_shift",
    "stable_default_profile": "dual_state_profile_core_source_mode_stable",
}

_PUBLIC_SCORE_MODE = {
    "default_fusion": "base_evidence_interpolation",
    "weak_adapt_residual": "coverage_compensation",
    "preserve_consensus": "consensus_compensation",
    "adapt_preferred": "task_evidence_preference",
}

_PUBLIC_CORE_SOURCE = {
    "agreement_top": "consensus_supported",
    "agreement_then_profile": "consensus_then_condition_preference",
    "profile_top": "condition_preferred",
    "preserved_top": "stability_supported",
    "adapted_top": "task_supported",
}


@dataclass(frozen=True)
class TargetEvidenceGuidedDiffusionContract:
    mode: str
    parameter_free: bool
    evidence_trust: float
    enable_density_ratio_guidance: bool
    trainable_target_estimator: bool
    target_optimizer_steps: int
    target_parameter_updates: int

    def validate(self) -> "TargetEvidenceGuidedDiffusionContract":
        if self.mode != "diagonal_gaussian":
            raise ValueError("formal target evidence must use diagonal_gaussian moments")
        if self.parameter_free is not True:
            raise ValueError("formal target evidence must be parameter-free")
        if self.evidence_trust != 0.03:
            raise ValueError("formal G003 evidence trust must be exactly 0.03")
        if self.enable_density_ratio_guidance:
            raise ValueError("the historical trainable density-ratio path must remain disabled")
        if self.trainable_target_estimator:
            raise ValueError("target-side trainable estimators are forbidden")
        if self.target_optimizer_steps != 0 or self.target_parameter_updates != 0:
            raise ValueError("target-side optimization or updates are forbidden")
        return self


@dataclass(frozen=True)
class HierarchicalSearchContract:
    target_condition_source: str
    query_fusion_control: str
    query_core_control: str
    score_composition: str
    boundary_control: str
    target_level: str
    query_level: str
    boundary_level: str
    final_output_count: int
    target_side_learning: bool

    def validate(self) -> "HierarchicalSearchContract":
        expected = {
            "target_condition_source": "unlabeled_query_workload_statistics",
            "query_fusion_control": "per_query_continuous_alpha",
            "query_core_control": "target_conditioned_query_evidence",
            "score_composition": "base_interpolation_plus_bounded_condition_correction",
            "boundary_control": "bidirectional_expand_keep_contract",
            "target_level": "target_level_retrieval_conditioning",
            "query_level": "query_level_evidence_and_core_calibration",
            "boundary_level": "boundary_level_endpoint_regulation",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"invalid hierarchical search contract {field}={getattr(self, field)!r}")
        if self.final_output_count != 1:
            raise ValueError("formal search must output one final community")
        if self.target_side_learning:
            raise ValueError("target-conditioned search must not learn on the target graph")
        return self


@dataclass(frozen=True)
class MethodContract:
    protocol: str
    release_identity: str
    method_identity: str
    training_checkpoint_source: str
    public_method_preset: str
    immutable_v575_preset: str
    target_evidence: TargetEvidenceGuidedDiffusionContract
    hierarchical_search: HierarchicalSearchContract
    exact_v575_overrides: Mapping[str, float]
    formal_constraints: Mapping[str, bool]

    @classmethod
    def from_json(cls, path: str | Path) -> "MethodContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        evidence = dict(payload.get("target_evidence_guided_diffusion", {}))
        search = dict(payload.get("hierarchical_target_conditioned_search", {}))
        contract = cls(
            protocol=str(payload.get("protocol")),
            release_identity=str(payload.get("release_identity")),
            method_identity=str(payload.get("method_identity")),
            training_checkpoint_source=str(payload.get("training_checkpoint_source")),
            public_method_preset=str(payload.get("public_method_preset")),
            immutable_v575_preset=str(payload.get("immutable_v575_preset")),
            target_evidence=TargetEvidenceGuidedDiffusionContract(
                mode=str(evidence.get("mode")),
                parameter_free=bool(evidence.get("parameter_free")),
                evidence_trust=float(evidence.get("evidence_trust")),
                enable_density_ratio_guidance=bool(evidence.get("enable_density_ratio_guidance")),
                trainable_target_estimator=bool(evidence.get("trainable_target_estimator")),
                target_optimizer_steps=int(evidence.get("target_optimizer_steps", -1)),
                target_parameter_updates=int(evidence.get("target_parameter_updates", -1)),
            ),
            hierarchical_search=HierarchicalSearchContract(
                target_condition_source=str(search.get("target_condition_source")),
                query_fusion_control=str(search.get("query_fusion_control")),
                query_core_control=str(search.get("query_core_control")),
                score_composition=str(search.get("score_composition")),
                boundary_control=str(search.get("boundary_control")),
                target_level=str(dict(search.get("public_hierarchy", {})).get("target_level")),
                query_level=str(dict(search.get("public_hierarchy", {})).get("query_level")),
                boundary_level=str(dict(search.get("public_hierarchy", {})).get("boundary_level")),
                final_output_count=int(search.get("final_output_count", -1)),
                target_side_learning=bool(search.get("target_side_learning")),
            ),
            exact_v575_overrides={
                str(k): float(v)
                for k, v in dict(payload.get("private_v575_overrides", {})).items()
            },
            formal_constraints={
                str(k): bool(v)
                for k, v in dict(payload.get("formal_constraints", {})).items()
            },
        )
        return contract.validate()

    def validate(self) -> "MethodContract":
        if self.protocol != METHOD_PROTOCOL:
            raise ValueError(f"unexpected method protocol: {self.protocol}")
        if self.release_identity != RELEASE_IDENTITY:
            raise ValueError(f"unexpected release identity: {self.release_identity}")
        if self.method_identity != METHOD_IDENTITY:
            raise ValueError(f"unexpected method identity: {self.method_identity}")
        if self.training_checkpoint_source != TRAINING_CHECKPOINT_SOURCE:
            raise ValueError("formal A4 checkpoint provenance is missing or changed")
        if self.public_method_preset != PUBLIC_METHOD_PRESET:
            raise ValueError("public method preset does not name the hierarchical method")
        if self.immutable_v575_preset != IMMUTABLE_V575_PRESET:
            raise ValueError("immutable V575 preset changed")
        self.target_evidence.validate()
        self.hierarchical_search.validate()
        if dict(self.exact_v575_overrides) != EXACT_V575_OVERRIDES:
            raise ValueError(
                "private V575 overrides differ from validated G003/RHO090/CORE125 values"
            )
        forbidden_true = {
            "target_labels",
            "target_ground_truth",
            "official_metrics_in_prediction",
            "target_optimizer",
            "target_parameter_updates",
            "dataset_name_rules",
            "new_scoring_kernel",
            "new_endpoint_kernel",
        }
        bad = sorted(k for k in forbidden_true if self.formal_constraints.get(k) is not False)
        if bad:
            raise ValueError(f"formal constraints must be false: {bad}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImmutableV575ExecutionPlan:
    graph_name: str
    checkpoint: str
    stable_evidence_checkpoint: str
    dataset_root: str
    output_dir: str
    seed: int
    command: tuple[str, ...]

    def run(self, *, cwd: str | Path) -> None:
        subprocess.run(list(self.command), cwd=Path(cwd), check=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_immutable_v575_execution_plan(
    *,
    project_root: str | Path,
    python_bin: str,
    graph_name: str,
    checkpoint: str | Path,
    stable_evidence_checkpoint: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    device: str,
    seed: int,
    chunk_size: int = 4096,
    deterministic: bool = True,
) -> ImmutableV575ExecutionPlan:
    root = Path(project_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    stable_path = Path(stable_evidence_checkpoint).resolve()
    data_path = Path(dataset_root).resolve()
    output_path = Path(output_dir).resolve()
    command = [
        str(python_bin),
        str(root / "tools" / "evaluate_fixed_checkpoint_ofacs.py"),
        "--graph-name",
        str(graph_name),
        "--checkpoint",
        str(checkpoint_path),
        "--dataset-root",
        str(data_path),
        "--output-dir",
        str(output_path),
        "--device",
        str(device),
        "--chunk-size",
        str(int(chunk_size)),
        "--embedding-export-format",
        "shards",
        "--dims",
        "64",
        "--hops",
        "4",
        "--local-topk",
        "10",
        "--pos-count",
        "10",
        "--neg-count",
        "10",
        "--senior-cache-mode",
        "required",
        "--feature-projection-mode",
        "identity",
        "--t-eval",
        "50",
        "--v55-preset",
        IMMUTABLE_V575_PRESET,
        "--preset-override-json",
        "@" + str((root / "configs" / "inference_override.json").resolve()),
        "--dual-branch-score",
        "--dual-preserved-checkpoint",
        str(stable_path),
        # Runtime diagnostics remain disabled to preserve the validated execution
        # path.  Paper-facing traces are extracted after predictions are frozen.
        "--no-write-candidate-diagnostics",
        "--no-write-dual-branch-diagnostics",
        "--no-write-ratio-guidance-diagnostics",
        "--seed",
        str(int(seed)),
    ]
    if deterministic:
        command.append("--deterministic")
    return ImmutableV575ExecutionPlan(
        graph_name=str(graph_name),
        checkpoint=str(checkpoint_path),
        stable_evidence_checkpoint=str(stable_path),
        dataset_root=str(data_path),
        output_dir=str(output_path),
        seed=int(seed),
        command=tuple(command),
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ordered_hash(rows: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _set_semantic_hash(rows: Sequence[Sequence[int]]) -> str:
    canonical = [sorted(set(int(node) for node in row)) for row in rows]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _read_prediction_rows(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"empty prediction row: {path}:{line_number}")
        rows.append([int(token) for token in line.split()])
    if not rows:
        raise ValueError(f"prediction file has no rows: {path}")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"empty JSONL row: {path}:{line_number}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _numeric(metadata: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(metadata: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mean(values: Iterable[float]) -> float:
    rows = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(rows) / len(rows)) if rows else 0.0


def _score_mode(profile_id: str, resolved: Mapping[str, Any]) -> tuple[str, str]:
    if profile_id in {"boundary_preserve_profile", "boundary_margin_sensitive_profile"} and bool(
        resolved.get("dual_hierarchical_boundary_state_enabled", resolved.get("hierarchical_boundary_state_enabled", False))
    ):
        internal = "default_fusion"
    else:
        key = _PROFILE_TO_SCORE_KEY.get(profile_id)
        internal = str(resolved.get(key, "default_fusion")) if key else "default_fusion"
    return _PUBLIC_SCORE_MODE.get(internal, "bounded_condition_correction"), internal


def _core_source(profile_id: str, metadata: Mapping[str, Any], resolved: Mapping[str, Any]) -> tuple[str, str]:
    key = _PROFILE_TO_CORE_SOURCE_KEY.get(profile_id)
    internal = str(resolved.get(key, metadata.get("dual_branch_state_profile_core_source_mode", "profile_top")))
    return _PUBLIC_CORE_SOURCE.get(internal, "condition_preferred"), internal


def _boundary_action(metadata: Mapping[str, Any]) -> str:
    if bool(metadata.get("dual_branch_ceiling_accepted", False)):
        return "contract"
    if bool(metadata.get("dual_branch_floor_accepted", False)):
        return "expand"
    reason = str(metadata.get("dual_branch_search_reason", ""))
    if "fallback" in reason or reason == "senior_global_search":
        return "fallback"
    return "keep"


def _optional_mean(
    metadata_rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> tuple[float | None, str | None]:
    values: list[float] = []
    used_key: str | None = None
    for row in metadata_rows:
        for key in keys:
            if key not in row:
                continue
            value = _numeric(row, key, float("nan"))
            if math.isfinite(value):
                values.append(value)
                used_key = key
                break
    return (_mean(values), used_key) if values else (None, None)


def _observed_target_runtime_statistics(
    metadata_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    ratios: list[float] = []
    overlaps: list[float] = []
    alphas: list[float] = []
    for row in metadata_rows:
        preserved = max(_numeric(row, "dual_branch_preserved_size", 1.0), 1.0)
        adapted = max(_numeric(row, "dual_branch_adapted_size", 0.0), 0.0)
        ratios.append(adapted / preserved)
        overlaps.append(_numeric(row, "dual_branch_preserved_adapted_jaccard", 0.0))
        alphas.append(_numeric(row, "dual_branch_alpha", 0.0))

    optional_specs = {
        "target_density_delta": (
            "dual_branch_state_profile_target_density_delta",
            "dual_branch_target_density_delta",
        ),
        "target_branch_disagreement": (
            "dual_branch_state_profile_target_disagreement",
            "dual_branch_target_disagreement",
        ),
        "target_score_entropy": (
            "dual_branch_state_profile_target_entropy",
            "dual_branch_target_entropy",
        ),
    }
    optional_values: dict[str, float] = {}
    runtime_sources: dict[str, str] = {}
    unavailable: list[dict[str, str]] = []
    for public_name, keys in optional_specs.items():
        value, source_key = _optional_mean(metadata_rows, keys)
        if value is None or source_key is None:
            unavailable.append(
                {
                    "field": public_name,
                    "reason": "not exported by formal prediction metadata; no value is inferred",
                }
            )
        else:
            optional_values[public_name] = value
            runtime_sources[public_name] = source_key

    observed: dict[str, Any] = {
        "query_count": len(metadata_rows),
        "mean_task_to_reference_candidate_size_ratio": _mean(ratios),
        "mean_branch_candidate_overlap_jaccard": _mean(overlaps),
        "mean_query_task_evidence_weight": _mean(alphas),
        "target_underexpansion_rate": _mean(
            _numeric(row, "dual_branch_target_underfloor_rate", 0.0)
            for row in metadata_rows
        ),
        "target_overexpansion_rate": _mean(
            _numeric(row, "dual_branch_target_overexpansion_rate", 0.0)
            for row in metadata_rows
        ),
        "boundary_plateau_query_rate": _mean(
            _numeric(row, "dual_branch_boundary_query_plateau_rate", 0.0)
            for row in metadata_rows
        ),
        "target_branch_jaccard_mean": _mean(
            _numeric(row, "dual_branch_target_branch_jaccard_mean", 0.0)
            for row in metadata_rows
        ),
        "target_probe_relative_quality_drop": _mean(
            _numeric(row, "dual_branch_target_probe_rel_quality_drop", 0.0)
            for row in metadata_rows
        ),
        "target_probe_relative_density_drop": _mean(
            _numeric(row, "dual_branch_target_probe_rel_density_drop", 0.0)
            for row in metadata_rows
        ),
        "optional_runtime_statistics": optional_values,
        "runtime_source_keys": runtime_sources,
    }
    return observed, unavailable


def _interpretable_method_summary(observed: Mapping[str, Any]) -> dict[str, Any]:
    ratio = float(observed.get("mean_task_to_reference_candidate_size_ratio", 0.0))
    overlap = float(observed.get("mean_branch_candidate_overlap_jaccard", 0.0))
    alpha = float(observed.get("mean_query_task_evidence_weight", 0.0))
    under = float(observed.get("target_underexpansion_rate", 0.0))
    over = float(observed.get("target_overexpansion_rate", 0.0))
    plateau = float(observed.get("boundary_plateau_query_rate", 0.0))
    return {
        "interpretation_scope": (
            "posthoc paper-facing summary derived only from observed runtime statistics; "
            "not an additional V575 controller"
        ),
        "coverage_demand_indicator": max(under, max(0.0, 1.0 - ratio)),
        "task_evidence_support_indicator": alpha,
        "saturation_risk_indicator": max(over, max(0.0, ratio - 1.0)),
        "boundary_uncertainty_indicator": max(plateau, 1.0 - overlap),
    }



def validate_protocol_report(
    protocol: Mapping[str, Any],
    *,
    expected_record_count: int | None = None,
) -> dict[str, Any]:
    """Validate the actual frozen One-for-All protocol report.

    Contract declarations are not accepted as substitutes for runtime evidence.
    Any target label, ground-truth, or metric use is a hard error.
    """

    if not isinstance(protocol, Mapping):
        raise ValueError("protocol report must be a mapping")
    labels_used = bool(protocol.get("labels_used", True))
    ground_truth_loaded = bool(
        protocol.get("ground_truth_loaded", protocol.get("ground_truth_used", True))
    )
    metrics_computed = bool(
        protocol.get("metrics_computed", protocol.get("official_metrics_used", True))
    )
    record_count = int(protocol.get("record_count", -1))
    errors: list[str] = []
    if labels_used:
        errors.append("target labels were used during frozen prediction")
    if ground_truth_loaded:
        errors.append("target ground truth was loaded during frozen prediction")
    if metrics_computed:
        errors.append("official metrics were computed during frozen prediction")
    if expected_record_count is not None and record_count != int(expected_record_count):
        errors.append(
            f"protocol record_count={record_count} does not match prediction query count="
            f"{int(expected_record_count)}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "passed": True,
        "labels_used": labels_used,
        "ground_truth_loaded": ground_truth_loaded,
        "metrics_computed": metrics_computed,
        "record_count": record_count,
    }


def validate_target_evidence_report(
    evidence: Mapping[str, Any],
    *,
    method_contract: MethodContract,
) -> dict[str, Any]:
    """Validate the actual target-evidence report; never fill runtime facts from defaults."""

    if not isinstance(evidence, Mapping):
        raise ValueError("target evidence report must be a mapping")
    expected_false = {
        "learned_target_estimator_loaded": False,
        "target_labels_used": False,
        "target_gt_used": False,
        "target_metrics_used": False,
    }
    errors: list[str] = []
    if evidence.get("protocol") != "strict-one-for-all-target-moment-evidence-v1":
        errors.append("unexpected or missing target evidence protocol")
    if evidence.get("strict_one_for_all") is not True:
        errors.append("target evidence report does not assert strict_one_for_all=true")
    for key, expected in expected_false.items():
        if evidence.get(key) is not expected:
            errors.append(f"{key} must be {expected!r}")
    if int(evidence.get("target_optimizer_steps", -1)) != 0:
        errors.append("target_optimizer_steps must be zero")
    if int(evidence.get("trainable_parameter_count", -1)) != 0:
        errors.append("trainable_parameter_count must be zero")
    if evidence.get("checkpoint_density_ratio_state_ignored") is not True:
        errors.append("checkpoint density-ratio state must be explicitly ignored")
    if float(evidence.get("guidance_weight", float("nan"))) != float(
        method_contract.target_evidence.evidence_trust
    ):
        errors.append("guidance_weight does not match the formal G003 evidence trust")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "passed": True,
        "protocol": str(evidence["protocol"]),
        "strict_one_for_all": True,
        "learned_target_estimator_loaded": False,
        "target_optimizer_steps": 0,
        "trainable_parameter_count": 0,
        "target_labels_used": False,
        "target_gt_used": False,
        "target_metrics_used": False,
        "checkpoint_density_ratio_state_ignored": True,
        "guidance_weight": float(evidence["guidance_weight"]),
    }


def _boundary_probe_trace(metadata: Mapping[str, Any], action: str) -> dict[str, Any]:
    floor_probe = {
        "relative_quality_drop": _numeric(metadata, "dual_branch_rel_quality_drop"),
        "relative_density_drop": _numeric(metadata, "dual_branch_rel_density_drop"),
    }
    ceiling_probe = {
        "relative_quality_drop": _numeric(metadata, "dual_branch_ceiling_rel_quality_drop"),
        "relative_density_drop": _numeric(metadata, "dual_branch_ceiling_rel_density_drop"),
    }
    selected_source: str | None
    selected: Mapping[str, Any] | None
    if action == "expand":
        selected_source, selected = "floor_recovery_probe", floor_probe
    elif action == "contract":
        selected_source, selected = "ceiling_contraction_probe", ceiling_probe
    else:
        selected_source, selected = None, None
    return {
        "floor_recovery_probe": floor_probe,
        "ceiling_contraction_probe": ceiling_probe,
        "selected_probe_source": selected_source,
        "selected_relative_quality_drop": (
            None if selected is None else selected["relative_quality_drop"]
        ),
        "selected_relative_density_drop": (
            None if selected is None else selected["relative_density_drop"]
        ),
    }


def build_hierarchical_search_trace(
    *,
    output_dir: str | Path,
    graph_name: str,
    method_contract: MethodContract,
    execution_plan: ImmutableV575ExecutionPlan,
) -> dict[str, Any]:
    """Extract paper-facing state from frozen artifacts without changing prediction."""

    root = Path(output_dir)
    pred_path = root / f"{graph_name}.pred"
    jsonl_path = root / f"{graph_name}.pred.jsonl"
    resolved_path = root / "fixed_inference_resolved_args.json"
    evidence_path = root / "target_evidence_guidance_report.json"
    protocol_path = root / f"{graph_name}.protocol_check.json"
    for required in (pred_path, jsonl_path, resolved_path, protocol_path, evidence_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"frozen prediction artifact required before trace extraction: {required}"
            )

    prediction_rows = _read_prediction_rows(pred_path)
    payload_rows = _read_jsonl(jsonl_path)
    if len(prediction_rows) != len(payload_rows):
        raise ValueError("plain prediction and JSONL query counts differ")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    protocol_validation = validate_protocol_report(
        protocol, expected_record_count=len(prediction_rows)
    )
    evidence_validation = validate_target_evidence_report(
        evidence, method_contract=method_contract
    )

    metadata_rows: list[dict[str, Any]] = []
    query_traces: list[dict[str, Any]] = []
    score_mode_counts: dict[str, int] = {}
    internal_score_mode_counts: dict[str, int] = {}
    core_source_counts: dict[str, int] = {}
    boundary_action_counts = {"expand": 0, "keep": 0, "contract": 0, "fallback": 0}
    internal_profile_counts: dict[str, int] = {}
    core_reliability_enabled = bool(
        resolved.get("dual_state_profile_core_reliability_enabled", False)
    )

    for index, payload in enumerate(payload_rows):
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"prediction metadata is not an object at query {index}")
        metadata_rows.append(metadata)
        profile_id = str(metadata.get("dual_branch_state_profile_id", "disabled"))
        internal_profile_counts[profile_id] = internal_profile_counts.get(profile_id, 0) + 1
        public_score_mode, internal_score_mode = _score_mode(profile_id, resolved)
        score_mode_counts[public_score_mode] = score_mode_counts.get(public_score_mode, 0) + 1
        internal_score_mode_counts[internal_score_mode] = internal_score_mode_counts.get(internal_score_mode, 0) + 1
        public_core_source, internal_core_source = _core_source(profile_id, metadata, resolved)
        core_source_counts[public_core_source] = core_source_counts.get(public_core_source, 0) + 1
        action = _boundary_action(metadata)
        boundary_action_counts[action] += 1

        reported_core_weight = _numeric(metadata, "dual_branch_state_profile_core_weight")
        query_traces.append(
            {
                "query_index": int(payload.get("query_index", index)),
                "query_nodes_local": payload.get("query_nodes_local", []),
                "final_community_size": len(prediction_rows[index]),
                "target_level_reference": {
                    "uses_shared_unlabeled_target_condition": True,
                    "private_internal_profile_id": profile_id,
                },
                "query_level": {
                    "branch_evidence_probing": {
                        "reference_candidate_size": _integer(metadata, "dual_branch_preserved_size"),
                        "target_calibrated_candidate_size": _integer(metadata, "dual_branch_adapted_size"),
                        "candidate_overlap_jaccard": _numeric(
                            metadata, "dual_branch_preserved_adapted_jaccard"
                        ),
                    },
                    "continuous_evidence_fusion": {
                        "continuous_alpha": _numeric(metadata, "dual_branch_alpha"),
                        "alpha_mode": str(metadata.get("dual_branch_alpha_mode", "adaptive")),
                        "target_conditioned_alpha_bias": _numeric(
                            metadata, "dual_branch_state_profile_alpha_bias"
                        ),
                        "effective_alpha": _numeric(
                            metadata,
                            "dual_branch_state_profile_alpha_final",
                            _numeric(metadata, "dual_branch_alpha"),
                        ),
                        "alpha_bound_reason": str(
                            metadata.get("dual_branch_state_profile_alpha_bound_reason", "")
                        ),
                    },
                    "query_core_summary": {
                        "configured_evidence_preference": public_core_source,
                        "reported_core_weight": reported_core_weight,
                        "core_reliability_enabled": core_reliability_enabled,
                        "effective_core_weight_available": False,
                        "effective_core_weight": None,
                        "node_count": _integer(
                            metadata, "dual_branch_state_profile_core_node_count"
                        ),
                        "agreement_node_count": _integer(
                            metadata, "dual_branch_state_profile_core_agreement_node_count"
                        ),
                        "actual_core_node_ids_available": False,
                        "effective_source_after_fallback_available": False,
                    },
                    "condition_selective_score_mode": {
                        "public_mode": public_score_mode,
                        "base_interpolation_uses_query_alpha": True,
                        "bounded_condition_correction_selected": (
                            internal_score_mode != "default_fusion"
                        ),
                        "actual_correction_magnitude_available": False,
                    },
                },
                "boundary_level": {
                    "action": action,
                    "initial_best_prefix": _integer(metadata, "dual_branch_best_prefix_k"),
                    "final_prefix": _integer(metadata, "dual_branch_final_prefix_k"),
                    "floor_triggered": bool(metadata.get("dual_branch_floor_triggered", False)),
                    "floor_accepted": bool(metadata.get("dual_branch_floor_accepted", False)),
                    "ceiling_triggered": bool(metadata.get("dual_branch_ceiling_triggered", False)),
                    "ceiling_accepted": bool(metadata.get("dual_branch_ceiling_accepted", False)),
                    "probe_metrics": _boundary_probe_trace(metadata, action),
                    "search_reason": str(metadata.get("dual_branch_search_reason", "")),
                    "final_community_size": len(prediction_rows[index]),
                },
                "private_v575_compatibility": {
                    "internal_profile_id": profile_id,
                    "internal_profile_reason": str(
                        metadata.get("dual_branch_state_profile_reason", "")
                    ),
                    "internal_score_mode": internal_score_mode,
                    "configured_internal_core_source_mode": internal_core_source,
                },
            }
        )

    observed, unavailable = _observed_target_runtime_statistics(metadata_rows)
    interpretation = _interpretable_method_summary(observed)
    unavailable.extend(
        [
            {
                "field": "actual_core_node_ids",
                "reason": "present inside V575 runtime state but not exported by formal prediction metadata",
            },
            {
                "field": "effective_core_source_after_fallback",
                "reason": "formal prediction metadata exposes configured source preference only",
            },
            {
                "field": "actual_effective_core_weight",
                "reason": "formal prediction metadata exposes reported/configured weight only",
            },
            {
                "field": "actual_score_correction_magnitude",
                "reason": "formal prediction metadata exposes selected score mode but not contribution values",
            },
        ]
    )

    alphas = [_numeric(row, "dual_branch_alpha") for row in metadata_rows]
    core_weights = [_numeric(row, "dual_branch_state_profile_core_weight") for row in metadata_rows]
    core_sizes = [_integer(row, "dual_branch_state_profile_core_node_count") for row in metadata_rows]

    return {
        "protocol": TRACE_PROTOCOL,
        "claim_scope": (
            "runtime-aligned paper reorganization of immutable V575 behavior; "
            "observed state and posthoc interpretation are explicitly separated"
        ),
        "prediction_frozen_before_trace": True,
        "model_forward_repeated": False,
        "scoring_repeated": False,
        "ranking_repeated": False,
        "endpoint_selection_repeated": False,
        "labels_used": protocol_validation["labels_used"],
        "ground_truth_used": protocol_validation["ground_truth_loaded"],
        "official_metrics_used": protocol_validation["metrics_computed"],
        "runtime_evidence_validation": {
            "protocol_report": protocol_validation,
            "target_evidence_report": evidence_validation,
        },
        "method_contract": method_contract.to_dict(),
        "execution_plan": execution_plan.to_dict(),
        "prediction": {
            "graph_name": graph_name,
            "query_count": len(prediction_rows),
            "ordered_prediction_sha256": _ordered_hash(prediction_rows),
            "set_semantic_prediction_sha256": _set_semantic_hash(prediction_rows),
            "community_size_min": min(len(row) for row in prediction_rows),
            "community_size_max": max(len(row) for row in prediction_rows),
            "community_size_mean": _mean(len(row) for row in prediction_rows),
        },
        "target_evidence_guided_diffusion": {
            "mode": method_contract.target_evidence.mode,
            "parameter_free": True,
            "enable_density_ratio_guidance": False,
            "historical_weight_field": "ratio_guidance_weight",
            "evidence_trust": float(evidence["guidance_weight"]),
            "guidance_clip": evidence.get("guidance_clip"),
            "variance_floor": evidence.get("variance_floor"),
            "variance_shrinkage": evidence.get("variance_shrinkage"),
            "node_mean_norm": evidence.get("node_mean_norm"),
            "node_variance_mean": evidence.get("node_variance_mean"),
            "neighbor_mean_norm": evidence.get("neighbor_mean_norm"),
            "neighbor_variance_mean": evidence.get("neighbor_variance_mean"),
            "target_optimizer_steps": int(evidence["target_optimizer_steps"]),
            "trainable_parameter_count": int(evidence["trainable_parameter_count"]),
            "learned_target_estimator_loaded": bool(
                evidence["learned_target_estimator_loaded"]
            ),
            "checkpoint_density_ratio_state_ignored": bool(
                evidence["checkpoint_density_ratio_state_ignored"]
            ),
        },
        "hierarchical_target_conditioned_search": {
            "public_hierarchy": {
                "target_level": "target_level_retrieval_conditioning",
                "query_level": "query_level_evidence_and_core_calibration",
                "boundary_level": "boundary_level_endpoint_regulation",
            },
            "target_level": {
                "observed_runtime_statistics": observed,
                "interpretable_method_summary": interpretation,
                "unavailable_runtime_fields": unavailable,
            },
            "query_level": {
                "continuous_evidence_fusion_summary": {
                    "control_is_per_query_continuous": True,
                    "alpha_min": min(alphas) if alphas else 0.0,
                    "alpha_max": max(alphas) if alphas else 0.0,
                    "alpha_mean": _mean(alphas),
                },
                "query_core_summary": {
                    "configured_core_source_counts": core_source_counts,
                    "reported_weight_min": min(core_weights) if core_weights else 0.0,
                    "reported_weight_max": max(core_weights) if core_weights else 0.0,
                    "reported_weight_mean": _mean(core_weights),
                    "core_node_count_mean": _mean(core_sizes),
                    "actual_core_node_ids_available": False,
                    "effective_source_after_fallback_available": False,
                },
                "condition_selective_score_summary": {
                    "public_mode_counts": score_mode_counts,
                    "base_interpolation_present": True,
                    "condition_selected_correction_present": any(
                        mode != "default_fusion" for mode in internal_score_mode_counts
                    ),
                    "actual_correction_magnitude_available": False,
                },
            },
            "boundary_level": {
                "supported_actions": ["expand", "keep", "contract", "fallback"],
                "action_counts": boundary_action_counts,
                "expand_is_underexpansion_recovery": True,
                "contract_is_overexpansion_suppression": True,
            },
            "condition_core_boundary_coupling": {
                "coarse_target_condition_precedes_query_calibration": True,
                "query_core_and_score_precede_boundary_refinement": True,
                "boundary_state_can_expand_keep_contract_or_fallback": True,
            },
            "single_final_community": True,
        },
        "query_traces": query_traces,
        "private_v575_compatibility": {
            "internal_profile_counts": internal_profile_counts,
            "internal_score_mode_counts": internal_score_mode_counts,
            "resolved_v575_preset": resolved.get("v55_preset"),
            "resolved_v575_overrides": resolved.get("preset_overrides_applied"),
            "checkpoint": resolved.get("checkpoint"),
            "stable_reference_checkpoint": resolved.get("dual_preserved_checkpoint"),
            "runtime_diagnostics_enabled": False,
        },
        "prediction_protocol": protocol,
    }

def write_hierarchical_search_trace(
    *,
    output_dir: str | Path,
    graph_name: str,
    method_contract: MethodContract,
    execution_plan: ImmutableV575ExecutionPlan,
) -> Path:
    payload = build_hierarchical_search_trace(
        output_dir=output_dir,
        graph_name=graph_name,
        method_contract=method_contract,
        execution_plan=execution_plan,
    )
    path = Path(output_dir) / "v58_hierarchical_search_trace.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
