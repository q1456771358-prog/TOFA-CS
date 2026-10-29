"""Fixed-inference helpers for V5 embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .chunked_scoring import chunked_query_cosine
from .community_output import (
    CommunityPredictionRecord,
    build_prediction_records_from_chunked_scores,
    write_plain_pred,
    write_prediction_jsonl,
    write_protocol_check,
)
from .fixed_checkpoint import FixedCheckpointStatus, write_fixed_inference_status
from .embedding_store import chunked_query_cosine_from_embedding_shards
from .multiview_search import MultiViewSelectionConfig, build_prediction_records_from_multiview_scores
from .dual_branch_scoring import DualBranchGateConfig
from .shift_adaptive_community_search import ShiftAdaptiveModularitySearch
from .target_calibrated_representation import TargetCalibratedRepresentation
from .senior_compat import SeniorQueryBatch


@dataclass(frozen=True)
class FixedInferenceOutputs:
    records: tuple[CommunityPredictionRecord, ...]
    jsonl_path: Path
    pred_path: Path
    protocol_check_path: Path
    status_path: Path


def run_fixed_inference_from_embeddings(
    *,
    graph_name: str,
    node_embeddings: torch.Tensor,
    query_batch: SeniorQueryBatch | Sequence[Sequence[int]],
    output_dir: str | Path,
    checkpoint_path: str | Path,
    global_node_ids: Sequence[int] | None = None,
    chunk_size: int = 4096,
    score_calibration: str = "none",
) -> FixedInferenceOutputs:
    """Run chunked cosine + senior GlobalSearch and write prediction files.

    This function computes no metrics and reads no ground truth.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_result = chunked_query_cosine(node_embeddings, query_batch, chunk_size=chunk_size, score_calibration=score_calibration)
    records = build_prediction_records_from_chunked_scores(
        graph_name=graph_name,
        score_result=score_result,
        global_node_ids=global_node_ids,
        metadata={"score_source": "v5_senior_mean_cosine_chunked"},
    )
    jsonl_path = write_prediction_jsonl(records, out_dir / f"{graph_name}.pred.jsonl")
    pred_path = write_plain_pred(records, out_dir / f"{graph_name}.pred", use_global_ids=global_node_ids is not None)
    check_path = write_protocol_check(records, out_dir / f"{graph_name}.protocol_check.json")
    status_path = write_fixed_inference_status(
        FixedCheckpointStatus(
            checkpoint_path=str(checkpoint_path),
            split="unseen_inference",
            graph_name=graph_name,
            labels_used=False,
            metrics_computed=False,
            checkpoint_selection_from_unseen=False,
            unseen_metrics_used=False,
            ground_truth_loaded=False,
            notes=("fixed checkpoint inference", "senior GlobalSearch", "no metrics"),
        ),
        out_dir / f"{graph_name}.fixed_inference_status.json",
    )
    return FixedInferenceOutputs(
        records=tuple(records),
        jsonl_path=jsonl_path,
        pred_path=pred_path,
        protocol_check_path=check_path,
        status_path=status_path,
    )


def run_fixed_inference_from_dual_branch_embedding_shards(
    *,
    graph_name: str,
    preserved_manifest: str | Path,
    adapted_manifest: str | Path,
    query_batch: SeniorQueryBatch,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    global_node_ids: Sequence[int] | None = None,
    chunk_size: int = 4096,
    runtime_jsonl: str | Path | None = None,
    dual_config: DualBranchGateConfig | None = None,
    write_dual_diagnostics: bool = True,
    edge_index=None,
) -> FixedInferenceOutputs:
    """Run the final integrated fixed-inference path.

    The public arguments retain the historical manifest names for script and
    checkpoint compatibility.  Internally, the manifests are treated as the
    complementary outputs of target-calibrated representation generation and
    are consumed by one shift-adaptive modularity-search stage.  This function
    reads no target ground truth and computes no metrics; ground truth remains
    restricted to the offline evaluator in the paper-table runner.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = (dual_config or DualBranchGateConfig(enabled=True)).validate()
    representations = TargetCalibratedRepresentation.from_manifests(
        reference_manifest=preserved_manifest,
        adapted_manifest=adapted_manifest,
    )
    searcher = ShiftAdaptiveModularitySearch(cfg)
    search_result = searcher.retrieve(
        graph_name=graph_name,
        representations=representations,
        query_batch=query_batch,
        global_node_ids=global_node_ids,
        chunk_size=chunk_size,
        runtime_jsonl=runtime_jsonl,
        edge_index=edge_index,
    )
    records = search_result.records
    if write_dual_diagnostics:
        searcher.write_diagnostics(out_dir / f"{graph_name}.dual_branch_queries.jsonl", search_result)
    jsonl_path = write_prediction_jsonl(records, out_dir / f"{graph_name}.pred.jsonl")
    pred_path = write_plain_pred(records, out_dir / f"{graph_name}.pred", use_global_ids=global_node_ids is not None)
    check_path = write_protocol_check(records, out_dir / f"{graph_name}.protocol_check.json")
    status_path = write_fixed_inference_status(
        FixedCheckpointStatus(
            checkpoint_path=str(checkpoint_path),
            split="unseen_inference",
            graph_name=graph_name,
            labels_used=False,
            metrics_computed=False,
            checkpoint_selection_from_unseen=False,
            unseen_metrics_used=False,
            ground_truth_loaded=False,
            notes=("fixed checkpoint inference", "integrated target-calibrated representation and shift-adaptive search", "embedding shards", "no metrics"),
        ),
        out_dir / f"{graph_name}.fixed_inference_status.json",
    )
    return FixedInferenceOutputs(
        records=tuple(records),
        jsonl_path=jsonl_path,
        pred_path=pred_path,
        protocol_check_path=check_path,
        status_path=status_path,
    )


def run_fixed_inference_from_multiview_embedding_shards(
    *,
    graph_name: str,
    embedding_manifests: Sequence[tuple[str, str | Path, str]],
    query_batch: SeniorQueryBatch,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    global_node_ids: Sequence[int] | None = None,
    chunk_size: int = 4096,
    runtime_jsonl: str | Path | None = None,
    selection_config: MultiViewSelectionConfig | None = None,
    view_profiles: dict[str, tuple[str, str]] | None = None,
    edge_index: torch.Tensor | None = None,
    write_candidate_diagnostics: bool = True,
) -> FixedInferenceOutputs:
    """Run label-free multi-view fixed inference from embedding shard manifests.

    ``embedding_manifests`` contains ``(view_name, manifest_path, score_calibration)``.
    Every view is scored with senior query-mean cosine and senior GlobalSearch;
    a final candidate is selected per query using only label-free score/prefix
    quality.  No target ground truth or metrics are read.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not embedding_manifests:
        raise RuntimeError("multi-view inference requires at least one embedding manifest")
    score_results = {}
    score_cache = {}
    resolved_profiles: dict[str, tuple[str, str]] = {}
    for view_name, manifest_path, calibration in embedding_manifests:
        # Only affine score calibrations are applied during chunked scoring.
        # Search profiles such as local_center/temperature/precision/recall are
        # handled inside multiview_search so they can change the GlobalSearch
        # endpoint without pretending to be a different embedding.
        profile = str(calibration or "none")
        chunk_calibration = profile if profile in {"none", "center", "zscore"} else "none"
        cache_key = (str(manifest_path), str(chunk_calibration))
        if cache_key in score_cache:
            result = score_cache[cache_key]
        else:
            result = chunked_query_cosine_from_embedding_shards(
                manifest_path=manifest_path,
                query_node_sets=query_batch,
                chunk_size=chunk_size,
                runtime_jsonl=runtime_jsonl,
                score_calibration=chunk_calibration,
            )
            score_cache[cache_key] = result
        key = str(view_name)
        if key in score_results:
            raise RuntimeError(f"duplicate multi-view name: {key}")
        score_results[key] = result
        if view_profiles is not None and key in view_profiles:
            resolved_profiles[key] = view_profiles[key]
        else:
            base_mode = key.split("__", 1)[0]
            resolved_profiles[key] = (base_mode, profile)
    candidate_diag = out_dir / f"{graph_name}.mv_candidates.jsonl" if write_candidate_diagnostics else None
    candidate_summary = out_dir / f"{graph_name}.mv_summary.json" if write_candidate_diagnostics else None
    records = build_prediction_records_from_multiview_scores(
        graph_name=graph_name,
        score_results=score_results,
        global_node_ids=global_node_ids,
        config=selection_config,
        view_profiles=resolved_profiles,
        edge_index=edge_index,
        candidate_diagnostics_path=candidate_diag,
        candidate_summary_path=candidate_summary,
    )
    jsonl_path = write_prediction_jsonl(records, out_dir / f"{graph_name}.pred.jsonl")
    pred_path = write_plain_pred(records, out_dir / f"{graph_name}.pred", use_global_ids=global_node_ids is not None)
    check_path = write_protocol_check(records, out_dir / f"{graph_name}.protocol_check.json")
    status_path = write_fixed_inference_status(
        FixedCheckpointStatus(
            checkpoint_path=str(checkpoint_path),
            split="unseen_inference",
            graph_name=graph_name,
            labels_used=False,
            metrics_computed=False,
            checkpoint_selection_from_unseen=False,
            unseen_metrics_used=False,
            ground_truth_loaded=False,
            notes=("fixed checkpoint inference", "multi-view senior GlobalSearch", "embedding shards", "no metrics"),
        ),
        out_dir / f"{graph_name}.fixed_inference_status.json",
    )
    return FixedInferenceOutputs(
        records=tuple(records),
        jsonl_path=jsonl_path,
        pred_path=pred_path,
        protocol_check_path=check_path,
        status_path=status_path,
    )


def run_fixed_inference_from_embedding_shards(
    *,
    graph_name: str,
    embedding_manifest: str | Path,
    query_batch: SeniorQueryBatch,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    global_node_ids: Sequence[int] | None = None,
    chunk_size: int = 4096,
    runtime_jsonl: str | Path | None = None,
    score_calibration: str = "none",
) -> FixedInferenceOutputs:
    """Run fixed prediction from disk-backed embedding shards without metrics."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_result = chunked_query_cosine_from_embedding_shards(
        manifest_path=embedding_manifest,
        query_node_sets=query_batch,
        chunk_size=chunk_size,
        runtime_jsonl=runtime_jsonl,
        score_calibration=score_calibration,
    )
    records = build_prediction_records_from_chunked_scores(
        graph_name=graph_name,
        score_result=score_result,
        global_node_ids=global_node_ids,
        metadata={"score_source": "v5_senior_mean_cosine_embedding_shards"},
    )
    jsonl_path = write_prediction_jsonl(records, out_dir / f"{graph_name}.pred.jsonl")
    pred_path = write_plain_pred(records, out_dir / f"{graph_name}.pred", use_global_ids=global_node_ids is not None)
    check_path = write_protocol_check(records, out_dir / f"{graph_name}.protocol_check.json")
    status_path = write_fixed_inference_status(
        FixedCheckpointStatus(
            checkpoint_path=str(checkpoint_path),
            split="unseen_inference",
            graph_name=graph_name,
            labels_used=False,
            metrics_computed=False,
            checkpoint_selection_from_unseen=False,
            unseen_metrics_used=False,
            ground_truth_loaded=False,
            notes=("fixed checkpoint inference", "senior GlobalSearch", "embedding shards", "no metrics"),
        ),
        out_dir / f"{graph_name}.fixed_inference_status.json",
    )
    return FixedInferenceOutputs(
        records=tuple(records),
        jsonl_path=jsonl_path,
        pred_path=pred_path,
        protocol_check_path=check_path,
        status_path=status_path,
    )
