"""Integrated entry point for shift-adaptive modularity-driven retrieval.

This facade exposes the validated dual-view scoring, unlabeled population-state
calibration, query-conditioned core update, and protected endpoint search as
one community-search stage.  The underlying functions remain unchanged to keep
fixed-checkpoint predictions numerically identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .chunked_scoring import ChunkedScoreResult
from .community_output import CommunityPredictionRecord
from .dual_branch_scoring import (
    DualBranchGateConfig,
    build_dual_branch_prediction_records,
    chunked_dual_branch_query_scores,
    write_dual_branch_diagnostics,
)
from .senior_compat import SeniorQueryBatch
from .target_calibrated_representation import TargetCalibratedRepresentation


@dataclass(frozen=True)
class ShiftAdaptiveSearchResult:
    """Complete output of the integrated community-search stage."""

    scores: ChunkedScoreResult
    report: Mapping[str, object]
    records: tuple[CommunityPredictionRecord, ...]


class ShiftAdaptiveModularitySearch:
    """High-level searcher with a numerically unchanged legacy core."""

    def __init__(self, config: DualBranchGateConfig | None = None) -> None:
        self.config = (config or DualBranchGateConfig(enabled=True)).validate()

    def retrieve(
        self,
        *,
        graph_name: str,
        representations: TargetCalibratedRepresentation,
        query_batch: SeniorQueryBatch,
        global_node_ids: Sequence[int] | None = None,
        chunk_size: int = 4096,
        runtime_jsonl: str | Path | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> ShiftAdaptiveSearchResult:
        """Score and retrieve communities without reading target ground truth."""

        score_result, report = chunked_dual_branch_query_scores(
            **representations.as_legacy_manifest_kwargs(),
            query_node_sets=query_batch,
            chunk_size=chunk_size,
            runtime_jsonl=runtime_jsonl,
            config=self.config,
        )
        records = build_dual_branch_prediction_records(
            graph_name=graph_name,
            score_result=score_result,
            dual_report=report,
            global_node_ids=global_node_ids,
            config=self.config,
            edge_index=edge_index,
        )
        return ShiftAdaptiveSearchResult(
            scores=score_result,
            report=report,
            records=tuple(records),
        )

    @staticmethod
    def write_diagnostics(path: str | Path, result: ShiftAdaptiveSearchResult) -> Path:
        """Write the existing label-free diagnostics without changing them."""

        return write_dual_branch_diagnostics(path, result.report)
