"""Small V5 training-step helpers.

This is not a monolithic trainer.  It wires the source-only query-boundary plan,
query-boundary loss, and guided diffusion loss so later scripts can keep module
boundaries auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .errors import V5DataLeakageError, V5ProtocolError
from .query_boundary_loss import QueryBoundaryLossConfig, QueryBoundaryLossResult, compute_query_boundary_loss
from .source_query_boundary_plan import SourceQueryBoundarySamplerConfig, SourceQueryBoundaryPlan, build_source_query_boundary_plan


@dataclass(frozen=True)
class V5Stage1LossConfig:
    query_boundary: QueryBoundaryLossConfig = QueryBoundaryLossConfig()
    sampler: SourceQueryBoundarySamplerConfig = SourceQueryBoundarySamplerConfig()
    diffusion_weight: float = 0.25
    legacy_aux_weight: float = 0.05
    source_only: bool = True

    def validate(self) -> "V5Stage1LossConfig":
        if not self.source_only:
            raise V5DataLeakageError("V5Stage1LossConfig must be source_only=True")
        if float(self.diffusion_weight) < 0 or float(self.legacy_aux_weight) < 0:
            raise V5ProtocolError("weights must be non-negative")
        self.query_boundary.validate()
        self.sampler.validate()
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5Stage1LossResult:
    total_loss: torch.Tensor
    query_boundary_result: QueryBoundaryLossResult
    source_query_plan: SourceQueryBoundaryPlan
    diffusion_loss: torch.Tensor | None
    legacy_aux_loss: torch.Tensor | None
    report: dict[str, object]


def compute_v5_stage1_source_loss(
    *,
    refined_embeddings: torch.Tensor,
    labels: torch.Tensor | Sequence[int],
    graph_name: str,
    query_node_sets: Sequence[Sequence[int] | torch.Tensor] | None = None,
    edge_index: torch.Tensor | Sequence[Sequence[int]] | None = None,
    node_features: torch.Tensor | None = None,
    current_scores: torch.Tensor | None = None,
    diffusion_loss: torch.Tensor | None = None,
    legacy_aux_loss: torch.Tensor | None = None,
    config: V5Stage1LossConfig | None = None,
) -> V5Stage1LossResult:
    """Build source samples and compute V5 Stage-1 loss.

    Labels are permitted here only because this function is source-only.  It must
    never be called on an unseen split.
    """

    cfg = (config or V5Stage1LossConfig()).validate()
    plan = build_source_query_boundary_plan(
        labels=labels,
        graph_name=graph_name,
        query_node_sets=query_node_sets,
        edge_index=edge_index,
        node_features=node_features,
        current_scores=current_scores,
        config=cfg.sampler,
        split="source_train",
    )
    qb = compute_query_boundary_loss(
        refined_embeddings,
        plan.samples,
        config=cfg.query_boundary,
        old_aux_loss=legacy_aux_loss,
    )
    total = qb.total_loss
    if diffusion_loss is not None and float(cfg.diffusion_weight) > 0:
        total = total + float(cfg.diffusion_weight) * diffusion_loss
    # old auxiliary is already included inside query-boundary config if supplied.
    report = {
        "graph_name": graph_name,
        "total_loss": float(total.detach().cpu().item()),
        "query_boundary": qb.report.to_dict(),
        "source_query_plan": plan.report,
        "diffusion_loss": None if diffusion_loss is None else float(diffusion_loss.detach().cpu().item()),
        "legacy_aux_loss": None if legacy_aux_loss is None else float(legacy_aux_loss.detach().cpu().item()),
        "source_only": True,
        "unseen_training_used": False,
        "full_cartesian_used": False,
    }
    return V5Stage1LossResult(
        total_loss=total,
        query_boundary_result=qb,
        source_query_plan=plan,
        diffusion_loss=diffusion_loss,
        legacy_aux_loss=legacy_aux_loss,
        report=report,
    )
