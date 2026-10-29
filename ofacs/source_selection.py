"""Source-only checkpoint selection from diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch

from .errors import V5DataLeakageError, V5ProtocolError
from .fixed_checkpoint import save_source_checkpoint


@dataclass(frozen=True)
class SourceDiagnosticSummary:
    diagnostic_path: str
    record_count: int
    metric_name: str
    metric_mean: float
    selected_recall_mean: float | None
    selected_k_mean: float | None
    source_only: bool = True
    labels_used: bool = True
    ground_truth_used: bool = False
    metrics_used_for_training: bool = False
    unseen_metrics_used: bool = False

    def __post_init__(self) -> None:
        if not self.source_only or self.ground_truth_used or self.metrics_used_for_training or self.unseen_metrics_used:
            raise V5DataLeakageError("unsafe source diagnostic summary")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _diagnostic_paths(path: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(path, (str, Path)):
        raw = [x.strip() for x in str(path).split(",") if x.strip()]
        return [Path(x) for x in raw]
    return [Path(x) for x in path]


def summarize_source_diagnostics(path: str | Path | Iterable[str | Path], *, metric_name: str = "selected_f1_source_diagnostic") -> SourceDiagnosticSummary:
    paths = _diagnostic_paths(path)
    if not paths:
        raise V5ProtocolError("at least one source diagnostic file is required")
    values: list[float] = []
    recalls: list[float] = []
    selected_ks: list[float] = []
    for p in paths:
        if not p.exists():
            raise V5ProtocolError(f"diagnostic file not found: {p}")
        lowered_name = p.name.lower()
        # Source-to-source validation files may include the token validation, but
        # must never contain unseen/test/metric feedback paths.
        if any(tok in lowered_name for tok in ("unseen", "test", "metric")):
            raise V5DataLeakageError(f"diagnostic filename looks unsafe for source selection: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("source_only") is False or row.get("ground_truth_used") is True or row.get("unseen_metrics_used") is True:
                    raise V5DataLeakageError(f"unsafe diagnostic flags at {p}:{line_no}")
                if "ranking" in row:
                    ranking = row["ranking"]
                    if ranking.get("source_only") is False or ranking.get("ground_truth_used") is True:
                        raise V5DataLeakageError(f"unsafe ranking flags at {p}:{line_no}")
                    metric_value = ranking.get("selected_f1", ranking.get("oracle_best_f1"))
                    recall_value = ranking.get("recall_at_gt_size")
                    selected_k = ranking.get("selected_k")
                else:
                    metric_value = row.get(metric_name, row.get("selected_f1_source_diagnostic"))
                    recall_value = row.get("selected_recall")
                    selected_k = row.get("selected_k")
                if metric_value is None:
                    continue
                values.append(float(metric_value))
                if recall_value is not None:
                    recalls.append(float(recall_value))
                if selected_k is not None:
                    selected_ks.append(float(selected_k))
    if not values:
        raise V5ProtocolError(f"no source diagnostic values found in {paths}")
    return SourceDiagnosticSummary(
        diagnostic_path=",".join(str(p) for p in paths),
        record_count=len(values),
        metric_name=metric_name,
        metric_mean=float(mean(values)),
        selected_recall_mean=None if not recalls else float(mean(recalls)),
        selected_k_mean=None if not selected_ks else float(mean(selected_ks)),
    )


def write_selected_checkpoint(
    *,
    checkpoint_path: str | Path,
    diagnostic_path: str | Path,
    output_path: str | Path,
    metric_name: str = "selected_f1_source_diagnostic",
) -> tuple[Path, SourceDiagnosticSummary]:
    summary = summarize_source_diagnostics(diagnostic_path, metric_name=metric_name)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if bool(ckpt.get("checkpoint_selection_from_unseen", False)) or bool(ckpt.get("unseen_metrics_used", False)):
        raise V5DataLeakageError("input checkpoint metadata indicates unseen-based selection")
    if "model_state_dict" not in ckpt:
        raise V5ProtocolError("checkpoint missing model_state_dict")
    out = save_source_checkpoint(
        output_path,
        model_state_dict=ckpt["model_state_dict"],
        epoch=int(ckpt.get("epoch", 0)),
        source_selection_metric=f"source_diagnostic:{summary.metric_name}",
        source_selection_value=summary.metric_mean,
        config={
            **dict(ckpt.get("config", {})),
            "source_diagnostic_summary": summary.to_dict(),
            "input_checkpoint_path": str(checkpoint_path),
        },
        extra_state={key: ckpt[key] for key in ("domain_guidance", "input_dim", "unified_input_dim", "feature_projection", "global_token_shape") if key in ckpt},
    )
    return out, summary
