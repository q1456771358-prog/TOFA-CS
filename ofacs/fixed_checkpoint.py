"""Fixed checkpoint helpers for V5.

Training can write source-selected checkpoints.  Fixed inference can only load a
specific checkpoint and write no-metric status; it must not select checkpoints
from unseen labels or metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import torch

from .errors import V5DataLeakageError, V5ProtocolError


@dataclass(frozen=True)
class FixedCheckpointStatus:
    checkpoint_path: str
    split: str
    graph_name: str | None = None
    labels_used: bool = False
    metrics_computed: bool = False
    checkpoint_selection_from_unseen: bool = False
    unseen_metrics_used: bool = False
    ground_truth_loaded: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split == "unseen_inference":
            if self.labels_used or self.metrics_computed or self.checkpoint_selection_from_unseen or self.unseen_metrics_used or self.ground_truth_loaded:
                raise V5DataLeakageError("unsafe fixed unseen inference status")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_source_checkpoint(
    path: str | Path,
    *,
    model_state_dict: Mapping[str, torch.Tensor],
    epoch: int,
    source_selection_metric: str | None = None,
    source_selection_value: float | None = None,
    config: Mapping[str, object] | None = None,
    extra_state: Mapping[str, object] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if source_selection_metric is not None and any(x in source_selection_metric.lower() for x in ("unseen", "test")):
        raise V5DataLeakageError("checkpoint source_selection_metric appears unsafe")
    cfg = dict(config or {})
    payload: dict[str, object] = {
        "epoch": int(epoch),
        "model_state_dict": dict(model_state_dict),
        "source_selection_metric": source_selection_metric,
        "source_selection_value": source_selection_value,
        "config": cfg,
        "input_dim": int(cfg["input_dim"]) if "input_dim" in cfg and cfg["input_dim"] is not None else None,
        "unified_input_dim": int(cfg["unified_input_dim"]) if "unified_input_dim" in cfg and cfg["unified_input_dim"] is not None else None,
        "feature_projection": cfg.get("feature_projection"),
        "checkpoint_selection_from_unseen": False,
        "unseen_metrics_used": False,
    }
    if extra_state:
        payload.update(dict(extra_state))
    torch.save(payload, out)
    return out


def load_fixed_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, object]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise V5ProtocolError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if bool(ckpt.get("checkpoint_selection_from_unseen", False)) or bool(ckpt.get("unseen_metrics_used", False)):
        raise V5DataLeakageError("checkpoint metadata indicates unseen-based selection")
    if "model_state_dict" not in ckpt:
        raise V5ProtocolError("checkpoint missing model_state_dict")
    return ckpt


def write_fixed_inference_status(status: FixedCheckpointStatus, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out
