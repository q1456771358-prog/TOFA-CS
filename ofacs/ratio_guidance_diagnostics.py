"""Streaming diagnostics for density-ratio reverse guidance.

The collector is observational only: tensors are detached before recording and
the reverse trajectory is never modified.  A bounded deterministic sample is
kept for percentile estimates so Reddit does not create a large diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import torch


@dataclass
class _Series:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)
    sample_seen: int = 0
    sample_limit: int = 8192

    def add(self, values: torch.Tensor) -> None:
        x = values.detach().float().reshape(-1).cpu()
        if x.numel() == 0:
            return
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            return
        self.count += int(finite.numel())
        self.total += float(finite.sum().item())
        self.total_squared += float((finite * finite).sum().item())
        self.maximum = max(self.maximum, float(finite.max().item()))
        take = min(int(finite.numel()), 1024)
        if take < int(finite.numel()):
            indices = torch.linspace(0, int(finite.numel()) - 1, steps=take).long()
            finite = finite.index_select(0, indices)
        for value in finite.tolist():
            self.sample_seen += 1
            if len(self.samples) < self.sample_limit:
                self.samples.append(float(value))
                continue
            # Deterministic reservoir replacement avoids first-chunk bias.
            slot = ((self.sample_seen * 1103515245 + 12345) & 0x7FFFFFFF) % self.sample_seen
            if slot < self.sample_limit:
                self.samples[slot] = float(value)

    def report(self) -> dict[str, float | int]:
        ordered = sorted(self.samples)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))] if ordered else 0.0
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else 0.0,
            "std": math.sqrt(max(0.0, self.total_squared / self.count - (self.total / self.count) ** 2)) if self.count else 0.0,
            "p95": float(p95),
            "max": self.maximum,
            "sample_count": len(ordered),
        }


class RatioGuidanceDiagnostics:
    """Accumulate node/neighbor guidance statistics across chunks and steps."""

    def __init__(self) -> None:
        self.log_ratio = _Series()
        self.abs_log_ratio = _Series()
        self.gradient_norm = _Series()
        self.update_norm = _Series()
        self.eps_prediction_norm = _Series()
        self.relative_update_norm = _Series()
        self.step_update_norm: dict[str, _Series] = {}
        self.scale_modes: dict[str, int] = {}
        self.clip_count = 0
        self.gradient_count = 0

    def record(
        self,
        *,
        branch: str,
        step: int,
        log_ratio: torch.Tensor,
        raw_gradient_norm: torch.Tensor,
        clipped_gradient_norm: torch.Tensor,
        guidance_update_norm: torch.Tensor,
        clip: float,
        eps_prediction_norm: torch.Tensor | None = None,
        scale_mode: str = "legacy_raw",
    ) -> None:
        lr = log_ratio.detach().float().reshape(-1)
        self.log_ratio.add(lr)
        self.abs_log_ratio.add(lr.abs())
        raw = raw_gradient_norm.detach().float().reshape(-1)
        self.gradient_norm.add(raw)
        self.update_norm.add(guidance_update_norm.detach().float().reshape(-1))
        if eps_prediction_norm is not None:
            eps = eps_prediction_norm.detach().float().reshape(-1)
            update = guidance_update_norm.detach().float().reshape(-1)
            self.eps_prediction_norm.add(eps)
            self.relative_update_norm.add(update / torch.clamp(eps, min=1e-12))
        self.scale_modes[str(scale_mode)] = self.scale_modes.get(str(scale_mode), 0) + int(raw.numel())
        key = f"{branch}:step_{int(step):04d}"
        self.step_update_norm.setdefault(key, _Series(sample_limit=256)).add(guidance_update_norm)
        self.gradient_count += int(raw.numel())
        if float(clip) > 0:
            self.clip_count += int((raw > float(clip)).sum().item())

    def to_dict(self) -> dict[str, Any]:
        log = self.log_ratio.report()
        abs_log = self.abs_log_ratio.report()
        grad = self.gradient_norm.report()
        update = self.update_norm.report()
        eps = self.eps_prediction_norm.report()
        relative = self.relative_update_norm.report()
        return {
            "protocol": "density-ratio-guidance-diagnostics-v3-effective-update",
            "labels_used": False,
            "ground_truth_used": False,
            "log_ratio_mean": log["mean"],
            "log_ratio_std": log["std"],
            "log_ratio_abs_mean": abs_log["mean"],
            "ratio_gradient_norm_mean": grad["mean"],
            "ratio_gradient_norm_p95": grad["p95"],
            "ratio_gradient_norm_max": grad["max"],
            "guidance_clip_active_rate": self.clip_count / self.gradient_count if self.gradient_count else 0.0,
            "guidance_update_norm_mean": update["mean"],
            "guidance_update_norm_p95": update["p95"],
            "guidance_update_norm_max": update["max"],
            "eps_prediction_norm_mean": eps["mean"],
            "guidance_relative_update_mean": relative["mean"],
            "guidance_relative_update_p95": relative["p95"],
            "guidance_scale_mode_counts": dict(sorted(self.scale_modes.items())),
            "each_step_guidance_update_norm": {
                key: value.report()["mean"] for key, value in sorted(self.step_update_norm.items())
            },
            "sample_count": int(log["sample_count"]),
            "sample_policy": "bounded_deterministic_reservoir",
        }


class RatioGuidanceRuntimeTimer:
    """Low-overhead timer used independently from diagnostic collection."""

    def __init__(self, *, synchronize_cuda: bool) -> None:
        self.synchronize_cuda = bool(synchronize_cuda)
        self.total_ns = 0
        self.call_count = 0
        self._start_ns: int | None = None
        self._cuda_start: torch.cuda.Event | None = None
        self._cuda_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def start(self, device: torch.device) -> None:
        if self._start_ns is not None or self._cuda_start is not None:
            raise RuntimeError("ratio guidance runtime timer is already active")
        if self.synchronize_cuda and device.type == "cuda" and torch.cuda.is_available():
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        else:
            self._start_ns = time.perf_counter_ns()

    def stop(self, device: torch.device) -> None:
        if self._cuda_start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._cuda_pairs.append((self._cuda_start, end))
            self._cuda_start = None
        elif self._start_ns is not None:
            self.total_ns += time.perf_counter_ns() - self._start_ns
            self._start_ns = None
        else:
            raise RuntimeError("ratio guidance runtime timer was not started")
        self.call_count += 1

    def to_dict(self) -> dict[str, float | int]:
        if self._start_ns is not None or self._cuda_start is not None:
            raise RuntimeError("ratio guidance runtime timer is still active")
        cuda_seconds = 0.0
        if self._cuda_pairs:
            torch.cuda.synchronize()
            cuda_seconds = sum(start.elapsed_time(end) for start, end in self._cuda_pairs) / 1000.0
        return {"seconds": self.total_ns / 1e9 + cuda_seconds, "call_count": self.call_count}
