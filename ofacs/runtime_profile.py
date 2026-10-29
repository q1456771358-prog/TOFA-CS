"""Small JSONL runtime/memory profiling helpers for V5 tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Mapping

import torch


def runtime_snapshot(event: str, **extra: object) -> dict[str, object]:
    """Return a lightweight runtime snapshot without requiring psutil."""

    cpu_rss_mb = None
    try:
        import psutil  # type: ignore

        cpu_rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        cpu_rss_mb = None
    snap: dict[str, object] = {
        "event": event,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # Wall-clock seconds are too coarse for the short search/state stages
        # used in the paper efficiency table.  monotonic_ns is process-local
        # and is therefore safe for duration pairing inside one JSONL trace.
        "monotonic_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "cpu_rss_mb": None if cpu_rss_mb is None else round(float(cpu_rss_mb), 3),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        snap.update(
            {
                "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 3),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 3),
                "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
            }
        )
    snap.update(extra)
    return snap


def append_jsonl(path: str | Path | None, record: Mapping[str, object]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def profile_event(path: str | Path | None, event: str, **extra: object) -> None:
    append_jsonl(path, runtime_snapshot(event, **extra))
