"""V5-safe monitored command runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

from .errors import V5DataLeakageError, V5ProtocolError

try:
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


_ALLOWED_TOOL_ENTRIES = {
    "tools/diagnose_diffusion_paths_ofacs.py",
    "tools/diagnose_ranking_boundary.py",
    "tools/evaluate_fixed_checkpoint_ofacs.py",
    "tools/prepare_real_data_run_plan_ofacs.py",
    "tools/resource_preflight_ofacs.py",
    "tools/run_v5_batch_plan.py",
    "tools/select_best_source_checkpoint_ofacs.py",
    "tools/select_source_checkpoint_ofacs.py",
    "tools/train_ofacs.py",
}

_FORBIDDEN_COMMAND_TOKENS = {
    "--gt",
    "--ground-truth",
    "--ground_truth",
    "--answer",
    "--answers",
    "--metric",
    "--metrics",
    "--f1",
    "--jaccard",
    "--nmi",
    "--ari",
    "--label-path",
    "--label_path",
    "--unseen-label",
    "--unseen_label",
    "--unseen-metric",
    "--unseen_metric",
}


@dataclass(frozen=True)
class MonitoredCommandReport:
    command: tuple[str, ...]
    cwd: str
    output_dir: str
    stdout_path: str
    resource_monitor_jsonl_path: str
    status_json_path: str
    return_code: int
    completed: bool
    timed_out: bool
    killed_by_sigkill: bool
    likely_external_kill: bool
    max_process_rss_bytes: int
    max_gpu_memory_used_mib: int | None
    max_gpu_utilization_percent: int | None
    last_resource_record: Mapping[str, object] | None
    metrics_computed: bool = False
    labels_used_for_unseen: bool = False
    checkpoint_selection_from_unseen: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_monitored_command(
    command: Sequence[str],
    *,
    output_dir: str | Path,
    cwd: str | Path,
    monitor_interval_seconds: float = 5.0,
    timeout_seconds: float | None = None,
    env: Mapping[str, str] | None = None,
) -> MonitoredCommandReport:
    normalized = tuple(str(x) for x in command if str(x))
    if not normalized:
        raise V5ProtocolError("monitored command must be non-empty")
    if monitor_interval_seconds <= 0:
        raise V5ProtocolError("monitor_interval_seconds must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise V5ProtocolError("timeout_seconds must be positive when provided")
    validate_monitored_command(normalized)

    resolved_cwd = Path(cwd).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stdout_path = out / "stdout.log"
    resource_path = out / "resource_monitor.jsonl"
    status_path = out / "status.json"
    _write_status(status_path, "running", {"command": list(normalized)})

    started = time.perf_counter()
    max_rss = 0
    max_gpu_memory = None
    max_gpu_utilization = None
    last_record: dict[str, object] | None = None
    timed_out = False
    return_code = 1
    with stdout_path.open("w", encoding="utf-8") as stdout, resource_path.open("w", encoding="utf-8") as monitor:
        child = subprocess.Popen(
            list(normalized),
            cwd=str(resolved_cwd),
            env=_child_env(resolved_cwd, env=env),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process = _psutil_process(child.pid)
        while child.poll() is None:
            elapsed = time.perf_counter() - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                timed_out = True
                _terminate_process_tree(child, process)
                break
            rss = _process_tree_rss(process)
            max_rss = max(max_rss, rss)
            last_record = _resource_snapshot(started=started, alive=True, return_code=None, process_tree_rss_bytes=rss)
            max_gpu_memory, max_gpu_utilization = _update_gpu_peaks(last_record, max_gpu_memory, max_gpu_utilization)
            monitor.write(json.dumps(last_record, ensure_ascii=False, sort_keys=True) + "\n")
            monitor.flush()
            time.sleep(float(monitor_interval_seconds))
        return_code = int(child.wait())
        rss = _process_tree_rss(process)
        max_rss = max(max_rss, rss)
        last_record = _resource_snapshot(started=started, alive=False, return_code=return_code, process_tree_rss_bytes=rss)
        max_gpu_memory, max_gpu_utilization = _update_gpu_peaks(last_record, max_gpu_memory, max_gpu_utilization)
        monitor.write(json.dumps(last_record, ensure_ascii=False, sort_keys=True) + "\n")

    report = MonitoredCommandReport(
        command=normalized,
        cwd=str(resolved_cwd),
        output_dir=str(out),
        stdout_path=str(stdout_path),
        resource_monitor_jsonl_path=str(resource_path),
        status_json_path=str(status_path),
        return_code=return_code,
        completed=return_code == 0 and not timed_out,
        timed_out=timed_out,
        killed_by_sigkill=return_code == -9,
        likely_external_kill=return_code == -9,
        max_process_rss_bytes=int(max_rss),
        max_gpu_memory_used_mib=max_gpu_memory,
        max_gpu_utilization_percent=max_gpu_utilization,
        last_resource_record=last_record,
    )
    _write_status(status_path, "completed" if report.completed else "failed", report.to_dict())
    return report


def validate_monitored_command(command: Sequence[str]) -> None:
    entry = _find_tool_entry(command)
    if entry not in _ALLOWED_TOOL_ENTRIES:
        raise V5ProtocolError(f"monitored command entry must be one of {sorted(_ALLOWED_TOOL_ENTRIES)}, got {entry!r}")
    for raw in command:
        token = str(raw).strip().lower()
        if token in _FORBIDDEN_COMMAND_TOKENS:
            raise V5DataLeakageError(f"monitored command contains forbidden feedback argument: {raw}")
        if token.endswith(".gt") or "/.gt" in token or "\\.gt" in token:
            raise V5DataLeakageError(f"monitored command appears to reference a .gt file: {raw}")


def _find_tool_entry(command: Sequence[str]) -> str:
    for part in command:
        text = str(part).replace("\\", "/")
        if text.endswith(".py"):
            path = Path(text)
            if len(path.parts) >= 2 and path.parts[-2] == "tools":
                return f"tools/{path.name}"
    return str(command[0]).replace("\\", "/")


def _resource_snapshot(*, started: float, alive: bool, return_code: int | None, process_tree_rss_bytes: int) -> dict[str, object]:
    memory = _virtual_memory()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "subprocess_alive": alive,
        "process_return_code": return_code,
        "process_tree_rss_bytes": int(process_tree_rss_bytes),
        "system_memory_total_bytes": memory["total"],
        "system_memory_used_bytes": memory["used"],
        "system_memory_available_bytes": memory["available"],
        "system_memory_percent": memory["percent"],
        "gpu_resources": _query_gpu_resources(),
    }


def _query_gpu_resources() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                    "utilization_percent": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _update_gpu_peaks(record: Mapping[str, object], max_memory: int | None, max_util: int | None) -> tuple[int | None, int | None]:
    resources = record.get("gpu_resources")
    if not isinstance(resources, list):
        return max_memory, max_util
    for gpu in resources:
        if not isinstance(gpu, Mapping):
            continue
        memory = gpu.get("memory_used_mib")
        util = gpu.get("utilization_percent")
        if isinstance(memory, int):
            max_memory = memory if max_memory is None else max(max_memory, memory)
        if isinstance(util, int):
            max_util = util if max_util is None else max(max_util, util)
    return max_memory, max_util


def _psutil_process(pid: int) -> object | None:
    if psutil is None:
        return None
    try:
        return psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[union-attr]
        return None


def _process_tree_rss(process: object | None) -> int:
    if psutil is None or process is None:
        return 0
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]  # type: ignore[attr-defined]
    except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[union-attr]
        return 0
    for current in processes:
        try:
            total += int(current.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[union-attr]
            pass
    return total


def _virtual_memory() -> dict[str, int | float]:
    if psutil is None:
        return {"total": 0, "used": 0, "available": 0, "percent": 0.0}
    mem = psutil.virtual_memory()
    return {"total": int(mem.total), "used": int(mem.used), "available": int(mem.available), "percent": float(mem.percent)}


def _terminate_process_tree(child: subprocess.Popen[str], process: object | None) -> None:
    if psutil is not None and process is not None:
        try:
            children = process.children(recursive=True)  # type: ignore[attr-defined]
            for current in children:
                current.terminate()
            process.terminate()  # type: ignore[attr-defined]
            _, alive = psutil.wait_procs([process, *children], timeout=5)  # type: ignore[arg-type,union-attr]
            for current in alive:
                current.kill()
            return
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[union-attr]
            pass
    child.kill()


def _child_env(cwd: Path, *, env: Mapping[str, str] | None) -> dict[str, str]:
    child_env = dict(os.environ if env is None else env)
    existing = child_env.get("PYTHONPATH", "")
    parts = [str(cwd)]
    if existing:
        parts.append(existing)
    child_env["PYTHONPATH"] = os.pathsep.join(parts)
    return child_env


def _write_status(path: Path, status: str, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "payload": dict(payload),
                "metrics_computed": False,
                "labels_used_for_unseen": False,
                "checkpoint_selection_from_unseen": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
