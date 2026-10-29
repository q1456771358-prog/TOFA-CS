"""Runtime verification for the copied V575 immutable numerical kernel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_kernel_lock(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    manifest_path = root / "configs" / "v575_kernel_lock.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for rel, expected in sorted(dict(payload["files"]).items()):
        path = root / rel
        if not path.is_file():
            errors.append(f"missing:{rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"sha256:{rel}:{actual}:{expected}")
    if errors:
        raise RuntimeError("immutable V575 kernel lock failed: " + "; ".join(errors))
    return {
        "passed": True,
        "file_count": len(payload["files"]),
        "manifest": str(manifest_path),
        "method_identity": payload.get("method_identity"),
        "checkpoint_source": payload.get("checkpoint_source"),
    }
