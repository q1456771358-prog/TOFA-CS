#!/usr/bin/env python3
"""Lightweight integrity check for the public OFA-CS V6 repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ofacs.v575_kernel_lock import verify_kernel_lock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data() -> int:
    checksum_file = ROOT / "data" / "SHA256SUMS.txt"
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        # pathlib accepts forward slashes on every supported platform; rebuild
        # from parts so the same checksum file works on Windows and Linux.
        path = ROOT / "data" / Path(*relative.strip().split("/"))
        if not path.is_file():
            raise RuntimeError(f"missing bundled data file: {relative}")
        actual = sha256(path)
        if actual != expected.lower():
            raise RuntimeError(f"data hash mismatch: {relative}: {actual} != {expected}")
        checked += 1
    return checked


def verify_json() -> int:
    paths = list((ROOT / "configs").glob("*.json")) + [ROOT / "RELEASE_PROVENANCE.json"]
    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))
    return len(paths)


def model_status() -> dict[str, object]:
    verified: list[str] = []
    incomplete: list[str] = []
    invalid: list[str] = []
    for directory in sorted((ROOT / "pretrained").glob("seed_*")):
        required = [
            directory / "trained_model.pt",
            directory / "frozen_reference.pt",
            directory / "model_manifest.json",
        ]
        if not all(path.is_file() for path in required):
            incomplete.append(directory.name)
            continue
        try:
            manifest = json.loads(required[2].read_text(encoding="utf-8"))
            trained = manifest["trained_model"]
            reference = manifest["frozen_reference"]
            checks = [
                (directory / str(trained["file"]), str(trained["sha256"])),
                (directory / str(reference["file"]), str(reference["sha256"])),
                (
                    (directory / str(manifest["training_config"])).resolve(),
                    str(manifest["training_config_sha256"]),
                ),
            ]
            if not all(path.is_file() and sha256(path) == expected for path, expected in checks):
                invalid.append(directory.name)
                continue
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid.append(directory.name)
            continue
        verified.append(directory.name)
    return {"verified": verified, "incomplete": incomplete, "invalid": invalid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()
    kernel = verify_kernel_lock(ROOT)
    data_count = verify_data()
    json_count = verify_json()
    models = model_status()
    if args.require_model and not models["verified"]:
        raise SystemExit("no verified pretrained model bundle")
    print(
        json.dumps(
            {
                "release_identity": "TOFA-CS",
                "source_integrity": {
                    "passed": bool(kernel["passed"]),
                    "file_count": int(kernel["file_count"]),
                },
                "bundled_data_files_verified": data_count,
                "json_files_verified": json_count,
                "pretrained_models": models,
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
