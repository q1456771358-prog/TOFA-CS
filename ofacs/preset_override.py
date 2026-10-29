"""Validated post-preset overrides for controlled V574 experiments.

The normal experiment path applies a compact preset first.  This module then
optionally applies a small JSON object *after* the preset so fixed-checkpoint
re-evaluation can vary a few declared parameters without duplicating presets or
silently being overwritten by :func:`apply_v55_preset`.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping


class PresetOverrideError(ValueError):
    """Raised when a preset override is malformed or targets an unknown field."""


def _load_json_object(spec: str | None) -> dict[str, Any]:
    if spec is None or not str(spec).strip():
        return {}
    raw = str(spec).strip()
    path: Path | None = None
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
    else:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            path = candidate
    if path is not None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PresetOverrideError(f"cannot read preset override file {path}: {exc}") from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PresetOverrideError(f"--preset-override-json must be a JSON object or @file: {exc}") from exc
    if not isinstance(obj, dict):
        raise PresetOverrideError("--preset-override-json must decode to an object")
    return dict(obj)


def _coerce_like(current: Any, value: Any, *, key: str) -> Any:
    """Validate JSON value against the argparse field's existing type."""
    if current is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise PresetOverrideError(f"override {key!r} must be boolean, got {type(value).__name__}")
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise PresetOverrideError(f"override {key!r} must be integer, got {type(value).__name__}")
        return int(value)
    if isinstance(current, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PresetOverrideError(f"override {key!r} must be numeric, got {type(value).__name__}")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise PresetOverrideError(f"override {key!r} must be string, got {type(value).__name__}")
        return value
    return value


def apply_preset_overrides(
    args: Namespace,
    spec: str | None,
    *,
    forbidden_keys: set[str] | None = None,
) -> tuple[Namespace, dict[str, Any]]:
    """Apply validated overrides to an argparse namespace in place.

    Unknown fields fail loudly.  This prevents a sweep from appearing to run
    while a misspelled parameter is silently ignored.  Identity/protocol fields
    can be blocked by ``forbidden_keys`` at the caller.
    """
    overrides = _load_json_object(spec)
    blocked = set(forbidden_keys or ())
    applied: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in blocked:
            raise PresetOverrideError(f"override of protocol field {key!r} is forbidden")
        if not hasattr(args, key):
            raise PresetOverrideError(f"unknown preset override field {key!r}")
        coerced = _coerce_like(getattr(args, key), value, key=key)
        setattr(args, key, coerced)
        applied[key] = coerced
    setattr(args, "preset_overrides_applied", applied)
    return args, applied


def namespace_jsonable(args: Namespace) -> dict[str, Any]:
    """Return a JSON-safe, sorted view of resolved argparse fields."""
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, Mapping):
            out[key] = dict(value)
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = repr(value)
    return dict(sorted(out.items()))
