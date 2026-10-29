"""Reproducibility helpers for OFA-CS V5 scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(
    seed: int | None,
    *,
    deterministic: bool = False,
    deterministic_strict: bool = False,
) -> int | None:
    if seed is None:
        return None
    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if deterministic or deterministic_strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=not bool(deterministic_strict))
        except Exception:
            if deterministic_strict:
                raise
    return value


def make_generator(seed: int | None, device: torch.device | str = "cpu") -> torch.Generator | None:
    """Create the stateful generator used by the validated R5 sampling path."""
    if seed is None:
        return None
    dev = torch.device(device)
    gen_device = dev.type if dev.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(seed))
    return generator
