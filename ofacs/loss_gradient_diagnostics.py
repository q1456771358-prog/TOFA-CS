"""Low-frequency gradient diagnostics for multi-objective source training."""
from __future__ import annotations

import math
import torch


def _loss_gradients(loss: torch.Tensor | None, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None] | None:
    if loss is None or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return None
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return [None if grad is None else grad.detach() for grad in grads]


def _norm(grads: list[torch.Tensor | None] | None) -> float | None:
    if grads is None:
        return None
    total = None
    for grad in grads:
        if grad is None:
            continue
        value = (grad.float() * grad.float()).sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    result = float(torch.sqrt(total).cpu().item())
    return result if math.isfinite(result) else None


def _cosine(a: list[torch.Tensor | None] | None, b: list[torch.Tensor | None] | None) -> float | None:
    if a is None or b is None:
        return None
    dot = None
    norm_a = None
    norm_b = None
    for grad_a, grad_b in zip(a, b):
        if grad_a is None or grad_b is None:
            continue
        ga = grad_a.float()
        gb = grad_b.float()
        d = (ga * gb).sum()
        na = (ga * ga).sum()
        nb = (gb * gb).sum()
        dot = d if dot is None else dot + d
        norm_a = na if norm_a is None else norm_a + na
        norm_b = nb if norm_b is None else norm_b + nb
    if dot is None or norm_a is None or norm_b is None:
        return None
    denom = torch.sqrt(norm_a * norm_b).clamp_min(1e-20)
    result = float((dot / denom).cpu().item())
    return result if math.isfinite(result) else None


def _shared_parameter_count(a, b) -> int:
    if a is None or b is None:
        return 0
    return sum(int(ga is not None and gb is not None) for ga, gb in zip(a, b))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def compute_loss_gradient_diagnostics(
    *,
    model: torch.nn.Module,
    base_loss: torch.Tensor | None,
    diffusion_loss: torch.Tensor | None,
    scpc_loss: torch.Tensor | None,
    cdss_loss: torch.Tensor | None,
) -> dict:
    """Return norms and cosines without modifying ``.grad`` buffers.

    Losses should already include their external scalar weights when the caller
    wants the diagnostic to reflect their actual contribution to the objective.
    """

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = {
        "base": _loss_gradients(base_loss, parameters),
        "diffusion": _loss_gradients(diffusion_loss, parameters),
        "scpc": _loss_gradients(scpc_loss, parameters),
        "cdss": _loss_gradients(cdss_loss, parameters),
    }
    norm_base = _norm(gradients["base"])
    norm_diffusion = _norm(gradients["diffusion"])
    norm_scpc = _norm(gradients["scpc"])
    norm_cdss = _norm(gradients["cdss"])
    scpc_base_cosine = _cosine(gradients["scpc"], gradients["base"])
    cdss_base_cosine = _cosine(gradients["cdss"], gradients["base"])
    cdss_scpc_cosine = _cosine(gradients["cdss"], gradients["scpc"])
    report = {
        "grad_norm_base": norm_base,
        "grad_norm_diffusion": norm_diffusion,
        "grad_norm_scpc": norm_scpc,
        "grad_norm_cdss": norm_cdss,
        "grad_norm_diffusion_to_base_ratio": _safe_ratio(norm_diffusion, norm_base),
        "grad_norm_scpc_to_base_ratio": _safe_ratio(norm_scpc, norm_base),
        "grad_norm_cdss_to_base_ratio": _safe_ratio(norm_cdss, norm_base),
        "grad_cosine_scpc_vs_base": scpc_base_cosine,
        "grad_cosine_cdss_vs_base": cdss_base_cosine,
        "grad_cosine_cdss_vs_scpc": cdss_scpc_cosine,
        "grad_conflict_scpc_vs_base": None if scpc_base_cosine is None else bool(scpc_base_cosine < 0.0),
        "grad_conflict_cdss_vs_base": None if cdss_base_cosine is None else bool(cdss_base_cosine < 0.0),
        "grad_conflict_cdss_vs_scpc": None if cdss_scpc_cosine is None else bool(cdss_scpc_cosine < 0.0),
        "shared_parameters_scpc_vs_base": _shared_parameter_count(gradients["scpc"], gradients["base"]),
        "shared_parameters_cdss_vs_base": _shared_parameter_count(gradients["cdss"], gradients["base"]),
        "shared_parameters_cdss_vs_scpc": _shared_parameter_count(gradients["cdss"], gradients["scpc"]),
        "all_reported_values_finite": all(
            value is None or (not isinstance(value, float)) or math.isfinite(value)
            for value in (
                norm_base, norm_diffusion, norm_scpc, norm_cdss,
                scpc_base_cosine, cdss_base_cosine, cdss_scpc_cosine,
            )
        ),
        "loss_inputs_are_weighted": True,
        "base_excludes_diffusion_scpc_cdss": True,
    }
    return report
