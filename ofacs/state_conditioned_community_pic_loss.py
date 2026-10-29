"""State-conditioned Community Prototype-Informed Clustering loss (SCPC).

Source-supervised only. Target labels/ground truth are never read here.

This is the validated V574/R5 loss path. Diagnostics are detached from the
optimized objective so enabling V575/CDSS cannot silently change the B0 loss.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return list(x.values())
    return list(x)


def _soft_core(emb, pos_idx, scores=None, temperature=0.1, topk=64):
    if len(pos_idx) == 0:
        return emb.mean(0), 0
    idx = torch.as_tensor(list(pos_idx), device=emb.device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < emb.shape[0])]
    if idx.numel() == 0:
        return emb.mean(0), 0
    if scores is not None:
        order = torch.argsort(scores[idx], descending=True)
        idx = idx[order[: min(int(topk), idx.numel())]]
    else:
        idx = idx[: min(int(topk), idx.numel())]
    x = emb[idx]
    if scores is None:
        w = torch.ones(x.size(0), device=emb.device, dtype=emb.dtype) / max(1, x.size(0))
    else:
        w = torch.softmax(scores[idx] / max(float(temperature), 1e-6), dim=0).to(dtype=emb.dtype)
    return (w[:, None] * x).sum(0), int(x.size(0))


def _source_state(pred_pos, gt_pos):
    pred = set(int(x) for x in pred_pos)
    gt = set(int(x) for x in gt_pos)
    if not gt:
        return "stable", 1.0, 1.0, 0
    inter = len(pred & gt)
    recall = inter / max(1, len(gt))
    precision = inter / max(1, len(pred))
    if recall < 0.50:
        state = "under"
    elif precision < 0.50:
        state = "boundary"
    elif recall < 0.80:
        state = "moderate"
    else:
        state = "stable"
    return state, float(precision), float(recall), int(inter)


def _state_weights(state):
    return {
        "under": (1.4, 0.6),
        "moderate": (1.15, 0.9),
        "boundary": (0.9, 1.4),
        "stable": (0.8, 0.8),
    }.get(state, (1.0, 1.0))


def _mean(sum_value: float, count: int) -> float:
    return float(sum_value) / float(max(1, count))


def compute_state_conditioned_community_pic_loss(
    embedding,
    query_sets,
    gt_sets,
    margin=0.05,
    temperature=0.1,
    topk_core=64,
    hard_neg_topk=64,
    state_top_fraction=0.10,
):
    """Compute source-only SCPC using the historical V574 top-fraction state."""
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        raise ValueError("SCPC embedding must be a non-empty [N,D] tensor")
    fraction = float(state_top_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("state_top_fraction must be in (0,1]")

    device = embedding.device
    qsets = _as_list(query_sets)
    gsets = _as_list(gt_sets)
    losses = []
    state_counts = {"under": 0, "moderate": 0, "boundary": 0, "stable": 0}
    per_state: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    emb_norm = embedding.detach().float().norm(dim=1)
    report: dict[str, Any] = {
        "valid": 0,
        "intra": 0.0,
        "inter": 0.0,
        "states": state_counts,
        "state_prediction_mode": "score_global_top_fraction",
        "state_top_fraction": fraction,
        "state_score_quantile": 1.0 - fraction,
        "margin": float(margin),
        "embedding_norm_mean": float(emb_norm.mean().cpu().item()),
        "embedding_norm_std": float(emb_norm.std(unbiased=False).cpu().item()),
        "embedding_norm_min": float(emb_norm.min().cpu().item()),
        "embedding_norm_max": float(emb_norm.max().cpu().item()),
        "source_only": True,
        "target_labels_used": False,
        "target_gt_used": False,
    }

    sums = defaultdict(float)
    threshold_quantile = max(0.0, min(1.0, 1.0 - fraction))

    for q, pos in zip(qsets, gsets):
        pos = sorted({int(x) for x in pos if 0 <= int(x) < embedding.shape[0]})
        if not pos:
            continue
        q = [int(x) for x in (list(q) if q is not None else []) if 0 <= int(x) < embedding.shape[0]]
        qvec = embedding[q].mean(0) if q else embedding.mean(0)
        score = F.cosine_similarity(embedding, qvec[None, :], dim=1)

        # The state boundary must not backpropagate through the quantile.
        score_threshold = torch.quantile(score.detach(), threshold_quantile)
        pred_idx = torch.where(score >= score_threshold)[0].tolist()
        state, precision, recall, tp = _source_state(pred_idx, pos)
        state_counts[state] += 1

        pos_tensor = torch.as_tensor(pos, device=device, dtype=torch.long)
        mask = torch.ones(embedding.size(0), dtype=torch.bool, device=device)
        mask[pos_tensor] = False
        neg = torch.where(mask)[0]
        if neg.numel() == 0:
            continue
        neg = neg[torch.topk(score[neg], min(int(hard_neg_topk), neg.numel())).indices]
        proto, core_count = _soft_core(embedding, pos, score, temperature, topk_core)

        intra = ((embedding[pos_tensor] - proto) ** 2).sum(1).mean()
        inter = ((embedding[neg] - proto) ** 2).sum(1).mean()
        alpha, beta = _state_weights(state)
        ratio = intra / (intra + inter + 1e-8)
        raw_gap = float(margin) + intra - inter
        separation = F.relu(raw_gap)
        loss = float(alpha) * ratio + float(beta) * 0.5 * separation
        losses.append(loss)

        emb_pos_n = F.normalize(embedding[pos_tensor], p=2, dim=-1, eps=1e-12)
        emb_neg_n = F.normalize(embedding[neg], p=2, dim=-1, eps=1e-12)
        proto_n = F.normalize(proto[None, :], p=2, dim=-1, eps=1e-12)
        cosine_intra = (1.0 - (emb_pos_n * proto_n).sum(dim=-1)).mean()
        cosine_inter = (1.0 - (emb_neg_n * proto_n).sum(dim=-1)).mean()
        cosine_ratio = cosine_intra / (cosine_intra + cosine_inter + 1e-8)

        n_pred = len(pred_idx)
        n_gt = len(pos)
        normalized_gap = raw_gap / (intra + inter + 1e-8)
        separation_active = float((raw_gap.detach() > 0).float().cpu().item())
        values = {
            "intra": float(intra.detach().cpu().item()),
            "inter": float(inter.detach().cpu().item()),
            "ratio_term": float(ratio.detach().cpu().item()),
            "separation_term": float(separation.detach().cpu().item()),
            "separation_active": separation_active,
            "margin_gap_raw": float(raw_gap.detach().cpu().item()),
            "margin_gap_normalized": float(normalized_gap.detach().cpu().item()),
            "cosine_intra": float(cosine_intra.detach().cpu().item()),
            "cosine_inter": float(cosine_inter.detach().cpu().item()),
            "cosine_ratio": float(cosine_ratio.detach().cpu().item()),
            "pred_size": float(n_pred),
            "gt_size": float(n_gt),
            "pred_gt_size_ratio": float(n_pred) / float(max(1, n_gt)),
            "state_precision": precision,
            "state_recall": recall,
            "state_tp": float(tp),
            "core_count": float(core_count),
            "prototype_norm": float(proto.detach().float().norm().cpu().item()),
            "score_threshold": float(score_threshold.detach().cpu().item()),
            "loss": float(loss.detach().cpu().item()),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        report["valid"] += 1
        for key, value in values.items():
            sums[key] += value
            per_state[state][key] += value
        per_state[state]["count"] += 1.0

    if not losses:
        report["per_state"] = {s: {"count": int(state_counts[s])} for s in state_counts}
        return torch.zeros((), device=device, requires_grad=True), report

    count = len(losses)
    report["intra"] = _mean(sums["intra"], count)
    report["inter"] = _mean(sums["inter"], count)
    for key in (
        "ratio_term", "separation_term", "margin_gap_raw", "margin_gap_normalized",
        "cosine_intra", "cosine_inter", "cosine_ratio", "pred_size", "gt_size",
        "pred_gt_size_ratio", "state_precision", "state_recall", "core_count",
        "prototype_norm", "score_threshold", "loss",
    ):
        report[f"mean_{key}"] = _mean(sums[key], count)
    report["separation_active_rate"] = _mean(sums["separation_active"], count)
    report["per_state"] = {}
    for state in ("under", "moderate", "boundary", "stable"):
        c = int(per_state[state].get("count", 0.0))
        row: dict[str, Any] = {"count": c, "fraction": float(c) / float(max(1, count))}
        if c:
            for key in (
                "intra", "inter", "ratio_term", "separation_term", "separation_active",
                "margin_gap_normalized", "cosine_intra", "cosine_inter",
                "pred_gt_size_ratio", "state_precision", "state_recall", "prototype_norm",
                "loss", "alpha", "beta",
            ):
                row[f"mean_{key}"] = _mean(per_state[state].get(key, 0.0), c)
        report["per_state"][state] = row

    return torch.stack(losses).mean(), report
