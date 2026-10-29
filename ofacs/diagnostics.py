"""Source-only ranking, boundary, and diffusion diagnostics for V5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .errors import V5DataLeakageError, V5ProtocolError
from .senior_compat import senior_global_search_prefix_limit


@dataclass(frozen=True)
class RankingDiagnosticReport:
    graph_name: str
    query_index: int
    query_size: int
    positive_count: int
    negative_count: int
    positive_score_mean: float
    negative_score_mean: float
    score_gap: float
    hard_negative_score_mean: float
    recall_at_gt_size: float
    topk_purity_at_gt_size: float
    oracle_best_f1: float
    oracle_best_jaccard: float
    oracle_best_k: int
    selected_k: int | None
    selected_f1: float | None
    selected_jaccard: float | None
    source_only: bool = True
    labels_used: bool = True
    ground_truth_used: bool = False
    metrics_used_for_training: bool = False
    full_qn_gpu_constructed: bool = False

    def __post_init__(self) -> None:
        if not self.source_only or self.metrics_used_for_training:
            raise V5DataLeakageError("ranking diagnostics must be source-only and not used for training")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryDiagnosticReport:
    graph_name: str
    query_index: int
    selected_k: int
    gt_size: int
    selected_over_gt: float
    oracle_best_k: int
    oracle_best_f1: float
    endpoint_limit: int
    selected_hits_endpoint: bool
    source_only: bool = True
    labels_used: bool = True
    ground_truth_used: bool = False
    metrics_used_for_training: bool = False

    def __post_init__(self) -> None:
        if not self.source_only or self.metrics_used_for_training:
            raise V5DataLeakageError("boundary diagnostics must be source-only and not used for training")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionPrePostDiagnosticReport:
    graph_name: str
    query_index: int
    pre_gap: float
    post_gap: float
    gap_delta: float
    correction_norm: float
    pre_post_cosine: float
    pre_entropy: float
    post_entropy: float
    selected_k_pre: int | None = None
    selected_k_post: int | None = None
    source_only: bool = True
    labels_used: bool = True
    ground_truth_used: bool = False
    metrics_used_for_training: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_bool_mask(ids_or_mask: torch.Tensor | Sequence[int], *, num_nodes: int, device: torch.device) -> torch.Tensor:
    if isinstance(ids_or_mask, torch.Tensor) and ids_or_mask.dtype == torch.bool:
        mask = ids_or_mask.to(device=device).flatten()
        if mask.numel() != num_nodes:
            raise V5ProtocolError("boolean mask length mismatch")
        return mask
    if isinstance(ids_or_mask, torch.Tensor):
        ids = ids_or_mask.detach().cpu().long().flatten()
    else:
        ids = torch.tensor([int(x) for x in ids_or_mask], dtype=torch.long)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if ids.numel() > 0:
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= num_nodes:
            raise V5ProtocolError("id outside graph range")
        mask[ids.to(device=device)] = True
    return mask


def _binary_metrics(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> tuple[float, float]:
    tp = (pred_mask & true_mask).sum().float()
    pred = pred_mask.sum().float()
    true = true_mask.sum().float()
    precision = tp / torch.clamp(pred, min=1.0)
    recall = tp / torch.clamp(true, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    union = (pred_mask | true_mask).sum().float()
    jac = tp / torch.clamp(union, min=1.0)
    return float(f1.item()), float(jac.item())


def oracle_prefix(scores: torch.Tensor, positive_mask: torch.Tensor, *, max_k: int | None = None) -> tuple[int, float, float]:
    scores = scores.detach().float().flatten()
    n = int(scores.numel())
    if positive_mask.numel() != n:
        raise V5ProtocolError("positive mask length mismatch")
    limit = n if max_k is None else min(int(max_k), n)
    order = torch.argsort(scores, descending=True)[:limit]
    true = positive_mask.bool()
    best_k, best_f1, best_jac = 1, -1.0, 0.0
    pred_mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
    for i, node in enumerate(order.tolist(), start=1):
        pred_mask[int(node)] = True
        f1, jac = _binary_metrics(pred_mask, true)
        if f1 > best_f1:
            best_k, best_f1, best_jac = i, f1, jac
    return int(best_k), float(best_f1), float(best_jac)


def ranking_diagnostic_from_scores(
    *,
    graph_name: str,
    query_index: int,
    scores: torch.Tensor,
    positive_ids_or_mask: torch.Tensor | Sequence[int],
    query_nodes: Sequence[int],
    selected_nodes: Sequence[int] | None = None,
    hard_negative_topk: int = 64,
    source_only: bool = True,
) -> RankingDiagnosticReport:
    """Decompose representation ranking quality for a source query."""

    if not source_only:
        raise V5DataLeakageError("ranking diagnostics with labels are source-only")
    scores = scores.detach().float().flatten().cpu()
    n = int(scores.numel())
    positive_mask = _as_bool_mask(positive_ids_or_mask, num_nodes=n, device=scores.device).cpu()
    negative_mask = ~positive_mask
    if positive_mask.sum().item() <= 0 or negative_mask.sum().item() <= 0:
        raise V5ProtocolError("positive and negative sets must both be non-empty")
    pos_scores = scores[positive_mask]
    neg_scores = scores[negative_mask]
    gt_size = int(positive_mask.sum().item())
    order = torch.argsort(scores, descending=True)
    top_gt = order[:gt_size]
    recall_at_gt = float(positive_mask[top_gt].sum().float().div(max(1, gt_size)).item())
    topk_purity = recall_at_gt
    hard_k = min(int(hard_negative_topk), int(neg_scores.numel()))
    hard_neg_mean = float(torch.topk(neg_scores, k=hard_k, largest=True).values.mean().item()) if hard_k > 0 else 0.0
    best_k, best_f1, best_jac = oracle_prefix(scores, positive_mask)

    selected_k = selected_f1 = selected_jac = None
    if selected_nodes is not None:
        pred = _as_bool_mask(selected_nodes, num_nodes=n, device=scores.device)
        selected_k = int(pred.sum().item())
        selected_f1, selected_jac = _binary_metrics(pred, positive_mask)

    return RankingDiagnosticReport(
        graph_name=graph_name,
        query_index=int(query_index),
        query_size=len(tuple(query_nodes)),
        positive_count=int(positive_mask.sum().item()),
        negative_count=int(negative_mask.sum().item()),
        positive_score_mean=float(pos_scores.mean().item()),
        negative_score_mean=float(neg_scores.mean().item()),
        score_gap=float(pos_scores.mean().item() - neg_scores.mean().item()),
        hard_negative_score_mean=hard_neg_mean,
        recall_at_gt_size=recall_at_gt,
        topk_purity_at_gt_size=topk_purity,
        oracle_best_f1=best_f1,
        oracle_best_jaccard=best_jac,
        oracle_best_k=best_k,
        selected_k=selected_k,
        selected_f1=selected_f1,
        selected_jaccard=selected_jac,
        source_only=True,
        labels_used=True,
        ground_truth_used=False,
        metrics_used_for_training=False,
        full_qn_gpu_constructed=False,
    )


def boundary_diagnostic_from_selection(
    *,
    graph_name: str,
    query_index: int,
    scores: torch.Tensor,
    positive_ids_or_mask: torch.Tensor | Sequence[int],
    selected_nodes: Sequence[int],
    source_only: bool = True,
) -> BoundaryDiagnosticReport:
    if not source_only:
        raise V5DataLeakageError("boundary diagnostics with labels are source-only")
    scores = scores.detach().float().flatten().cpu()
    n = int(scores.numel())
    positive_mask = _as_bool_mask(positive_ids_or_mask, num_nodes=n, device=scores.device).cpu()
    gt_size = int(positive_mask.sum().item())
    selected_k = len(tuple(dict.fromkeys(int(x) for x in selected_nodes)))
    oracle_k, oracle_f1, _ = oracle_prefix(scores, positive_mask)
    endpoint = senior_global_search_prefix_limit(n)
    return BoundaryDiagnosticReport(
        graph_name=graph_name,
        query_index=int(query_index),
        selected_k=int(selected_k),
        gt_size=int(gt_size),
        selected_over_gt=float(selected_k / max(1, gt_size)),
        oracle_best_k=int(oracle_k),
        oracle_best_f1=float(oracle_f1),
        endpoint_limit=int(endpoint),
        selected_hits_endpoint=bool(selected_k >= endpoint),
        source_only=True,
        labels_used=True,
        ground_truth_used=False,
        metrics_used_for_training=False,
    )


def score_entropy(scores: torch.Tensor, eps: float = 1e-12) -> float:
    s = scores.detach().float().flatten().cpu()
    p = torch.softmax(s, dim=0)
    ent = -(p * torch.log(p + eps)).sum() / torch.log(torch.tensor(float(max(2, s.numel()))))
    return float(ent.item())


def diffusion_prepost_diagnostic(
    *,
    graph_name: str,
    query_index: int,
    pre_embeddings: torch.Tensor,
    post_embeddings: torch.Tensor,
    query_nodes: Sequence[int],
    positive_ids_or_mask: torch.Tensor | Sequence[int],
    selected_k_pre: int | None = None,
    selected_k_post: int | None = None,
    source_only: bool = True,
) -> DiffusionPrePostDiagnosticReport:
    if not source_only:
        raise V5DataLeakageError("diffusion pre/post diagnostics with labels are source-only")
    if pre_embeddings.shape != post_embeddings.shape or pre_embeddings.ndim != 2:
        raise V5ProtocolError("pre/post embeddings must have same [N,H] shape")
    n = int(pre_embeddings.shape[0])
    q = torch.tensor([int(x) for x in query_nodes], dtype=torch.long, device=pre_embeddings.device)
    if q.numel() == 0 or int(q.min().item()) < 0 or int(q.max().item()) >= n:
        raise V5ProtocolError("invalid query nodes")
    pos_mask = _as_bool_mask(positive_ids_or_mask, num_nodes=n, device=pre_embeddings.device)
    neg_mask = ~pos_mask

    def _scores(emb: torch.Tensor) -> torch.Tensor:
        e = F.normalize(emb.float(), p=2, dim=-1)
        q_emb = F.normalize(e.index_select(0, q).mean(dim=0, keepdim=True), p=2, dim=-1)
        return (e * q_emb).sum(dim=-1)

    pre_s = _scores(pre_embeddings)
    post_s = _scores(post_embeddings)
    pre_gap = float((pre_s[pos_mask].mean() - pre_s[neg_mask].mean()).detach().cpu().item())
    post_gap = float((post_s[pos_mask].mean() - post_s[neg_mask].mean()).detach().cpu().item())
    correction_norm = float((post_embeddings - pre_embeddings).float().norm(dim=-1).mean().detach().cpu().item())
    pre_post_cos = float(F.cosine_similarity(pre_embeddings.float(), post_embeddings.float(), dim=-1).mean().detach().cpu().item())
    return DiffusionPrePostDiagnosticReport(
        graph_name=graph_name,
        query_index=int(query_index),
        pre_gap=pre_gap,
        post_gap=post_gap,
        gap_delta=post_gap - pre_gap,
        correction_norm=correction_norm,
        pre_post_cosine=pre_post_cos,
        pre_entropy=score_entropy(pre_s),
        post_entropy=score_entropy(post_s),
        selected_k_pre=selected_k_pre,
        selected_k_post=selected_k_post,
        source_only=True,
        labels_used=True,
        ground_truth_used=False,
        metrics_used_for_training=False,
    )
