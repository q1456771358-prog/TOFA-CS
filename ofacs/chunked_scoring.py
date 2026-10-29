"""Chunked senior-compatible query cosine scoring.

The function here preserves the senior score semantics:
``query_embedding = mean(seed embeddings)`` and ``score = cosine(query, node)``.
It avoids keeping a full GPU ``[Q, N]`` matrix alive, while still computing the
full score sum required by senior GlobalSearch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .errors import V5ProtocolError
from .senior_compat import SeniorQueryBatch, senior_global_search_prefix_limit


@dataclass(frozen=True)
class ChunkedScoreResult:
    query_node_sets: tuple[tuple[int, ...], ...]
    top_nodes: torch.Tensor  # [Q, K] CPU long
    top_scores: torch.Tensor  # [Q, K] CPU float
    full_score_sums: torch.Tensor  # [Q] CPU float
    query_seed_scores: tuple[dict[int, float], ...]
    num_nodes: int
    embedding_dim: int
    top_k_pool: int
    chunk_size: int
    chunk_count: int
    max_live_score_shape: tuple[int, int]
    full_qn_gpu_constructed: bool = False

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["top_nodes_shape"] = tuple(self.top_nodes.shape)
        d["top_scores_shape"] = tuple(self.top_scores.shape)
        d["full_score_sums_shape"] = tuple(self.full_score_sums.shape)
        d.pop("top_nodes", None)
        d.pop("top_scores", None)
        d.pop("full_score_sums", None)
        return d


def _normalize_query_sets(
    query_node_sets: SeniorQueryBatch | Sequence[Sequence[int] | torch.Tensor],
    *,
    num_nodes: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(query_node_sets, SeniorQueryBatch):
        qsets = query_node_sets.query_node_sets
    else:
        qsets = tuple(
            tuple(int(x) for x in (q.detach().cpu().flatten().tolist() if isinstance(q, torch.Tensor) else q))
            for q in query_node_sets
        )
    if not qsets:
        raise V5ProtocolError("at least one query set is required")
    out: list[tuple[int, ...]] = []
    for qset in qsets:
        dedup = tuple(dict.fromkeys(int(x) for x in qset))
        if not dedup:
            raise V5ProtocolError("empty query set is not allowed")
        if min(dedup) < 0 or max(dedup) >= int(num_nodes):
            raise V5ProtocolError(f"query set contains node outside [0,{num_nodes}): {dedup}")
        out.append(dedup)
    return tuple(out)


def default_top_k_pool(num_nodes: int, max_query_seed_count: int = 1) -> int:
    """Pool size required by senior GlobalSearch plus room for query seeds."""

    n = int(num_nodes)
    if n <= 0:
        raise V5ProtocolError("num_nodes must be positive")
    return min(n, senior_global_search_prefix_limit(n) + max(1, int(max_query_seed_count)))


def chunked_query_cosine(
    node_embeddings: torch.Tensor,
    query_node_sets: SeniorQueryBatch | Sequence[Sequence[int] | torch.Tensor],
    *,
    chunk_size: int = 4096,
    top_k_pool: int | None = None,
    eps: float = 1e-12,
    score_calibration: str = "none",
) -> ChunkedScoreResult:
    """Compute top scores and full score sums without a persistent full ``[Q,N]`` tensor.

    Args:
        node_embeddings: ``[N, H]`` tensor.  It may live on CPU or GPU.
        query_node_sets: senior query seed sets using local node IDs.
        chunk_size: number of candidate nodes per scoring chunk.
        top_k_pool: retained candidate pool.  Defaults to senior endpoint cap.
        eps: cosine normalization epsilon.

    Returns:
        CPU tensors for top nodes/scores and full score sums.
    """

    if node_embeddings.ndim != 2:
        raise V5ProtocolError(f"node_embeddings must be [N,H], got {tuple(node_embeddings.shape)}")
    n, h = int(node_embeddings.shape[0]), int(node_embeddings.shape[1])
    if n <= 0 or h <= 0:
        raise V5ProtocolError("node_embeddings must have positive shape")
    if int(chunk_size) <= 0:
        raise V5ProtocolError("chunk_size must be positive")

    qsets = _normalize_query_sets(query_node_sets, num_nodes=n)
    q_count = len(qsets)
    max_q = max(len(q) for q in qsets)
    k_pool = default_top_k_pool(n, max_q) if top_k_pool is None else min(int(top_k_pool), n)
    if k_pool <= 0:
        raise V5ProtocolError("top_k_pool must be positive")

    device = node_embeddings.device
    dtype = node_embeddings.dtype if node_embeddings.is_floating_point() else torch.float32
    embeddings = node_embeddings.to(dtype=dtype)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1, eps=eps)

    query_embs = []
    for qset in qsets:
        q_ids = torch.tensor(qset, dtype=torch.long, device=device)
        query_embs.append(embeddings_norm.index_select(0, q_ids).mean(dim=0))
    query_embs = F.normalize(torch.stack(query_embs, dim=0), p=2, dim=-1, eps=eps)  # [Q,H]

    calibration = str(score_calibration or "none").lower()
    if calibration not in {"none", "center", "zscore"}:
        raise V5ProtocolError(f"unsupported score_calibration={score_calibration!r}")
    cal_mean = torch.zeros(q_count, dtype=torch.float64, device=device)
    cal_std = torch.ones(q_count, dtype=torch.float64, device=device)
    if calibration != "none":
        sum1 = torch.zeros(q_count, dtype=torch.float64, device=device)
        sum2 = torch.zeros(q_count, dtype=torch.float64, device=device)
        for start in range(0, n, int(chunk_size)):
            end = min(start + int(chunk_size), n)
            chunk = embeddings_norm[start:end]
            scores = query_embs @ chunk.t()
            sum1 += scores.double().sum(dim=1)
            sum2 += scores.double().pow(2).sum(dim=1)
        cal_mean = sum1 / float(n)
        if calibration == "zscore":
            var = torch.clamp(sum2 / float(n) - cal_mean.pow(2), min=1e-12)
            cal_std = torch.sqrt(var)

    # Running top-k state; shape [Q, <=K].  This is bounded by senior endpoint cap.
    top_scores = torch.empty((q_count, 0), dtype=dtype, device=device)
    top_nodes = torch.empty((q_count, 0), dtype=torch.long, device=device)
    full_sums = torch.zeros(q_count, dtype=torch.float64, device=device)
    query_seed_scores: list[dict[int, float]] = [dict() for _ in range(q_count)]

    chunk_count = 0
    max_live_c = 0
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = embeddings_norm[start:end]
        scores = query_embs @ chunk.t()  # [Q,C]
        if calibration != "none":
            scores = (scores.double() - cal_mean.view(-1, 1)) / cal_std.view(-1, 1)
            scores = scores.to(dtype=dtype)
        chunk_count += 1
        max_live_c = max(max_live_c, int(scores.shape[1]))
        full_sums += scores.double().sum(dim=1)

        node_ids = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0).expand(q_count, -1)
        merged_scores = torch.cat([top_scores, scores], dim=1)
        merged_nodes = torch.cat([top_nodes, node_ids], dim=1)
        keep = min(k_pool, int(merged_scores.shape[1]))
        top_scores, pos = torch.topk(merged_scores, k=keep, dim=1, largest=True, sorted=True)
        top_nodes = torch.gather(merged_nodes, dim=1, index=pos)

        # Keep exact query seed scores without full score matrix.
        for qi, qset in enumerate(qsets):
            for seed in qset:
                if start <= seed < end:
                    query_seed_scores[qi][int(seed)] = float(scores[qi, seed - start].detach().cpu().item())

    return ChunkedScoreResult(
        query_node_sets=qsets,
        top_nodes=top_nodes.detach().cpu(),
        top_scores=top_scores.detach().cpu(),
        full_score_sums=full_sums.detach().cpu().float(),
        query_seed_scores=tuple(query_seed_scores),
        num_nodes=n,
        embedding_dim=h,
        top_k_pool=k_pool,
        chunk_size=int(chunk_size),
        chunk_count=int(chunk_count),
        max_live_score_shape=(q_count, max_live_c),
        full_qn_gpu_constructed=False,
    )


def full_query_cosine_for_small_graphs(
    node_embeddings: torch.Tensor,
    query_node_sets: SeniorQueryBatch | Sequence[Sequence[int] | torch.Tensor],
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reference full score matrix for tests and tiny source diagnostics only."""

    if node_embeddings.ndim != 2:
        raise V5ProtocolError("node_embeddings must be [N,H]")
    n = int(node_embeddings.shape[0])
    qsets = _normalize_query_sets(query_node_sets, num_nodes=n)
    emb = F.normalize(node_embeddings.float(), p=2, dim=-1, eps=eps)
    q_embs = []
    for qset in qsets:
        q_ids = torch.tensor(qset, dtype=torch.long, device=emb.device)
        q_embs.append(emb.index_select(0, q_ids).mean(dim=0))
    q_embs = F.normalize(torch.stack(q_embs, dim=0), p=2, dim=-1, eps=eps)
    return q_embs @ emb.t()
