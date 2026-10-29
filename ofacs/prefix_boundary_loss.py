"""Source-only prefix-aware boundary losses for OFA-CS V5.

The fixed OFA-CS evaluator is not a pairwise classifier: it ranks nodes by
query-mean cosine scores and then applies the senior GlobalSearch prefix rule.
This module adds differentiable losses around that *selected prefix* while using
only source-query ground truth during training.  It never reads target ``*.gt``
files.

The 2026-06-28 state-aware refinement extends the previous static prefix loss
with source-query state diagnosis.  The goal is not to tune one global loss
weight for every graph, but to route source queries into over-expanded,
under-recalled, mixed-boundary, or stable states and apply the appropriate
precision / size / hard-positive / boundary-ranking signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .errors import V5ProtocolError
from .senior_compat import parse_query_lines, senior_global_search_from_ranked_scores, senior_global_search_prefix_limit


@dataclass(frozen=True)
class PrefixBoundaryLossConfig:
    margin: float = 0.05
    query_limit: int = 8
    neg_topk: int = 64
    pos_bottomk: int = 64
    fp_topk: int = 64
    precision_loss_weight: float = 0.0
    adaptive_precision: bool = False
    precision_over_ratio: float = 1.25
    pairwise_boundary_loss: bool = False
    fn_loss_weight: float = 1.0
    size_loss_weight: float = 0.0
    size_over_tolerance: float = 0.15

    # New state-aware loss routing.  Disabled by default to preserve exact
    # behavior of the earlier validated runs.
    state_aware_loss: bool = False
    stratified_query_sampling: bool = False
    state_candidate_limit: int = 128
    state_over_ratio: float = 1.40
    state_under_fn_ratio: float = 0.20
    state_mixed_gap: float = 0.02
    state_precision_mult: float = 1.50
    state_size_mult: float = 1.50
    state_fn_mult: float = 1.50
    state_pairwise_mult: float = 1.00
    state_hard_positive_mult: float = 2.00

    # Source-only query-local structural hard negatives.  This targets the Cora-like
    # failure mode where communities are not merely over-expanded but have
    # structurally close non-community nodes mixed into the boundary.
    structural_boundary_loss: bool = False
    structural_boundary_weight: float = 0.0
    structural_boundary_hop: int = 2
    structural_neg_topk: int = 64
    state_stratified_over: float = 0.40
    state_stratified_mixed: float = 0.30
    state_stratified_under: float = 0.20
    state_stratified_stable: float = 0.10

    eps: float = 1e-8

    def validate(self) -> "PrefixBoundaryLossConfig":
        if self.query_limit <= 0:
            raise V5ProtocolError("prefix query_limit must be positive")
        if self.neg_topk <= 0 or self.pos_bottomk <= 0 or self.fp_topk <= 0:
            raise V5ProtocolError("prefix neg_topk/pos_bottomk/fp_topk must be positive")
        if self.precision_loss_weight < 0:
            raise V5ProtocolError("prefix precision_loss_weight must be non-negative")
        if self.precision_over_ratio <= 0 or self.state_over_ratio <= 0:
            raise V5ProtocolError("precision/state ratio thresholds must be positive")
        if self.fn_loss_weight < 0:
            raise V5ProtocolError("prefix fn_loss_weight must be non-negative")
        if self.state_candidate_limit <= 0:
            raise V5ProtocolError("state_candidate_limit must be positive")
        if self.state_hard_positive_mult <= 0:
            raise V5ProtocolError("state_hard_positive_mult must be positive")
        if self.structural_boundary_weight < 0:
            raise V5ProtocolError("structural_boundary_weight must be non-negative")
        if self.structural_boundary_hop <= 0 or self.structural_neg_topk <= 0:
            raise V5ProtocolError("structural_boundary_hop/structural_neg_topk must be positive")
        if any(x < 0 for x in (self.state_stratified_over, self.state_stratified_mixed, self.state_stratified_under, self.state_stratified_stable)):
            raise V5ProtocolError("state stratified quotas must be non-negative")
        if self.eps <= 0:
            raise V5ProtocolError("eps must be positive")
        return self


@dataclass(frozen=True)
class PrefixBoundaryLossReport:
    query_count: int
    used_query_count: int
    avg_pred_size: float
    avg_gt_size: float
    avg_pred_over_gt: float
    avg_fp_count: float
    avg_fn_count: float
    prefix_rank_loss: float
    prefix_precision_loss: float
    prefix_size_loss: float
    skipped_queries: int
    state_over_count: int = 0
    state_under_count: int = 0
    state_mixed_count: int = 0
    state_stable_count: int = 0
    avg_boundary_gap: float = 0.0
    structural_boundary_loss: float = 0.0
    structural_boundary_queries: int = 0
    state_aware_loss: bool = False
    stratified_query_sampling: bool = False
    labels_used: bool = True
    target_gt_used: bool = False
    metrics_used: bool = False
    source_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _QueryDiag:
    qi: int
    pred_size: int
    gt_size: int
    fp_count: int
    fn_count: int
    pred_over_gt: float
    fn_ratio: float
    boundary_gap: float
    state: str


def load_source_gt_sets(path: str | Path) -> tuple[tuple[int, ...], ...]:
    p = Path(path)
    if not p.exists():
        raise V5ProtocolError(f"source gt file not found: {p}")
    return parse_query_lines(p.read_text(encoding="utf-8").splitlines())


def _normalize_query_sets(query_sets: Sequence[Sequence[int]], *, num_nodes: int) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for q in query_sets:
        vals = tuple(dict.fromkeys(int(x) for x in q))
        if not vals:
            raise V5ProtocolError("empty source query set")
        if min(vals) < 0 or max(vals) >= num_nodes:
            raise V5ProtocolError("source query id outside graph")
        out.append(vals)
    return tuple(out)


def _normalize_gt_sets(gt_sets: Sequence[Sequence[int]], *, num_nodes: int) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for g in gt_sets:
        vals = tuple(dict.fromkeys(int(x) for x in g))
        if not vals:
            raise V5ProtocolError("empty source gt community")
        if min(vals) < 0 or max(vals) >= num_nodes:
            raise V5ProtocolError("source gt id outside graph")
        out.append(vals)
    return tuple(out)


def _margin_rank_loss(neg_scores: torch.Tensor, pos_scores: torch.Tensor, *, margin: float, pairwise: bool) -> torch.Tensor:
    if neg_scores.numel() == 0 or pos_scores.numel() == 0:
        return neg_scores.new_tensor(0.0)
    if pairwise:
        return F.softplus(float(margin) + neg_scores.view(-1, 1) - pos_scores.view(1, -1)).mean()
    return F.softplus(float(margin) + neg_scores.mean() - pos_scores.mean())


def _adaptive_precision_gate(pred_size: int, gt_size: int, *, threshold: float, eps: float) -> float:
    ratio = float(pred_size + eps) / float(gt_size + eps)
    if ratio <= float(threshold):
        return 0.0
    return float(min(1.0, max(0.0, (ratio - float(threshold)) / max(float(threshold), eps))))


def _state_from_diag(*, pred_over_gt: float, fn_ratio: float, fp_count: int, fn_count: int, boundary_gap: float, cfg: PrefixBoundaryLossConfig) -> str:
    over = pred_over_gt >= float(cfg.state_over_ratio) and fp_count > 0
    under = fn_ratio >= float(cfg.state_under_fn_ratio) and fn_count > 0
    mixed = fp_count > 0 and fn_count > 0 and boundary_gap <= float(cfg.state_mixed_gap)
    if over and not under:
        return "over"
    if under and not over:
        return "under"
    if mixed or (over and under):
        return "mixed"
    return "stable"


def _diagnose_query(
    emb: torch.Tensor,
    qset: Sequence[int],
    gt: set[int],
    *,
    qi: int,
    cfg: PrefixBoundaryLossConfig,
    top_k: int,
) -> _QueryDiag:
    device = emb.device
    q_ids = torch.tensor(tuple(qset), dtype=torch.long, device=device)
    q_emb = F.normalize(emb.index_select(0, q_ids).mean(dim=0, keepdim=True), p=2, dim=-1, eps=cfg.eps)
    scores = (q_emb @ emb.t()).squeeze(0)
    ranked_scores_t, ranked_nodes_t = torch.topk(scores.detach(), k=top_k, largest=True, sorted=True)
    q_seed_scores = [float(scores[int(seed)].detach().cpu().item()) for seed in qset]
    pred = senior_global_search_from_ranked_scores(
        query_nodes=qset,
        query_scores=q_seed_scores,
        ranked_nodes=[int(x) for x in ranked_nodes_t.detach().cpu().tolist()],
        ranked_scores=[float(x) for x in ranked_scores_t.detach().cpu().tolist()],
        num_nodes=int(emb.shape[0]),
        full_score_sum=float(scores.detach().sum().cpu().item()),
    )
    pred_set = set(int(x) for x in pred)
    fp = sorted(pred_set - gt)
    fn = sorted(gt - pred_set)
    pred_over_gt = float(len(pred_set) + cfg.eps) / float(len(gt) + cfg.eps)
    fn_ratio = float(len(fn)) / float(max(1, len(gt)))

    # Boundary gap: weak positives minus hard negatives.  Small/negative gap
    # indicates mixed ranking around the GlobalSearch boundary.
    pos_ids = torch.tensor(sorted(gt), dtype=torch.long, device=device)
    pos_scores = scores.index_select(0, pos_ids)
    pos_k = min(int(cfg.pos_bottomk), int(pos_scores.numel()))
    weak_pos = torch.topk(pos_scores, k=pos_k, largest=False, sorted=False).values
    if fp:
        fp_ids = torch.tensor(fp, dtype=torch.long, device=device)
        fp_scores = scores.index_select(0, fp_ids)
        fp_k = min(int(cfg.fp_topk), int(fp_scores.numel()))
        hard_neg = torch.topk(fp_scores, k=fp_k, largest=True, sorted=False).values
    else:
        neg_mask = torch.ones(int(emb.shape[0]), dtype=torch.bool, device=device)
        neg_mask[pos_ids] = False
        neg_scores = scores[neg_mask]
        neg_k = min(int(cfg.neg_topk), int(neg_scores.numel()))
        hard_neg = torch.topk(neg_scores, k=neg_k, largest=True, sorted=False).values
    boundary_gap = float((weak_pos.mean() - hard_neg.mean()).detach().cpu().item()) if weak_pos.numel() and hard_neg.numel() else 0.0
    state = _state_from_diag(
        pred_over_gt=pred_over_gt,
        fn_ratio=fn_ratio,
        fp_count=len(fp),
        fn_count=len(fn),
        boundary_gap=boundary_gap,
        cfg=cfg,
    )
    return _QueryDiag(qi=int(qi), pred_size=len(pred_set), gt_size=len(gt), fp_count=len(fp), fn_count=len(fn), pred_over_gt=pred_over_gt, fn_ratio=fn_ratio, boundary_gap=boundary_gap, state=state)


def _random_query_indices(q_count: int, query_limit: int, *, device: torch.device) -> torch.Tensor:
    limit = min(int(query_limit), int(q_count))
    if limit >= q_count:
        return torch.arange(q_count, device=device, dtype=torch.long)
    return torch.randperm(q_count, device=device)[:limit]


def _stratified_query_indices(diags: Sequence[_QueryDiag], cfg: PrefixBoundaryLossConfig, *, device: torch.device) -> torch.Tensor:
    limit = min(int(cfg.query_limit), len(diags))
    if limit >= len(diags):
        return torch.tensor([d.qi for d in diags], device=device, dtype=torch.long)
    buckets: dict[str, list[int]] = {"over": [], "mixed": [], "under": [], "stable": []}
    for d in diags:
        buckets.setdefault(d.state, []).append(int(d.qi))
    quotas = {
        "over": int(round(limit * float(cfg.state_stratified_over))),
        "mixed": int(round(limit * float(cfg.state_stratified_mixed))),
        "under": int(round(limit * float(cfg.state_stratified_under))),
        "stable": int(round(limit * float(cfg.state_stratified_stable))),
    }
    # Ensure at least one slot for non-empty important buckets when possible.
    for name in ("over", "mixed", "under"):
        if buckets.get(name) and quotas.get(name, 0) == 0 and sum(quotas.values()) < limit:
            quotas[name] = 1
    selected: list[int] = []
    gen = torch.Generator(device="cpu")
    # Use global RNG seed indirectly by drawing a torch seed; keeps stochasticity
    # compatible with caller while simplifying CPU bucket shuffling.
    gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    for name in ("over", "mixed", "under", "stable"):
        ids = list(buckets.get(name, []))
        if not ids:
            continue
        perm = torch.randperm(len(ids), generator=gen).tolist()
        take = min(int(quotas.get(name, 0)), len(ids))
        selected.extend([ids[j] for j in perm[:take]])
    if len(selected) < limit:
        remaining = [d.qi for d in diags if d.qi not in set(selected)]
        if remaining:
            perm = torch.randperm(len(remaining), generator=gen).tolist()
            selected.extend([remaining[j] for j in perm[: limit - len(selected)]])
    return torch.tensor(selected[:limit], device=device, dtype=torch.long)


def _build_adj_lists(edge_index: torch.Tensor | Sequence[Sequence[int]] | None, *, num_nodes: int) -> list[set[int]] | None:
    if edge_index is None:
        return None
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().long()
        if ei.ndim != 2:
            return None
        if ei.shape[0] != 2 and ei.shape[1] == 2:
            ei = ei.t().contiguous()
        if ei.shape[0] != 2:
            return None
        rows = ei[0].tolist()
        cols = ei[1].tolist()
    else:
        vals = list(edge_index)
        if len(vals) != 2:
            return None
        rows = [int(x) for x in vals[0]]
        cols = [int(x) for x in vals[1]]
    adj = [set() for _ in range(int(num_nodes))]
    for u, v in zip(rows, cols):
        u_i, v_i = int(u), int(v)
        if 0 <= u_i < num_nodes and 0 <= v_i < num_nodes and u_i != v_i:
            adj[u_i].add(v_i)
            adj[v_i].add(u_i)
    return adj


def _structural_shell(adj: list[set[int]] | None, seeds: set[int], *, hop: int, num_nodes: int) -> set[int]:
    if adj is None or not seeds:
        return set()
    visited = set(int(x) for x in seeds if 0 <= int(x) < num_nodes)
    frontier = set(visited)
    for _ in range(int(hop)):
        nxt: set[int] = set()
        for u in frontier:
            nxt.update(adj[int(u)])
        nxt -= visited
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return visited - set(seeds)


def compute_prefix_boundary_loss(
    embeddings: torch.Tensor,
    query_sets: Sequence[Sequence[int]],
    gt_sets: Sequence[Sequence[int]],
    *,
    config: PrefixBoundaryLossConfig | None = None,
    edge_index: torch.Tensor | Sequence[Sequence[int]] | None = None,
) -> tuple[torch.Tensor, PrefixBoundaryLossReport]:
    cfg = (config or PrefixBoundaryLossConfig()).validate()
    if embeddings.ndim != 2:
        raise V5ProtocolError(f"embeddings must be [N,D], got {tuple(embeddings.shape)}")
    n = int(embeddings.shape[0])
    if n <= 0:
        raise V5ProtocolError("embeddings must be non-empty")
    qsets = _normalize_query_sets(query_sets, num_nodes=n)
    gsets = _normalize_gt_sets(gt_sets, num_nodes=n)
    if len(qsets) != len(gsets):
        raise V5ProtocolError(f"source query/gt count mismatch: {len(qsets)} vs {len(gsets)}")

    device = embeddings.device
    dtype = embeddings.dtype if embeddings.is_floating_point() else torch.float32
    emb = F.normalize(embeddings.to(dtype=dtype), p=2, dim=-1, eps=cfg.eps)
    endpoint_cap = senior_global_search_prefix_limit(n)
    max_seed = max(len(q) for q in qsets)
    top_k = min(n, endpoint_cap + max_seed)

    # Optional state diagnosis for stratified sampling and state-aware loss
    # routing.  This is source-only and uses detached scores; no target labels or
    # metrics are read.
    diag_by_q: dict[int, _QueryDiag] = {}
    if cfg.state_aware_loss or cfg.stratified_query_sampling:
        candidate_count = min(len(qsets), int(cfg.state_candidate_limit))
        if candidate_count < len(qsets):
            cand = torch.randperm(len(qsets), device=device)[:candidate_count].detach().cpu().tolist()
        else:
            cand = list(range(len(qsets)))
        with torch.no_grad():
            for qi in cand:
                gt = set(int(x) for x in gsets[int(qi)])
                diag = _diagnose_query(emb.detach(), qsets[int(qi)], gt, qi=int(qi), cfg=cfg, top_k=top_k)
                diag_by_q[int(qi)] = diag
    if cfg.stratified_query_sampling and diag_by_q:
        idx = _stratified_query_indices(tuple(diag_by_q.values()), cfg, device=device)
    else:
        idx = _random_query_indices(len(qsets), cfg.query_limit, device=device)

    losses: list[torch.Tensor] = []
    size_losses: list[torch.Tensor] = []
    precision_losses: list[torch.Tensor] = []
    pred_sizes: list[int] = []
    gt_sizes: list[int] = []
    fp_counts: list[int] = []
    fn_counts: list[int] = []
    boundary_gaps: list[float] = []
    state_counts = {"over": 0, "under": 0, "mixed": 0, "stable": 0}
    skipped = 0
    all_nodes = torch.arange(n, device=device, dtype=torch.long)
    adj_lists = _build_adj_lists(edge_index, num_nodes=n) if bool(cfg.structural_boundary_loss) and float(cfg.structural_boundary_weight) > 0 else None
    structural_losses: list[torch.Tensor] = []
    structural_query_count = 0

    for i_t in idx:
        qi = int(i_t.detach().cpu().item())
        qset = qsets[qi]
        gt = set(int(x) for x in gsets[qi])
        q_ids = torch.tensor(qset, dtype=torch.long, device=device)
        q_emb = F.normalize(emb.index_select(0, q_ids).mean(dim=0, keepdim=True), p=2, dim=-1, eps=cfg.eps)
        scores = (q_emb @ emb.t()).squeeze(0)
        ranked_scores_t, ranked_nodes_t = torch.topk(scores.detach(), k=top_k, largest=True, sorted=True)
        q_seed_scores = [float(scores[int(seed)].detach().cpu().item()) for seed in qset]
        pred = senior_global_search_from_ranked_scores(
            query_nodes=qset,
            query_scores=q_seed_scores,
            ranked_nodes=[int(x) for x in ranked_nodes_t.detach().cpu().tolist()],
            ranked_scores=[float(x) for x in ranked_scores_t.detach().cpu().tolist()],
            num_nodes=n,
            full_score_sum=float(scores.detach().sum().cpu().item()),
        )
        pred_set = set(int(x) for x in pred)
        fp = sorted(pred_set - gt)
        fn = sorted(gt - pred_set)
        pos_all = torch.tensor(sorted(gt), dtype=torch.long, device=device)
        neg_mask = torch.ones(n, dtype=torch.bool, device=device)
        neg_mask[pos_all] = False
        neg_all = all_nodes[neg_mask]
        if pos_all.numel() == 0 or neg_all.numel() == 0:
            skipped += 1
            continue

        # State-aware hard positive expansion: only for under/mixed cases.
        diag = diag_by_q.get(qi)
        state = diag.state if diag is not None else _state_from_diag(
            pred_over_gt=float(len(pred_set) + cfg.eps) / float(len(gt) + cfg.eps),
            fn_ratio=float(len(fn)) / float(max(1, len(gt))),
            fp_count=len(fp),
            fn_count=len(fn),
            boundary_gap=0.0,
            cfg=cfg,
        )
        state_counts[state] = state_counts.get(state, 0) + 1
        pos_bottomk = int(cfg.pos_bottomk)
        if cfg.state_aware_loss and state in {"under", "mixed"}:
            pos_bottomk = int(max(pos_bottomk, round(pos_bottomk * float(cfg.state_hard_positive_mult))))

        pos_scores = scores.index_select(0, pos_all)
        pos_k = min(pos_bottomk, int(pos_scores.numel()))
        boundary_pos_scores = torch.topk(pos_scores, k=pos_k, largest=False, sorted=False).values
        neg_scores_all = scores.index_select(0, neg_all)
        hard_neg_k = min(int(cfg.neg_topk), int(neg_scores_all.numel()))
        hard_neg_scores = torch.topk(neg_scores_all, k=hard_neg_k, largest=True, sorted=False).values
        if fp:
            neg_ids = torch.tensor(fp, dtype=torch.long, device=device)
            fp_scores_all = scores.index_select(0, neg_ids)
            fp_k = min(int(cfg.fp_topk), int(fp_scores_all.numel()))
            neg_scores = torch.topk(fp_scores_all, k=fp_k, largest=True, sorted=False).values
        else:
            neg_scores = hard_neg_scores
        if neg_scores.numel() == 0 or boundary_pos_scores.numel() == 0:
            skipped += 1
            continue

        use_pairwise = bool(cfg.pairwise_boundary_loss)
        if cfg.state_aware_loss and state == "mixed":
            use_pairwise = True
        rank_loss = _margin_rank_loss(neg_scores, boundary_pos_scores, margin=float(cfg.margin), pairwise=use_pairwise)
        precision_loss = _margin_rank_loss(hard_neg_scores, boundary_pos_scores, margin=float(cfg.margin), pairwise=use_pairwise)
        precision_weight = float(cfg.precision_loss_weight)
        if precision_weight > 0:
            if bool(cfg.adaptive_precision):
                precision_weight *= _adaptive_precision_gate(len(pred_set), len(gt), threshold=float(cfg.precision_over_ratio), eps=float(cfg.eps))
            if cfg.state_aware_loss:
                if state == "over":
                    precision_weight *= float(cfg.state_precision_mult)
                elif state == "under":
                    precision_weight *= 0.50
                elif state == "mixed":
                    precision_weight *= float(cfg.state_pairwise_mult)
            rank_loss = rank_loss + precision_weight * precision_loss
        precision_losses.append(precision_loss)

        if adj_lists is not None and float(cfg.structural_boundary_weight) > 0:
            # Structural hard negatives are local-shell nodes around the query/gt
            # community that are not in the source gt.  This is source-only and
            # intended for Cora-like local boundary confusions.
            seeds = set(int(x) for x in qset) | set(int(x) for x in gt)
            shell = _structural_shell(adj_lists, seeds, hop=int(cfg.structural_boundary_hop), num_nodes=n)
            struct_neg = sorted(int(x) for x in (shell - gt) if 0 <= int(x) < n)
            if struct_neg:
                struct_ids = torch.tensor(struct_neg, dtype=torch.long, device=device)
                struct_scores = scores.index_select(0, struct_ids)
                struct_k = min(int(cfg.structural_neg_topk), int(struct_scores.numel()))
                struct_hard = torch.topk(struct_scores, k=struct_k, largest=True, sorted=False).values
                s_loss = _margin_rank_loss(struct_hard, boundary_pos_scores, margin=float(cfg.margin), pairwise=use_pairwise)
                rank_loss = rank_loss + float(cfg.structural_boundary_weight) * s_loss
                structural_losses.append(s_loss)
                structural_query_count += 1

        fn_weight = float(cfg.fn_loss_weight)
        if cfg.state_aware_loss:
            if state in {"under", "mixed"}:
                fn_weight *= float(cfg.state_fn_mult)
            elif state == "over":
                fn_weight *= 0.50
        if fn and fn_weight > 0:
            fn_ids = torch.tensor(fn, dtype=torch.long, device=device)
            fn_scores = scores.index_select(0, fn_ids)
            miss_loss = _margin_rank_loss(neg_scores, fn_scores, margin=float(cfg.margin), pairwise=use_pairwise)
            rank_loss = rank_loss + fn_weight * miss_loss
        losses.append(rank_loss)
        pred_sizes.append(len(pred_set))
        gt_sizes.append(len(gt))
        fp_counts.append(len(fp))
        fn_counts.append(len(fn))
        if diag is not None:
            boundary_gaps.append(float(diag.boundary_gap))

        size_weight = float(cfg.size_loss_weight)
        if cfg.state_aware_loss:
            if state == "over":
                size_weight *= float(cfg.state_size_mult)
            elif state == "under":
                size_weight *= 0.25
        over_ratio = torch.tensor(float(len(pred_set) + cfg.eps) / float(len(gt) + cfg.eps), device=device, dtype=dtype)
        size_losses.append(size_weight * F.relu(torch.log(over_ratio) - float(cfg.size_over_tolerance)))

    rank = torch.stack(losses).mean() if losses else embeddings.new_tensor(0.0)
    precision_loss_report = torch.stack(precision_losses).mean() if precision_losses else embeddings.new_tensor(0.0)
    size_loss = torch.stack(size_losses).mean() if size_losses else embeddings.new_tensor(0.0)
    total = rank + size_loss
    used = len(losses)
    avg_pred = float(sum(pred_sizes) / max(1, len(pred_sizes)))
    avg_gt = float(sum(gt_sizes) / max(1, len(gt_sizes)))
    report = PrefixBoundaryLossReport(
        query_count=len(qsets),
        used_query_count=used,
        avg_pred_size=avg_pred,
        avg_gt_size=avg_gt,
        avg_pred_over_gt=float(avg_pred / max(1e-8, avg_gt)),
        avg_fp_count=float(sum(fp_counts) / max(1, len(fp_counts))),
        avg_fn_count=float(sum(fn_counts) / max(1, len(fn_counts))),
        prefix_rank_loss=float(rank.detach().cpu().item()),
        prefix_precision_loss=float(precision_loss_report.detach().cpu().item()),
        prefix_size_loss=float(size_loss.detach().cpu().item()),
        skipped_queries=int(skipped),
        state_over_count=int(state_counts.get("over", 0)),
        state_under_count=int(state_counts.get("under", 0)),
        state_mixed_count=int(state_counts.get("mixed", 0)),
        state_stable_count=int(state_counts.get("stable", 0)),
        avg_boundary_gap=float(sum(boundary_gaps) / max(1, len(boundary_gaps))),
        structural_boundary_loss=float(torch.stack(structural_losses).mean().detach().cpu().item()) if structural_losses else 0.0,
        structural_boundary_queries=int(structural_query_count),
        state_aware_loss=bool(cfg.state_aware_loss),
        stratified_query_sampling=bool(cfg.stratified_query_sampling),
    )
    return total, report
