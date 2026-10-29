"""Memory-aware source query-chunk training helpers for V5.

This module keeps V5 source training away from full-graph autograd.  It builds
the source-only query-boundary supervision plan once, then trains over bounded
query chunks by encoding only the node IDs that appear in those samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .errors import V5ProtocolError
from .query_boundary_loss import QueryBoundarySample, compute_query_boundary_loss
from .runtime_profile import profile_event
from .source_query_boundary_plan import build_source_query_boundary_plan
from .training_loop import V5Stage1LossConfig


@dataclass(frozen=True)
class QueryChunkTrainingResult:
    graph_name: str
    epoch: int
    steps_run: int
    samples_seen: int
    avg_total_loss: float
    avg_query_boundary_loss: float
    avg_diffusion_loss: float | None
    max_chunk_node_count: int
    max_query_chunk_size: int
    source_only: bool = True
    unseen_training_used: bool = False
    full_cartesian_used: bool = False
    full_graph_autograd_used: bool = False
    loss_list_backward_used: bool = False
    reduced_query_count_for_resources: bool = False
    score_aware_sampling_enabled: bool = False
    score_aware_sampling_used: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sample_node_ids(sample: QueryBoundarySample) -> tuple[int, ...]:
    ids: list[int] = []
    for field_name in (
        "query_local_ids",
        "positive_local_ids",
        "negative_local_ids",
        "boundary_positive_local_ids",
        "boundary_negative_local_ids",
        "high_score_negative_local_ids",
        "low_score_positive_local_ids",
    ):
        ids.extend(int(x) for x in getattr(sample, field_name))
    return tuple(dict.fromkeys(ids))


def _remap_ids(ids: Sequence[int], id_map: dict[int, int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(id_map[int(x)]) for x in ids))


def remap_query_boundary_samples(samples: Sequence[QueryBoundarySample]) -> tuple[torch.Tensor, tuple[QueryBoundarySample, ...]]:
    """Collect chunk-local nodes and remap sample IDs to ``[0, chunk_nodes)``."""

    if not samples:
        raise V5ProtocolError("cannot remap an empty query chunk")
    unique_ids = tuple(dict.fromkeys(x for sample in samples for x in _sample_node_ids(sample)))
    if not unique_ids:
        raise V5ProtocolError("query chunk did not contain any node IDs")
    if min(unique_ids) < 0:
        raise V5ProtocolError("query chunk contains negative node ID")
    id_map = {old: new for new, old in enumerate(unique_ids)}
    remapped: list[QueryBoundarySample] = []
    for sample in samples:
        remapped.append(
            QueryBoundarySample(
                query_local_ids=_remap_ids(sample.query_local_ids, id_map),
                positive_local_ids=_remap_ids(sample.positive_local_ids, id_map),
                negative_local_ids=_remap_ids(sample.negative_local_ids, id_map),
                boundary_positive_local_ids=_remap_ids(sample.boundary_positive_local_ids, id_map) if sample.boundary_positive_local_ids else (),
                boundary_negative_local_ids=_remap_ids(sample.boundary_negative_local_ids, id_map) if sample.boundary_negative_local_ids else (),
                high_score_negative_local_ids=_remap_ids(sample.high_score_negative_local_ids, id_map) if sample.high_score_negative_local_ids else (),
                low_score_positive_local_ids=_remap_ids(sample.low_score_positive_local_ids, id_map) if sample.low_score_positive_local_ids else (),
                graph_name=sample.graph_name,
                source_only=True,
                sample_reason_stats=sample.sample_reason_stats,
            )
        )
    return torch.tensor(unique_ids, dtype=torch.long), tuple(remapped)


def run_query_chunk_training_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_token_seqs: torch.Tensor,
    labels: torch.Tensor,
    graph_name: str,
    epoch: int,
    device: torch.device,
    query_node_sets: Sequence[Sequence[int] | torch.Tensor] | None = None,
    edge_index: torch.Tensor | Sequence[Sequence[int]] | None = None,
    node_features: torch.Tensor | None = None,
    structure_condition: torch.Tensor | None = None,
    config: V5Stage1LossConfig | None = None,
    query_chunk_size: int = 16,
    max_query_chunk_nodes: int | None = None,
    max_steps: int | None = None,
    start_global_step: int = 0,
    enable_score_aware_mining: bool = True,
    score_aware_embedding_chunk_size: int = 8192,
    events_jsonl: str | Path | None = None,
    runtime_jsonl: str | Path | None = None,
) -> QueryChunkTrainingResult:
    """Run one source-only epoch with per-query-chunk optimizer steps."""

    cfg = (config or V5Stage1LossConfig()).validate()
    if global_token_seqs.ndim != 3:
        raise V5ProtocolError(f"global_token_seqs must be [N,hops+1,D], got {tuple(global_token_seqs.shape)}")
    if labels.numel() != int(global_token_seqs.shape[0]):
        raise V5ProtocolError("labels/global_token_seqs num_nodes mismatch")
    if int(query_chunk_size) <= 0:
        raise V5ProtocolError("query_chunk_size must be positive")
    if max_query_chunk_nodes is not None and int(max_query_chunk_nodes) <= 0:
        raise V5ProtocolError("max_query_chunk_nodes must be positive when set")

    plan = build_source_query_boundary_plan(
        labels=labels.detach().cpu().long(),
        graph_name=graph_name,
        query_node_sets=query_node_sets,
        edge_index=edge_index,
        node_features=node_features,
        config=cfg.sampler,
        split="source_train",
    )
    score_aware_used = 0
    if bool(enable_score_aware_mining):
        # Build source-only current score vectors from the current model state.
        # This is bounded CPU storage (Q score vectors for source queries) and is
        # never used on unseen graphs or for checkpoint selection.
        qsets = tuple(sample.query_local_ids for sample in plan.samples)
        current_scores = _build_source_current_scores(
            model=model,
            global_token_seqs=global_token_seqs,
            query_node_sets=qsets,
            device=device,
            structure_condition=structure_condition,
            chunk_size=int(score_aware_embedding_chunk_size),
            runtime_jsonl=runtime_jsonl,
            graph_name=graph_name,
            epoch=epoch,
        )
        plan = build_source_query_boundary_plan(
            labels=labels.detach().cpu().long(),
            graph_name=graph_name,
            query_node_sets=qsets,
            edge_index=edge_index,
            node_features=node_features,
            current_scores=current_scores,
            config=cfg.sampler,
            split="source_train",
        )
        score_aware_used = int(plan.report.get("used_score_aware_sampling", 0))
    samples = list(plan.samples)
    event_path = Path(events_jsonl) if events_jsonl is not None else None
    if event_path is not None:
        event_path.parent.mkdir(parents=True, exist_ok=True)

    loss_values: list[float] = []
    qb_values: list[float] = []
    diffusion_values: list[float] = []
    max_nodes = 0
    steps_run = 0
    samples_seen = 0
    model.train()

    for offset in range(0, len(samples), int(query_chunk_size)):
        if max_steps is not None and steps_run >= int(max_steps):
            break
        chunk_samples = samples[offset : offset + int(query_chunk_size)]
        unique_ids, remapped_samples = remap_query_boundary_samples(chunk_samples)
        if int(unique_ids.max().item()) >= int(global_token_seqs.shape[0]):
            raise V5ProtocolError("query chunk node ID outside global_token_seqs")
        chunk_node_count = int(unique_ids.numel())
        if max_query_chunk_nodes is not None and chunk_node_count > int(max_query_chunk_nodes):
            raise V5ProtocolError(
                f"query chunk needs {chunk_node_count} nodes, above max_query_chunk_nodes={max_query_chunk_nodes}"
            )
        max_nodes = max(max_nodes, chunk_node_count)

        profile_event(
            runtime_jsonl,
            "query_chunk_train_step_start",
            graph_name=graph_name,
            epoch=epoch,
            global_step=start_global_step + steps_run + 1,
            chunk_offset=offset,
            chunk_sample_count=len(chunk_samples),
            chunk_node_count=chunk_node_count,
        )
        chunk_tokens = global_token_seqs.index_select(0, unique_ids).to(device)
        chunk_condition = None
        if structure_condition is not None:
            if structure_condition.ndim != 2 or int(structure_condition.shape[0]) != int(global_token_seqs.shape[0]):
                raise V5ProtocolError("structure_condition must be [N,C] and match global_token_seqs")
            chunk_condition = structure_condition.index_select(0, unique_ids).to(device)
        optimizer.zero_grad(set_to_none=True)
        forward = model(None, chunk_tokens, structure_condition=chunk_condition, compute_diffusion_loss=True)
        qb = compute_query_boundary_loss(
            forward.refined_embedding,
            remapped_samples,
            config=cfg.query_boundary,
        )
        total = qb.total_loss
        diffusion_value = None
        if forward.diffusion_loss is not None and float(cfg.diffusion_weight) > 0:
            total = total + float(cfg.diffusion_weight) * forward.diffusion_loss
            diffusion_value = float(forward.diffusion_loss.detach().cpu().item())
            diffusion_values.append(diffusion_value)
        total.backward()
        optimizer.step()

        steps_run += 1
        samples_seen += len(chunk_samples)
        loss_value = float(total.detach().cpu().item())
        qb_value = float(qb.total_loss.detach().cpu().item())
        loss_values.append(loss_value)
        qb_values.append(qb_value)
        event = {
            "epoch": int(epoch),
            "global_step": int(start_global_step + steps_run),
            "chunk_offset": int(offset),
            "chunk_sample_count": int(len(chunk_samples)),
            "chunk_node_count": chunk_node_count,
            "total_loss": loss_value,
            "query_boundary_loss": qb_value,
            "diffusion_loss": diffusion_value,
            "query_boundary": qb.report.to_dict(),
            "source_query_plan": plan.report,
            "score_aware_sampling_enabled": bool(enable_score_aware_mining),
            "score_aware_sampling_used": int(score_aware_used),
            "training_mode": "query_chunk",
            "source_only": True,
            "unseen_training_used": False,
            "full_cartesian_used": False,
            "full_graph_autograd_used": False,
            "loss_list_backward_used": False,
        }
        if event_path is not None:
            with event_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(event, ensure_ascii=False, sort_keys=True))
        profile_event(
            runtime_jsonl,
            "query_chunk_train_step_end",
            graph_name=graph_name,
            epoch=epoch,
            global_step=start_global_step + steps_run,
            total_loss=loss_value,
        )

    if steps_run <= 0:
        raise V5ProtocolError("query-chunk epoch did not run any steps")
    return QueryChunkTrainingResult(
        graph_name=graph_name,
        epoch=int(epoch),
        steps_run=steps_run,
        samples_seen=samples_seen,
        avg_total_loss=float(sum(loss_values) / max(1, len(loss_values))),
        avg_query_boundary_loss=float(sum(qb_values) / max(1, len(qb_values))),
        avg_diffusion_loss=None if not diffusion_values else float(sum(diffusion_values) / len(diffusion_values)),
        max_chunk_node_count=max_nodes,
        max_query_chunk_size=int(query_chunk_size),
        score_aware_sampling_enabled=bool(enable_score_aware_mining),
        score_aware_sampling_used=int(score_aware_used),
    )


@torch.no_grad()
def _build_source_current_scores(
    *,
    model: torch.nn.Module,
    global_token_seqs: torch.Tensor,
    query_node_sets: Sequence[Sequence[int]],
    device: torch.device,
    structure_condition: torch.Tensor | None = None,
    chunk_size: int = 8192,
    runtime_jsonl: str | Path | None = None,
    graph_name: str = "source",
    epoch: int | None = None,
) -> dict[int, torch.Tensor]:
    """Compute source-only query-to-node score vectors for hard mining.

    The vectors live on CPU and are rebuilt from the current source model.  This
    avoids full [Q,N] GPU residency and does not use unseen labels, metrics, or
    ground truth.
    """

    profile_event(runtime_jsonl, "score_aware_source_embedding_start", graph_name=graph_name, epoch=epoch)
    emb = encode_refined_in_chunks(
        model=model,
        global_token_seqs=global_token_seqs,
        device=device,
        chunk_size=int(chunk_size),
        runtime_jsonl=runtime_jsonl,
        graph_name=graph_name,
        structure_condition=structure_condition,
    )
    emb = F.normalize(emb.float(), p=2, dim=-1, eps=1e-12)
    scores: dict[int, torch.Tensor] = {}
    n = int(emb.shape[0])
    for qi, qset in enumerate(query_node_sets):
        q_ids = torch.tensor([int(x) for x in qset], dtype=torch.long)
        if q_ids.numel() == 0 or int(q_ids.min().item()) < 0 or int(q_ids.max().item()) >= n:
            raise V5ProtocolError("score-aware query ids outside source graph")
        q = F.normalize(emb.index_select(0, q_ids).mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-12)
        scores[int(qi)] = (emb @ q.squeeze(0)).detach().cpu()
    profile_event(runtime_jsonl, "score_aware_source_embedding_end", graph_name=graph_name, epoch=epoch, query_count=len(scores))
    return scores


@torch.no_grad()

def adaptive_raw_guided_fusion(
    raw: torch.Tensor,
    guided: torch.Tensor,
    *,
    min_guided: float = 0.25,
    max_guided: float = 0.85,
    confidence_scale: float = 8.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Adaptive raw-guided fusion for target scoring embeddings.

    Fixed 0.5 averaging helped some targets but hurt others.  This gate keeps
    more raw representation when raw/guided disagree strongly, and trusts guided
    refinement more when both views are aligned.  It is label-free and operates
    per node.
    """

    if raw.shape != guided.shape:
        raise V5ProtocolError("raw/guided fusion shape mismatch")
    lo = float(min_guided)
    hi = float(max_guided)
    if hi < lo:
        lo, hi = hi, lo
    cos = torch.nn.functional.cosine_similarity(raw.float(), guided.float(), dim=-1, eps=eps).clamp(-1.0, 1.0)
    # Center around 0.5: cos≈1 -> high guided weight; cos small/negative -> raw-heavy.
    gate = torch.sigmoid((cos - 0.50) * float(confidence_scale)).unsqueeze(-1).to(dtype=raw.dtype, device=raw.device)
    gate = lo + (hi - lo) * gate
    return (1.0 - gate) * raw + gate * guided

def encode_refined_in_chunks(
    *,
    model: torch.nn.Module,
    global_token_seqs: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8192,
    runtime_jsonl: str | Path | None = None,
    graph_name: str = "source",
    t_eval: int | None = None,
    structure_condition: torch.Tensor | None = None,
    ratio_guidance_pair: object | None = None,
    ratio_guidance_weight: float = 0.0,
    ratio_guidance_clip: float = 5.0,
    adaptive_ratio_guidance: bool = False,
    ratio_guidance_confidence_scale: float = 1.0,
    ratio_guidance_min_mult: float = 0.25,
    ratio_guidance_max_mult: float = 1.25,
    guidance_diagnostics: object | None = None,
    guidance_runtime_timer: object | None = None,
) -> torch.Tensor:
    """Encode all nodes without full-graph gradients or full GPU residency."""

    if int(chunk_size) <= 0:
        raise V5ProtocolError("encode chunk_size must be positive")
    if global_token_seqs.ndim != 3:
        raise V5ProtocolError("global_token_seqs must be [N,hops+1,D]")
    was_training = bool(model.training)
    model.eval()
    chunks: list[torch.Tensor] = []
    n = int(global_token_seqs.shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        profile_event(runtime_jsonl, "encode_refined_chunk_start", graph_name=graph_name, start=start, end=end)
        cond = None
        if structure_condition is not None:
            if structure_condition.ndim != 2 or int(structure_condition.shape[0]) != n:
                raise V5ProtocolError("structure_condition must be [N,C] and match global_token_seqs")
            cond = structure_condition[start:end].to(device)
        emb = model.encode_refined(
            global_token_seqs[start:end].to(device),
            structure_condition=cond,
            t_eval=t_eval,
            stochastic=False,
            ratio_guidance_node=getattr(ratio_guidance_pair, "node", None) if ratio_guidance_pair is not None else None,
            ratio_guidance_neighbor=getattr(ratio_guidance_pair, "neighbor", None) if ratio_guidance_pair is not None else None,
            ratio_guidance_weight=ratio_guidance_weight,
            ratio_guidance_clip=ratio_guidance_clip,
            adaptive_ratio_guidance=adaptive_ratio_guidance,
            ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
            ratio_guidance_min_mult=ratio_guidance_min_mult,
            ratio_guidance_max_mult=ratio_guidance_max_mult,
            guidance_diagnostics=guidance_diagnostics,
            guidance_runtime_timer=guidance_runtime_timer,
        )
        chunks.append(emb.detach().cpu())
        profile_event(runtime_jsonl, "encode_refined_chunk_end", graph_name=graph_name, start=start, end=end)
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)



@torch.no_grad()
def encode_embedding_in_chunks(
    *,
    model: torch.nn.Module,
    global_token_seqs: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8192,
    runtime_jsonl: str | Path | None = None,
    graph_name: str = "source",
    t_eval: int | None = None,
    structure_condition: torch.Tensor | None = None,
    score_embedding: str = "refined",
    ratio_guidance_pair: object | None = None,
    ratio_guidance_weight: float = 0.0,
    ratio_guidance_clip: float = 5.0,
    adaptive_ratio_guidance: bool = False,
    ratio_guidance_confidence_scale: float = 1.0,
    ratio_guidance_min_mult: float = 0.25,
    ratio_guidance_max_mult: float = 1.25,
    adaptive_fusion: bool = False,
    fusion_min_guided: float = 0.25,
    fusion_max_guided: float = 0.85,
    fusion_confidence_scale: float = 8.0,
    guidance_diagnostics: object | None = None,
    guidance_runtime_timer: object | None = None,
) -> torch.Tensor:
    """Encode all nodes with a selectable scoring embedding.

    Modes:
    - ``raw``: senior transformer raw concat embedding, no diffusion.
    - ``refined``: deterministic full reverse denoising (previous default).
    - ``guided_refined``: refined embedding with density-ratio reverse guidance.
    - ``adaptive_guided_refined``: guided refined embedding with node-adaptive ratio strength.
    - ``one_step``: deterministic one-step x0 estimate, matching the training path.
    - ``raw_plus_refined``: average of raw and refined embeddings.
    - ``raw_plus_guided_refined``: fixed average of raw and guided refined embeddings.
    - ``adaptive_raw_plus_guided_refined``: label-free adaptive raw/guided fusion.
    """

    mode = str(score_embedding)
    if mode in {"refined", "guided_refined", "adaptive_guided_refined"}:
        if mode in {"guided_refined", "adaptive_guided_refined"} and ratio_guidance_pair is None:
            raise V5ProtocolError(f"score_embedding={mode} requires density_ratio_guidance checkpoint state")
        return encode_refined_in_chunks(
            model=model,
            global_token_seqs=global_token_seqs,
            device=device,
            chunk_size=chunk_size,
            runtime_jsonl=runtime_jsonl,
            graph_name=graph_name,
            t_eval=t_eval,
            structure_condition=structure_condition,
            ratio_guidance_pair=ratio_guidance_pair if mode in {"guided_refined", "adaptive_guided_refined"} else None,
            ratio_guidance_weight=ratio_guidance_weight if mode in {"guided_refined", "adaptive_guided_refined"} else 0.0,
            ratio_guidance_clip=ratio_guidance_clip,
            adaptive_ratio_guidance=bool(adaptive_ratio_guidance or mode == "adaptive_guided_refined"),
            ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
            ratio_guidance_min_mult=ratio_guidance_min_mult,
            ratio_guidance_max_mult=ratio_guidance_max_mult,
            guidance_diagnostics=guidance_diagnostics,
            guidance_runtime_timer=guidance_runtime_timer,
        )
    if mode not in {"raw", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}:
        raise V5ProtocolError(f"unsupported score_embedding={mode!r}")
    if int(chunk_size) <= 0:
        raise V5ProtocolError("encode chunk_size must be positive")
    was_training = bool(model.training)
    model.eval()
    chunks: list[torch.Tensor] = []
    n = int(global_token_seqs.shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        profile_event(runtime_jsonl, "encode_embedding_chunk_start", graph_name=graph_name, start=start, end=end, score_embedding=mode)
        cond = None
        if structure_condition is not None:
            if structure_condition.ndim != 2 or int(structure_condition.shape[0]) != n:
                raise V5ProtocolError("structure_condition must be [N,C] and match global_token_seqs")
            cond = structure_condition[start:end].to(device)
        tokens = global_token_seqs[start:end].to(device)
        if mode == "raw":
            emb = model.encode_raw(tokens)[2]
        elif mode == "one_step":
            emb = model.encode_one_step_estimate(tokens, structure_condition=cond, t_eval=t_eval, noise_mode="zero")
        elif mode == "raw_plus_refined":
            raw = model.encode_raw(tokens)[2]
            refined = model.encode_refined(tokens, structure_condition=cond, t_eval=t_eval, stochastic=False)
            if mode == "adaptive_raw_plus_guided_refined" or adaptive_fusion:
                emb = adaptive_raw_guided_fusion(
                    raw,
                    refined,
                    min_guided=fusion_min_guided,
                    max_guided=fusion_max_guided,
                    confidence_scale=fusion_confidence_scale,
                )
            else:
                emb = 0.5 * (raw + refined)
        else:
            if ratio_guidance_pair is None:
                raise V5ProtocolError(f"score_embedding={mode} requires density_ratio_guidance checkpoint state")
            raw = model.encode_raw(tokens)[2]
            refined = model.encode_refined(
                tokens,
                structure_condition=cond,
                t_eval=t_eval,
                stochastic=False,
                ratio_guidance_node=getattr(ratio_guidance_pair, "node", None),
                ratio_guidance_neighbor=getattr(ratio_guidance_pair, "neighbor", None),
                ratio_guidance_weight=ratio_guidance_weight,
                ratio_guidance_clip=ratio_guidance_clip,
                adaptive_ratio_guidance=bool(adaptive_ratio_guidance),
                ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
                ratio_guidance_min_mult=ratio_guidance_min_mult,
                ratio_guidance_max_mult=ratio_guidance_max_mult,
                guidance_diagnostics=guidance_diagnostics,
                guidance_runtime_timer=guidance_runtime_timer,
            )
            if mode == "adaptive_raw_plus_guided_refined" or adaptive_fusion:
                emb = adaptive_raw_guided_fusion(
                    raw,
                    refined,
                    min_guided=fusion_min_guided,
                    max_guided=fusion_max_guided,
                    confidence_scale=fusion_confidence_scale,
                )
            else:
                emb = 0.5 * (raw + refined)
        chunks.append(emb.detach().cpu())
        profile_event(runtime_jsonl, "encode_embedding_chunk_end", graph_name=graph_name, start=start, end=end, score_embedding=mode)
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def iter_embedding_chunks(
    *,
    model: torch.nn.Module,
    global_token_seqs: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8192,
    runtime_jsonl: str | Path | None = None,
    graph_name: str = "source",
    t_eval: int | None = None,
    structure_condition: torch.Tensor | None = None,
    score_embedding: str = "refined",
    ratio_guidance_pair: object | None = None,
    ratio_guidance_weight: float = 0.0,
    ratio_guidance_clip: float = 5.0,
    adaptive_ratio_guidance: bool = False,
    ratio_guidance_confidence_scale: float = 1.0,
    ratio_guidance_min_mult: float = 0.25,
    ratio_guidance_max_mult: float = 1.25,
    adaptive_fusion: bool = False,
    fusion_min_guided: float = 0.25,
    fusion_max_guided: float = 0.85,
    fusion_confidence_scale: float = 8.0,
    guidance_diagnostics: object | None = None,
    guidance_runtime_timer: object | None = None,
):
    """Yield fixed inference embedding chunks for a selectable scoring view."""

    mode = str(score_embedding)
    if mode in {"refined", "guided_refined", "adaptive_guided_refined"}:
        if mode in {"guided_refined", "adaptive_guided_refined"} and ratio_guidance_pair is None:
            raise V5ProtocolError(f"score_embedding={mode} requires density_ratio_guidance checkpoint state")
        yield from iter_refined_embedding_chunks(
            model=model,
            global_token_seqs=global_token_seqs,
            device=device,
            chunk_size=chunk_size,
            runtime_jsonl=runtime_jsonl,
            graph_name=graph_name,
            t_eval=t_eval,
            structure_condition=structure_condition,
            ratio_guidance_pair=ratio_guidance_pair if mode in {"guided_refined", "adaptive_guided_refined"} else None,
            ratio_guidance_weight=ratio_guidance_weight if mode in {"guided_refined", "adaptive_guided_refined"} else 0.0,
            ratio_guidance_clip=ratio_guidance_clip,
            adaptive_ratio_guidance=bool(adaptive_ratio_guidance or mode == "adaptive_guided_refined"),
            ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
            ratio_guidance_min_mult=ratio_guidance_min_mult,
            ratio_guidance_max_mult=ratio_guidance_max_mult,
            guidance_diagnostics=guidance_diagnostics,
            guidance_runtime_timer=guidance_runtime_timer,
        )
        return
    if mode not in {"raw", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}:
        raise V5ProtocolError(f"unsupported score_embedding={mode!r}")
    if int(chunk_size) <= 0:
        raise V5ProtocolError("encode chunk_size must be positive")
    was_training = bool(model.training)
    model.eval()
    n = int(global_token_seqs.shape[0])
    try:
        for start in range(0, n, int(chunk_size)):
            end = min(n, start + int(chunk_size))
            profile_event(runtime_jsonl, "iter_embedding_chunk_start", graph_name=graph_name, start=start, end=end, score_embedding=mode)
            cond = None
            if structure_condition is not None:
                if structure_condition.ndim != 2 or int(structure_condition.shape[0]) != n:
                    raise V5ProtocolError("structure_condition must be [N,C] and match global_token_seqs")
                cond = structure_condition[start:end].to(device)
            tokens = global_token_seqs[start:end].to(device)
            if mode == "raw":
                emb = model.encode_raw(tokens)[2]
            elif mode == "one_step":
                emb = model.encode_one_step_estimate(tokens, structure_condition=cond, t_eval=t_eval, noise_mode="zero")
            elif mode == "raw_plus_refined":
                raw = model.encode_raw(tokens)[2]
                refined = model.encode_refined(tokens, structure_condition=cond, t_eval=t_eval, stochastic=False)
                emb = 0.5 * (raw + refined)
            else:
                if ratio_guidance_pair is None:
                    raise V5ProtocolError(f"score_embedding={mode} requires density_ratio_guidance checkpoint state")
                raw = model.encode_raw(tokens)[2]
                refined = model.encode_refined(
                    tokens,
                    structure_condition=cond,
                    t_eval=t_eval,
                    stochastic=False,
                    ratio_guidance_node=getattr(ratio_guidance_pair, "node", None),
                    ratio_guidance_neighbor=getattr(ratio_guidance_pair, "neighbor", None),
                    ratio_guidance_weight=ratio_guidance_weight,
                    ratio_guidance_clip=ratio_guidance_clip,
                    adaptive_ratio_guidance=bool(adaptive_ratio_guidance),
                    ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
                    ratio_guidance_min_mult=ratio_guidance_min_mult,
                    ratio_guidance_max_mult=ratio_guidance_max_mult,
                    guidance_diagnostics=guidance_diagnostics,
                    guidance_runtime_timer=guidance_runtime_timer,
                )
                if mode == "adaptive_raw_plus_guided_refined" or adaptive_fusion:
                    emb = adaptive_raw_guided_fusion(
                        raw,
                        refined,
                        min_guided=fusion_min_guided,
                        max_guided=fusion_max_guided,
                        confidence_scale=fusion_confidence_scale,
                    )
                else:
                    emb = 0.5 * (raw + refined)
            profile_event(runtime_jsonl, "iter_embedding_chunk_end", graph_name=graph_name, start=start, end=end, score_embedding=mode)
            yield start, end, emb.detach().cpu()
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def iter_refined_embedding_chunks(
    *,
    model: torch.nn.Module,
    global_token_seqs: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8192,
    runtime_jsonl: str | Path | None = None,
    graph_name: str = "source",
    t_eval: int | None = None,
    structure_condition: torch.Tensor | None = None,
    ratio_guidance_pair: object | None = None,
    ratio_guidance_weight: float = 0.0,
    ratio_guidance_clip: float = 5.0,
    adaptive_ratio_guidance: bool = False,
    ratio_guidance_confidence_scale: float = 1.0,
    ratio_guidance_min_mult: float = 0.25,
    ratio_guidance_max_mult: float = 1.25,
    guidance_diagnostics: object | None = None,
    guidance_runtime_timer: object | None = None,
):
    """Yield ``(start, end, embedding_chunk)`` without concatenating all chunks."""

    if int(chunk_size) <= 0:
        raise V5ProtocolError("encode chunk_size must be positive")
    if global_token_seqs.ndim != 3:
        raise V5ProtocolError("global_token_seqs must be [N,hops+1,D]")
    was_training = bool(model.training)
    model.eval()
    n = int(global_token_seqs.shape[0])
    try:
        for start in range(0, n, int(chunk_size)):
            end = min(n, start + int(chunk_size))
            profile_event(runtime_jsonl, "encode_refined_chunk_start", graph_name=graph_name, start=start, end=end)
            cond = None
            if structure_condition is not None:
                if structure_condition.ndim != 2 or int(structure_condition.shape[0]) != n:
                    raise V5ProtocolError("structure_condition must be [N,C] and match global_token_seqs")
                cond = structure_condition[start:end].to(device)
            emb = model.encode_refined(
                global_token_seqs[start:end].to(device),
                structure_condition=cond,
                t_eval=t_eval,
                stochastic=False,
                ratio_guidance_node=getattr(ratio_guidance_pair, "node", None) if ratio_guidance_pair is not None else None,
                ratio_guidance_neighbor=getattr(ratio_guidance_pair, "neighbor", None) if ratio_guidance_pair is not None else None,
                ratio_guidance_weight=ratio_guidance_weight,
                ratio_guidance_clip=ratio_guidance_clip,
                adaptive_ratio_guidance=adaptive_ratio_guidance,
                ratio_guidance_confidence_scale=ratio_guidance_confidence_scale,
                ratio_guidance_min_mult=ratio_guidance_min_mult,
                ratio_guidance_max_mult=ratio_guidance_max_mult,
                guidance_diagnostics=guidance_diagnostics,
                guidance_runtime_timer=guidance_runtime_timer,
            )
            profile_event(runtime_jsonl, "encode_refined_chunk_end", graph_name=graph_name, start=start, end=end)
            yield start, end, emb.detach().cpu()
    finally:
        if was_training:
            model.train()
