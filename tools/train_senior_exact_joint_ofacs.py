#!/usr/bin/env python3
"""Joint SeniorExact source training for the paper-table setting.

This is the paper-table counterpart of train_senior_exact_ofacs.py.  It trains one
shared SeniorExact model over multiple source graphs that all use the strict
senior d64/khop4/Nl10/pos10/neg10 cache.  It intentionally does not use V5
query-boundary loss, score-aware mining, domain guidance, or hashing.

Default source weighting is step_weighted: each source contributes
ceil(num_nodes / batch_size) full-graph optimization steps per epoch, matching
approximately the step distribution induced by a senior DataLoader without
reducing graph semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from senior_core.model_v5 import PretrainModelV5
from senior_core.lr import PolynomialDecayLR
from ofacs.errors import V5ProtocolError
from ofacs.fixed_checkpoint import save_source_checkpoint
from ofacs.runtime_profile import profile_event
from ofacs.randomness import make_generator, set_global_seed
from ofacs.senior_data import load_senior_graph_data
from ofacs.prefix_boundary_loss import PrefixBoundaryLossConfig, compute_prefix_boundary_loss, load_source_gt_sets
from ofacs.density_ratio_guidance import DensityRatioGuidancePair, compute_source_target_alignment_loss
from ofacs.state_conditioned_boundary_loss import compute_state_conditioned_boundary_loss
from ofacs.core_prototype_consistency_loss import compute_core_prototype_consistency_loss
from ofacs.state_conditioned_community_pic_loss import compute_state_conditioned_community_pic_loss
from ofacs.controlled_community_error_state_synthesis import (
    ControlledStateSynthesisConfig,
    build_controlled_state_selection_plan,
    compute_controlled_community_error_state_loss,
)
from ofacs.loss_gradient_diagnostics import compute_loss_gradient_diagnostics


def _timing_mark(enabled: bool, device: torch.device) -> float:
    """Return a synchronized timestamp only for the dedicated timing run."""

    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--source-graphs", required=True, help="comma-separated source graphs, e.g. texas,photo,dblp")
    p.add_argument("--target-graphs", default="", help="comma-separated unlabeled target graphs used only for optional target-unlabeled alignment/guidance losses")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3, help="base/peak learning rate; senior original uses 0.01 with polynomial schedule")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam", help="adam keeps previous V5 behavior; adamw matches senior original optimizer")
    p.add_argument("--lr-scheduler", choices=["none", "polynomial"], default="none", help="polynomial matches senior original PolynomialDecayLR")
    p.add_argument("--warmup-updates", type=int, default=400)
    p.add_argument("--tot-updates", type=int, default=1000)
    p.add_argument("--end-lr", type=float, default=1e-9)
    p.add_argument("--lr-power", type=float, default=1.0)
    p.add_argument("--dims", type=int, default=64)
    p.add_argument("--hops", type=int, default=4)
    p.add_argument("--local-topk", type=int, default=10)
    p.add_argument("--pos-count", type=int, default=10)
    p.add_argument("--neg-count", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--attention-dropout", type=float, default=0.4)
    p.add_argument("--readout", choices=["mean", "sum", "max"], default="mean")
    p.add_argument("--diffusion-T", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--senior-contrastive-weight", type=float, default=1.0)
    p.add_argument("--diffusion-weight", type=float, default=1.0)
    p.add_argument("--enable-link-loss", action="store_true")
    p.add_argument("--max-link-loss-nodes", type=int, default=10000)
    p.add_argument("--max-full-graph-nodes", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=10000, help="used for step_weighted source repetition only; training uses full-graph tensors per step")
    p.add_argument("--source-weighting", choices=["step_weighted", "uniform", "explicit"], default="step_weighted", help="final mainline uses senior-faithful step weighting; explicit is kept only for ablations")
    p.add_argument("--source-repeat-weights", default="texas=1,photo=1,dblp=3", help="for --source-weighting explicit, e.g. texas=1,photo=1,dblp=3")
    p.add_argument("--enable-prefix-loss", action="store_true", help="source-only GlobalSearch-prefix boundary loss; uses source query/gt only")
    p.add_argument("--prefix-loss-weight", type=float, default=0.05)
    p.add_argument("--prefix-on", choices=["raw", "refined", "both"], default="refined")
    p.add_argument("--prefix-margin", type=float, default=0.05)
    p.add_argument("--prefix-query-limit", type=int, default=8)
    p.add_argument("--prefix-neg-topk", type=int, default=64)
    p.add_argument("--prefix-pos-bottomk", type=int, default=64)
    p.add_argument("--prefix-fp-topk", type=int, default=64, help="hard false-positive top-k used by prefix boundary loss")
    p.add_argument("--prefix-precision-loss-weight", type=float, default=0.20, help="extra precision-oriented hard-negative loss weight")
    p.add_argument("--prefix-adaptive-precision", action=argparse.BooleanOptionalAction, default=False, help="gate precision loss by source predicted/gt size ratio")
    p.add_argument("--prefix-precision-over-ratio", type=float, default=1.25, help="over-prediction ratio threshold for adaptive precision loss")
    p.add_argument("--prefix-pairwise-boundary-loss", action=argparse.BooleanOptionalAction, default=False, help="use pairwise hard-neg vs weak-pos boundary ranking instead of mean-score loss")
    p.add_argument("--prefix-fn-loss-weight", type=float, default=1.0, help="weight for source-only false-negative recovery term")
    p.add_argument("--prefix-state-aware-loss", action=argparse.BooleanOptionalAction, default=False, help="route prefix/size/precision/FN losses by source query error state")
    p.add_argument("--prefix-stratified-query-sampling", action=argparse.BooleanOptionalAction, default=False, help="sample source queries from over/under/mixed/stable state buckets")
    p.add_argument("--prefix-state-candidate-limit", type=int, default=128)
    p.add_argument("--prefix-state-over-ratio", type=float, default=1.40)
    p.add_argument("--prefix-state-under-fn-ratio", type=float, default=0.20)
    p.add_argument("--prefix-state-mixed-gap", type=float, default=0.02)
    p.add_argument("--prefix-state-precision-mult", type=float, default=1.50)
    p.add_argument("--prefix-state-size-mult", type=float, default=1.50)
    p.add_argument("--prefix-state-fn-mult", type=float, default=1.50)
    p.add_argument("--prefix-state-pairwise-mult", type=float, default=1.00)
    p.add_argument("--prefix-state-hard-positive-mult", type=float, default=2.00)
    p.add_argument("--enable-structural-boundary-loss", action=argparse.BooleanOptionalAction, default=False, help="source-only query-local structural hard negative boundary loss")
    p.add_argument("--structural-boundary-weight", type=float, default=0.0)
    p.add_argument("--structural-boundary-hop", type=int, default=2)
    p.add_argument("--structural-neg-topk", type=int, default=64)
    p.add_argument("--enable-size-loss", action="store_true", help="penalize source prefixes that are much larger than source gt communities")
    p.add_argument("--enable-state-conditioned-boundary-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--state-boundary-loss-weight", type=float, default=0.05)
    p.add_argument("--state-boundary-margin", type=float, default=0.05)
    p.add_argument("--state-boundary-topk-pos", type=int, default=64)
    p.add_argument("--state-boundary-topk-neg", type=int, default=64)
    p.add_argument("--state-boundary-under-recall-threshold", type=float, default=0.25)
    p.add_argument("--state-boundary-over-precision-threshold", type=float, default=0.70)
    p.add_argument("--enable-core-prototype-consistency-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enable-state-conditioned-community-pic-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--community-pic-loss-weight", type=float, default=0.05)
    p.add_argument("--community-pic-temperature", type=float, default=0.1)
    p.add_argument("--community-pic-topk-core", type=int, default=64)
    p.add_argument("--community-pic-hard-neg-topk", type=int, default=64)
    p.add_argument("--community-pic-state-top-fraction", type=float, default=0.10, help="source-only pseudo-community fraction; 0.10 is retained for checkpoint compatibility")
    p.add_argument("--enable-controlled-error-state-synthesis-loss", action=argparse.BooleanOptionalAction, default=False, help="source-only controlled community error-state synthesis auxiliary loss")
    p.add_argument("--cdss-loss-weight", type=float, default=0.0125, help="external coefficient for controlled synthesis; effective strength must be judged from gradient diagnostics")
    p.add_argument("--cdss-temperature", type=float, default=0.1)
    p.add_argument("--cdss-core-topk", type=int, default=64)
    p.add_argument("--cdss-boundary-pos-topk", type=int, default=64)
    p.add_argument("--cdss-hard-neg-topk", type=int, default=64)
    p.add_argument("--cdss-under-recall", type=float, default=0.40)
    p.add_argument("--cdss-moderate-recall", type=float, default=0.75)
    p.add_argument("--cdss-boundary-precision", type=float, default=0.60)
    p.add_argument("--cdss-stable-recall", type=float, default=0.95)
    p.add_argument("--cdss-enabled-states", default="under,moderate,boundary,stable")
    p.add_argument("--cdss-aggregation", choices=["query_mean", "capped_state_balance"], default="query_mean")
    p.add_argument("--cdss-balance-min", type=float, default=0.5)
    p.add_argument("--cdss-balance-max", type=float, default=2.0)
    p.add_argument("--cdss-distance-mode", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--cdss-separation-mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--cdss-absolute-margin", type=float, default=0.05)
    p.add_argument("--cdss-absolute-separation-cap", type=float, default=5.0)
    p.add_argument("--cdss-relative-margin", type=float, default=0.10)
    p.add_argument("--loss-gradient-diagnostics-every", type=int, default=0, help="run expensive per-objective gradient diagnostics every N source steps; 0 disables")
    p.add_argument("--core-prototype-loss-weight", type=float, default=0.02)
    p.add_argument("--core-prototype-margin", type=float, default=0.05)
    p.add_argument("--core-prototype-topk", type=int, default=32)
    p.add_argument("--core-prototype-hard-neg-topk", type=int, default=32)
    p.add_argument("--size-loss-weight", type=float, default=0.01)
    p.add_argument("--size-over-tolerance", type=float, default=0.15)
    p.add_argument("--enable-ranking-preservation-loss", action=argparse.BooleanOptionalAction, default=False, help="source-only V5.5 loss that discourages refined ranking drift away from raw/cache embedding")
    p.add_argument("--ranking-preservation-weight", type=float, default=0.0)
    p.add_argument("--ranking-preservation-mode", choices=["cosine", "l2", "cosine_l2"], default="cosine")
    p.add_argument("--enable-density-ratio-guidance", action="store_true", help="train and save DiffGDA-style target-unlabeled density-ratio reverse guidance; no target labels/gt")
    p.add_argument("--ratio-guidance-weight", type=float, default=0.01, help="reverse denoise coefficient for grad_z log q_T/p_S saved in checkpoints")
    p.add_argument("--ratio-guidance-clip", type=float, default=5.0, help="per-node L2 clip for density-ratio reverse guidance gradient")
    p.add_argument("--ratio-guidance-lr", type=float, default=1e-3)
    p.add_argument("--ratio-guidance-hidden-dim", type=int, default=256)
    p.add_argument("--ratio-guidance-time-emb-dim", type=int, default=64)
    p.add_argument("--ratio-guidance-max-nodes", type=int, default=4096)
    p.add_argument("--ratio-guidance-sampling-seed", type=int, default=None, help="optional stateless target/guidance sampling seed shared across model seeds")
    p.add_argument("--target-guidance-schedule", choices=["legacy_double_step", "round_robin"], default="legacy_double_step", help="target graph schedule for unlabeled guidance; legacy preserves historical runs")
    p.add_argument("--grad-clip-norm", type=float, default=0.0, help="optional main-model gradient clipping; 0 disables")
    p.add_argument("--enable-target-alignment-loss", action="store_true", help="optional MMD/CORAL auxiliary proxy; kept separate from true density-ratio reverse guidance")
    p.add_argument("--alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--alignment-max-nodes", type=int, default=4096)
    p.add_argument("--alignment-coral-weight", type=float, default=0.1)
    p.add_argument("--runtime-jsonl", default=None)
    p.add_argument(
        "--stage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record synchronized per-step method-stage timings for the paper efficiency audit",
    )
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--save-initial-checkpoint", action="store_true", help="save epoch_0000.pt before any optimizer step for target-best/epoch-0 diagnostics")
    p.add_argument("--save-epoch-dir", default=None)
    p.add_argument("--seed", type=int, default=None, help="fixed random seed for joint source training")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--deterministic-strict", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _parse_graphs(s: str) -> list[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise SystemExit("--source-graphs is empty")
    return out


def _parse_repeat_weights(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad --source-repeat-weights item: {part!r}; expected graph=int")
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            iv = int(v)
        except ValueError as exc:
            raise SystemExit(f"bad repeat weight for {k!r}: {v!r}") from exc
        if iv <= 0:
            raise SystemExit(f"repeat weight for {k!r} must be positive")
        out[k] = iv
    return out


def _checkpoint_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_family": "V5-SeniorExact-Joint",
        "input_dim": int(args.dims),
        "unified_input_dim": int(args.dims),
        "hidden_dim": int(args.hidden_dim),
        "hops": int(args.hops),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "dropout": float(args.dropout),
        "attention_dropout": float(args.attention_dropout),
        "readout": str(args.readout),
        "diffusion_T": int(args.diffusion_T),
        "senior_cache_mode": "required",
        "feature_projection": {"mode": "identity", "unified_input_dim": int(args.dims), "raw_feature_dim": int(args.dims)},
        "uses_senior_legacy_loss": True,
        "uses_query_boundary_loss": bool(args.enable_prefix_loss),
        "uses_prefix_boundary_loss": bool(args.enable_prefix_loss),
        "uses_size_loss": bool(args.enable_size_loss),
        "uses_score_aware_mining": False,
        "uses_learned_target_density_ratio": bool(args.enable_density_ratio_guidance),
        "joint_source_training": True,
        "source_graphs": _parse_graphs(args.source_graphs),
        "source_weighting": str(args.source_weighting),
        "enable_link_loss": bool(args.enable_link_loss),
        "max_link_loss_nodes": int(args.max_link_loss_nodes),
        "senior_contrastive_weight": float(args.senior_contrastive_weight),
        "diffusion_weight": float(args.diffusion_weight),
        "gamma": float(args.gamma),
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_updates": int(args.warmup_updates),
        "tot_updates": int(args.tot_updates),
        "end_lr": float(args.end_lr),
        "lr_power": float(args.lr_power),
        "source_repeat_weights": str(args.source_repeat_weights),
        "target_graphs_for_unlabeled_alignment": _parse_graphs(args.target_graphs) if str(args.target_graphs).strip() else [],
        "enable_prefix_loss": bool(args.enable_prefix_loss),
        "prefix_loss_weight": float(args.prefix_loss_weight),
        "prefix_on": str(args.prefix_on),
        "prefix_fp_topk": int(getattr(args, "prefix_fp_topk", 64)),
        "prefix_precision_loss_weight": float(getattr(args, "prefix_precision_loss_weight", 0.0)),
        "prefix_state_aware_loss": bool(getattr(args, "prefix_state_aware_loss", False)),
        "prefix_stratified_query_sampling": bool(getattr(args, "prefix_stratified_query_sampling", False)),
        "prefix_state_candidate_limit": int(getattr(args, "prefix_state_candidate_limit", 128)),
        "enable_structural_boundary_loss": bool(getattr(args, "enable_structural_boundary_loss", False)),
        "structural_boundary_weight": float(getattr(args, "structural_boundary_weight", 0.0)),
        "structural_boundary_hop": int(getattr(args, "structural_boundary_hop", 2)),
        "structural_neg_topk": int(getattr(args, "structural_neg_topk", 64)),
        "enable_size_loss": bool(args.enable_size_loss),
        "enable_ranking_preservation_loss": bool(getattr(args, "enable_ranking_preservation_loss", False)),
        "ranking_preservation_weight": float(getattr(args, "ranking_preservation_weight", 0.0)),
        "ranking_preservation_mode": str(getattr(args, "ranking_preservation_mode", "cosine")),
        "size_loss_weight": float(args.size_loss_weight),
        "enable_density_ratio_guidance": bool(args.enable_density_ratio_guidance),
        "ratio_guidance_weight": float(getattr(args, "ratio_guidance_weight", 0.05)),
        "ratio_guidance_clip": float(getattr(args, "ratio_guidance_clip", 5.0)),
        "ratio_guidance_hidden_dim": int(getattr(args, "ratio_guidance_hidden_dim", 256)),
        "ratio_guidance_time_emb_dim": int(getattr(args, "ratio_guidance_time_emb_dim", 64)),
        "ratio_guidance_sampling_seed": None if getattr(args, "ratio_guidance_sampling_seed", None) is None else int(args.ratio_guidance_sampling_seed),
        "target_guidance_schedule": str(getattr(args, "target_guidance_schedule", "legacy_double_step")),
        "grad_clip_norm": float(getattr(args, "grad_clip_norm", 0.0)),
        "enable_state_conditioned_community_pic_loss": bool(getattr(args, "enable_state_conditioned_community_pic_loss", False)),
        "community_pic_loss_weight": float(getattr(args, "community_pic_loss_weight", 0.0)),
        "community_pic_temperature": float(getattr(args, "community_pic_temperature", 0.1)),
        "community_pic_topk_core": int(getattr(args, "community_pic_topk_core", 64)),
        "community_pic_hard_neg_topk": int(getattr(args, "community_pic_hard_neg_topk", 64)),
        "community_pic_state_top_fraction": float(getattr(args, "community_pic_state_top_fraction", 0.10)),
        "enable_controlled_error_state_synthesis_loss": bool(getattr(args, "enable_controlled_error_state_synthesis_loss", False)),
        "cdss_loss_weight": float(getattr(args, "cdss_loss_weight", 0.0)),
        "cdss_temperature": float(getattr(args, "cdss_temperature", 0.1)),
        "cdss_core_topk": int(getattr(args, "cdss_core_topk", 64)),
        "cdss_boundary_pos_topk": int(getattr(args, "cdss_boundary_pos_topk", 64)),
        "cdss_hard_neg_topk": int(getattr(args, "cdss_hard_neg_topk", 64)),
        "cdss_under_recall": float(getattr(args, "cdss_under_recall", 0.40)),
        "cdss_moderate_recall": float(getattr(args, "cdss_moderate_recall", 0.75)),
        "cdss_boundary_precision": float(getattr(args, "cdss_boundary_precision", 0.60)),
        "cdss_stable_recall": float(getattr(args, "cdss_stable_recall", 0.95)),
        "cdss_enabled_states": str(getattr(args, "cdss_enabled_states", "under,moderate,boundary,stable")),
        "cdss_aggregation": str(getattr(args, "cdss_aggregation", "query_mean")),
        "cdss_balance_min": float(getattr(args, "cdss_balance_min", 0.5)),
        "cdss_balance_max": float(getattr(args, "cdss_balance_max", 2.0)),
        "cdss_distance_mode": str(getattr(args, "cdss_distance_mode", "euclidean")),
        "cdss_separation_mode": str(getattr(args, "cdss_separation_mode", "absolute")),
        "cdss_absolute_margin": float(getattr(args, "cdss_absolute_margin", 0.05)),
        "cdss_absolute_separation_cap": float(getattr(args, "cdss_absolute_separation_cap", 5.0)),
        "cdss_relative_margin": float(getattr(args, "cdss_relative_margin", 0.10)),
        "loss_gradient_diagnostics_every": int(getattr(args, "loss_gradient_diagnostics_every", 0)),
        "deterministic": bool(getattr(args, "deterministic", False)),
        "deterministic_strict": bool(getattr(args, "deterministic_strict", False)),
        "enable_target_alignment_loss": bool(getattr(args, "enable_target_alignment_loss", False)),
        "alignment_loss_weight": float(args.alignment_loss_weight),
    }


def _save_ckpt(path: Path, *, model: PretrainModelV5, epoch: int, metric: str, value: float, args: argparse.Namespace, extra_state: Mapping[str, object] | None = None, ratio_guidance_pair: DensityRatioGuidancePair | None = None) -> Path:
    extra = {
        "source_graph_name": ",".join(_parse_graphs(args.source_graphs)),
        "source_graphs": _parse_graphs(args.source_graphs),
        "senior_exact": True,
        "joint_source_training": True,
        "epoch_checkpoint": path.name.startswith("epoch_"),
    }
    if ratio_guidance_pair is not None:
        extra["density_ratio_guidance"] = ratio_guidance_pair.export_state(
            guidance_weight=float(getattr(args, "ratio_guidance_weight", 0.05)),
            guidance_clip=float(getattr(args, "ratio_guidance_clip", 5.0)),
        )
    if extra_state:
        extra.update(dict(extra_state))
    return save_source_checkpoint(
        path,
        model_state_dict=model.state_dict(),
        epoch=int(epoch),
        source_selection_metric=metric,
        source_selection_value=float(value),
        config=_checkpoint_config(args),
        extra_state=extra,
    )


def _dense_adj_and_minus(adj: torch.Tensor, *, max_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(adj.shape[0])
    if n > int(max_nodes):
        raise V5ProtocolError(f"dense link loss requested for N={n}, above max_link_loss_nodes={max_nodes}")
    dense = adj.coalesce().to_dense().float() if adj.is_sparse else adj.float()
    dense = (dense > 0).float().to(device)
    minus = (torch.ones_like(dense) - dense).to(device)
    return dense, minus


def _target_token_batch(
    tokens_cpu: torch.Tensor,
    *,
    max_nodes: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    n = int(tokens_cpu.shape[0])
    k = min(int(max_nodes), n)
    if n <= k:
        return tokens_cpu.to(device)
    idx = torch.randperm(n, generator=generator)[:k]
    return tokens_cpu.index_select(0, idx).to(device)


def _cdss_config_from_args(args: argparse.Namespace) -> ControlledStateSynthesisConfig:
    enabled_states = tuple(part.strip() for part in str(args.cdss_enabled_states).split(",") if part.strip())
    return ControlledStateSynthesisConfig(
        temperature=float(args.cdss_temperature),
        core_topk=int(args.cdss_core_topk),
        boundary_pos_topk=int(args.cdss_boundary_pos_topk),
        hard_neg_topk=int(args.cdss_hard_neg_topk),
        under_recall=float(args.cdss_under_recall),
        moderate_recall=float(args.cdss_moderate_recall),
        boundary_precision=float(args.cdss_boundary_precision),
        stable_recall=float(args.cdss_stable_recall),
        enabled_states=enabled_states,
        aggregation=str(args.cdss_aggregation),
        balance_min=float(args.cdss_balance_min),
        balance_max=float(args.cdss_balance_max),
        distance_mode=str(args.cdss_distance_mode),
        separation_mode=str(args.cdss_separation_mode),
        absolute_margin=float(args.cdss_absolute_margin),
        absolute_separation_cap=float(args.cdss_absolute_separation_cap),
        relative_margin=float(args.cdss_relative_margin),
    )


def main() -> None:
    args = parse_args()
    set_global_seed(
        args.seed,
        deterministic=bool(args.deterministic),
        deterministic_strict=bool(args.deterministic_strict),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_jsonl = args.runtime_jsonl or str(out_dir / "runtime_profile.jsonl")
    source_graphs = _parse_graphs(args.source_graphs)
    target_graphs = _parse_graphs(args.target_graphs) if str(args.target_graphs).strip() else []
    explicit_repeats = _parse_repeat_weights(args.source_repeat_weights)
    if args.source_weighting == "explicit":
        missing = [g for g in source_graphs if g not in explicit_repeats]
        extra = [g for g in explicit_repeats if g not in source_graphs]
        if missing:
            raise SystemExit(f"--source-weighting explicit requires repeat weights for all sources; missing={missing}, extra={extra}")
    profile_event(runtime_jsonl, "senior_exact_joint_train_start", source_graphs=source_graphs, device=args.device)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    ratio_seed = args.ratio_guidance_sampling_seed
    if ratio_seed is None:
        ratio_seed = args.seed
    ratio_cpu_generator = make_generator(ratio_seed, "cpu")
    ratio_device_generator = make_generator(ratio_seed, device)

    loaded = []
    for gname in source_graphs:
        graph = load_senior_graph_data(
            args.dataset_root,
            gname,
            split="source_train",
            purpose="senior_exact_joint_source_training",
            dims=int(args.dims),
            hops=int(args.hops),
            local_topk=int(args.local_topk),
            pos_count=int(args.pos_count),
            neg_count=int(args.neg_count),
            require_labels=True,
            load_queries=True,
            prefer_preprocessed=True,
            senior_cache_mode="required",
            unified_input_dim=int(args.dims),
            feature_projection_mode="identity",
            runtime_jsonl=runtime_jsonl,
        )
        if graph.pos_samples is None or graph.neg_samples is None:
            raise V5ProtocolError(f"SeniorExact joint training requires pos/neg samples in strict cache: {gname}")
        if int(graph.global_token_seqs.shape[-1]) != int(args.dims):
            raise V5ProtocolError(f"strict senior token dimension mismatch for {gname}")
        n = int(graph.global_token_seqs.shape[0])
        if n > int(args.max_full_graph_nodes):
            raise V5ProtocolError(f"SeniorExact full-graph trainer refuses {gname} N={n} > max_full_graph_nodes={args.max_full_graph_nodes}")
        if args.source_weighting == "uniform":
            repeats = 1
        elif args.source_weighting == "explicit":
            repeats = int(explicit_repeats[gname])
        else:
            repeats = max(1, int(math.ceil(n / max(1, int(args.batch_size)))))
        query_sets = None
        gt_sets = None
        needs_source_query_gt = any([
            bool(args.enable_prefix_loss),
            bool(getattr(args, "enable_state_conditioned_boundary_loss", False)),
            bool(getattr(args, "enable_core_prototype_consistency_loss", False)),
            bool(getattr(args, "enable_state_conditioned_community_pic_loss", False)),
            bool(getattr(args, "enable_controlled_error_state_synthesis_loss", False)),
        ])
        if needs_source_query_gt:
            if graph.query_batch is None:
                raise V5ProtocolError(f"source query-supervised loss requires source query file: {gname}")
            if not graph.files.gt_path_present_but_not_read:
                raise V5ProtocolError(f"source query-supervised loss requires source gt file: {gname}")
            query_sets = graph.query_batch.query_node_sets
            gt_sets = load_source_gt_sets(graph.files.gt_path_present_but_not_read)
            if len(query_sets) != len(gt_sets):
                raise V5ProtocolError(f"source query/gt count mismatch for {gname}: {len(query_sets)} vs {len(gt_sets)}")
        entry = {
            "name": gname,
            "graph": graph,
            "tokens": graph.global_token_seqs.float().to(device),
            "pos": graph.pos_samples.long().to(device),
            "neg": graph.neg_samples.long().to(device),
            "query_sets": query_sets,
            "gt_sets": gt_sets,
            "n": n,
            "repeats": repeats,
            "adj_dense": None,
            "minus_dense": None,
        }
        if bool(args.enable_link_loss):
            entry["adj_dense"], entry["minus_dense"] = _dense_adj_and_minus(graph.adjacency, max_nodes=int(args.max_link_loss_nodes), device=device)
        loaded.append(entry)

    target_loaded = []
    if bool(args.enable_density_ratio_guidance):
        if not target_graphs:
            raise V5ProtocolError("--enable-density-ratio-guidance requires --target-graphs")
        for tname in target_graphs:
            tg = load_senior_graph_data(
                args.dataset_root,
                tname,
                split="unlabeled_target_alignment",
                purpose="density_ratio_guidance_unlabeled_target",
                dims=int(args.dims),
                hops=int(args.hops),
                local_topk=int(args.local_topk),
                pos_count=int(args.pos_count),
                neg_count=int(args.neg_count),
                require_labels=False,
                load_queries=False,
                prefer_preprocessed=True,
                senior_cache_mode="required",
                unified_input_dim=int(args.dims),
                feature_projection_mode="identity",
                runtime_jsonl=runtime_jsonl,
            )
            target_loaded.append({"name": tname, "tokens_cpu": tg.global_token_seqs.float(), "n": int(tg.global_token_seqs.shape[0])})

    input_status = {
        "protocol": "OFA-CS-V5-SeniorExact-joint-source-training",
        "source_graphs": source_graphs,
        "source_weighting": str(args.source_weighting),
        "source_repeat_weights": dict(explicit_repeats),
        "source_steps_per_epoch": {x["name"]: int(x["repeats"]) for x in loaded},
        "source_num_nodes": {x["name"]: int(x["n"]) for x in loaded},
        "uses_senior_cache": True,
        "uses_query_boundary_loss": bool(args.enable_prefix_loss),
        "uses_prefix_boundary_loss": bool(args.enable_prefix_loss),
        "uses_size_loss": bool(args.enable_size_loss),
        "uses_score_aware_mining": False,
        "uses_learned_target_density_ratio": bool(args.enable_density_ratio_guidance),
        "target_graphs_for_unlabeled_alignment": target_graphs,
        "source_labels_used": True,
        "target_unlabeled_features_used": bool(args.enable_density_ratio_guidance or args.enable_target_alignment_loss),
        "target_labels_used": False,
        "target_gt_used": False,
        "labels_used": True,
        "unseen_training_used": bool(
            args.enable_density_ratio_guidance or args.enable_target_alignment_loss
        ),
        "strict_one_for_all_source_training": not bool(
            args.enable_density_ratio_guidance or args.enable_target_alignment_loss
        ),
        "enable_state_conditioned_community_pic_loss": bool(args.enable_state_conditioned_community_pic_loss),
        "community_pic_loss_weight": float(args.community_pic_loss_weight),
        "community_pic_state_top_fraction": float(args.community_pic_state_top_fraction),
        "enable_controlled_error_state_synthesis_loss": bool(args.enable_controlled_error_state_synthesis_loss),
        "cdss_loss_weight": float(args.cdss_loss_weight),
        "cdss_source_supervised_only": True,
        "cdss_target_labels_used": False,
        "cdss_target_gt_used": False,
        "ratio_guidance_sampling_seed": None if args.ratio_guidance_sampling_seed is None else int(args.ratio_guidance_sampling_seed),
        "target_guidance_schedule": str(args.target_guidance_schedule),
        "grad_clip_norm": float(args.grad_clip_norm),
        "deterministic": bool(args.deterministic),
        "deterministic_strict": bool(args.deterministic_strict),
        "save_every_epoch": bool(args.save_every_epoch),
        "enable_link_loss": bool(args.enable_link_loss),
    }
    (out_dir / "source_training_input_status.json").write_text(json.dumps(input_status, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")

    cfg = SimpleNamespace(
        hidden_dim=int(args.hidden_dim),
        hops=int(args.hops),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
        readout=str(args.readout),
        diffusion_T=int(args.diffusion_T),
        condition_dim=16,
        legacy_link_loss_max_nodes=int(args.max_link_loss_nodes),
    )
    model = PretrainModelV5(input_dim=int(args.dims), config=cfg).to(device)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    cdss_reference_report = {}
    cdss_config_static = None
    if bool(getattr(args, "enable_controlled_error_state_synthesis_loss", False)):
        cdss_config_static = _cdss_config_from_args(args)
        # Freeze one deterministic pre-optimization source representation and
        # precompute all node selections once. This prevents moving hard-mining
        # and avoids repeating O(Q*N*D) reference scoring at every train step.
        model.eval()
        with torch.no_grad():
            for entry in loaded:
                reference_forward = model(
                    None,
                    entry["tokens"],
                    compute_diffusion_loss=False,
                    disable_diffusion_for_embedding=True,
                )
                reference = reference_forward.raw_embedding.detach().clone()
                entry["cdss_reference_embedding"] = reference
                selection_plan = build_controlled_state_selection_plan(
                    reference,
                    entry["query_sets"],
                    entry["gt_sets"],
                    config=cdss_config_static,
                    graph_name=str(entry["name"]),
                )
                entry["cdss_selection_plan"] = selection_plan
                cdss_reference_report[entry["name"]] = {
                    "num_nodes": int(reference.shape[0]),
                    "dim": int(reference.shape[1]),
                    "mean_l2_norm": float(reference.norm(dim=-1).mean().cpu().item()),
                    "requires_grad": bool(reference.requires_grad),
                    "reference_stage": "pre_optimization_raw_embedding",
                    "selection_plan_valid_queries": int(selection_plan["valid_query_count"]),
                    "selection_plan_skipped_queries": int(selection_plan["skipped_query_count"]),
                    "selection_plan_mean_gt_size": float(selection_plan.get("mean_gt_size", 0.0)),
                    "selection_plan_state_summary": selection_plan.get("state_plan_summary", {}),
                    "boundary_state_semantics": selection_plan.get("boundary_state_semantics", ""),
                    "selection_plan_precomputed": True,
                }
        model.train()
    scheduler = None
    if args.lr_scheduler == "polynomial":
        scheduler = PolynomialDecayLR(
            opt,
            warmup_updates=int(args.warmup_updates),
            tot_updates=int(args.tot_updates),
            lr=float(args.lr),
            end_lr=float(args.end_lr),
            power=float(args.lr_power),
        )

    ratio_guidance_pair = None
    ratio_guidance_optimizer = None
    if bool(args.enable_density_ratio_guidance):
        ratio_guidance_pair = DensityRatioGuidancePair(
            int(args.hidden_dim),
            hidden_dim=int(getattr(args, "ratio_guidance_hidden_dim", 256)),
            time_emb_dim=int(getattr(args, "ratio_guidance_time_emb_dim", 64)),
        ).to(device)
        ratio_guidance_optimizer = torch.optim.AdamW(
            ratio_guidance_pair.parameters(),
            lr=float(args.ratio_guidance_lr),
            weight_decay=float(args.weight_decay),
        )

    events_path = out_dir / "senior_exact_joint_training_events.jsonl"
    best_loss = None
    best_path = out_dir / "best_senior_exact_joint.pt"
    last_path = out_dir / "last_senior_exact_joint.pt"
    epoch_dir = Path(args.save_epoch_dir) if args.save_epoch_dir else out_dir / "epoch_checkpoints"
    if args.save_every_epoch:
        epoch_dir.mkdir(parents=True, exist_ok=True)
    if args.save_initial_checkpoint:
        epoch_dir.mkdir(parents=True, exist_ok=True)
        _save_ckpt(
            epoch_dir / "epoch_0000.pt",
            model=model,
            epoch=0,
            metric="initial_untrained",
            value=float("nan"),
            args=args,
            extra_state={"selection_note": "initial checkpoint before any optimizer step"},
            ratio_guidance_pair=ratio_guidance_pair,
        )

    scpc_aggregate = {
        "steps": 0,
        "valid_queries": 0,
        "intra_sum": 0.0,
        "inter_sum": 0.0,
        "ratio_sum": 0.0,
        "separation_sum": 0.0,
        "separation_active_sum": 0,
        "core_nodes_sum": 0.0,
        "states": {"under": 0, "moderate": 0, "boundary": 0, "stable": 0},
    }
    cdss_aggregate = {
        "steps": 0,
        "valid_queries": 0,
        "skipped_queries": 0,
        "positive_distance_sum": 0.0,
        "negative_distance_sum": 0.0,
        "ratio_sum": 0.0,
        "separation_sum": 0.0,
        "separation_active_sum": 0.0,
        "core_nodes_sum": 0.0,
        "positive_nodes_sum": 0.0,
        "negative_nodes_sum": 0.0,
        "states": {"under": 0, "moderate": 0, "boundary": 0, "stable": 0},
        "per_graph": {},
    }
    gradient_diagnostics_events = []
    global_step = 0
    for epoch in range(1, int(args.epochs)+1):
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        per_graph_loss = {x["name"]: [] for x in loaded}
        for entry in loaded:
            for _ in range(int(entry["repeats"])):
                stage_timing_enabled = bool(getattr(args, "stage_timing", False))
                stage_times: dict[str, float] = {}
                step_t0 = _timing_mark(stage_timing_enabled, device)
                model.train()
                opt.zero_grad(set_to_none=True)
                forward_t0 = _timing_mark(stage_timing_enabled, device)
                forward = model(
                    None,
                    entry["tokens"],
                    compute_diffusion_loss=True,
                    disable_diffusion_for_embedding=not bool(args.enable_prefix_loss),
                )
                forward_t1 = _timing_mark(stage_timing_enabled, device)
                stage_times["model_forward_and_diffusion_seconds"] = float(forward_t1 - forward_t0)
                established_t0 = forward_t1
                senior_loss = model.legacy_transformer_loss(
                    forward.node_tensor,
                    forward.neighbor_tensor,
                    entry["pos"],
                    entry["neg"],
                    adj_=entry["adj_dense"],
                    minus_adj=entry["minus_dense"],
                    enable_link_loss=bool(args.enable_link_loss),
                    gamma=float(args.gamma),
                )
                senior_component = float(args.senior_contrastive_weight) * senior_loss
                total = senior_component
                diffusion_value = None
                weighted_diffusion_component = None
                if forward.diffusion_loss is not None and float(args.diffusion_weight) != 0.0:
                    weighted_diffusion_component = float(args.diffusion_weight) * forward.diffusion_loss
                    total = total + weighted_diffusion_component
                    diffusion_value = float(forward.diffusion_loss.detach().cpu().item())

                prefix_value = None
                prefix_report = None
                if bool(args.enable_prefix_loss):
                    p_cfg = PrefixBoundaryLossConfig(
                        margin=float(args.prefix_margin),
                        query_limit=int(args.prefix_query_limit),
                        neg_topk=int(args.prefix_neg_topk),
                        pos_bottomk=int(args.prefix_pos_bottomk),
                        fp_topk=int(getattr(args, "prefix_fp_topk", args.prefix_neg_topk)),
                        precision_loss_weight=float(getattr(args, "prefix_precision_loss_weight", 0.0)),
                        adaptive_precision=bool(getattr(args, "prefix_adaptive_precision", False)),
                        precision_over_ratio=float(getattr(args, "prefix_precision_over_ratio", 1.25)),
                        pairwise_boundary_loss=bool(getattr(args, "prefix_pairwise_boundary_loss", False)),
                        fn_loss_weight=float(getattr(args, "prefix_fn_loss_weight", 1.0)),
                        state_aware_loss=bool(getattr(args, "prefix_state_aware_loss", False)),
                        stratified_query_sampling=bool(getattr(args, "prefix_stratified_query_sampling", False)),
                        state_candidate_limit=int(getattr(args, "prefix_state_candidate_limit", 128)),
                        state_over_ratio=float(getattr(args, "prefix_state_over_ratio", 1.40)),
                        state_under_fn_ratio=float(getattr(args, "prefix_state_under_fn_ratio", 0.20)),
                        state_mixed_gap=float(getattr(args, "prefix_state_mixed_gap", 0.02)),
                        state_precision_mult=float(getattr(args, "prefix_state_precision_mult", 1.50)),
                        state_size_mult=float(getattr(args, "prefix_state_size_mult", 1.50)),
                        state_fn_mult=float(getattr(args, "prefix_state_fn_mult", 1.50)),
                        state_pairwise_mult=float(getattr(args, "prefix_state_pairwise_mult", 1.00)),
                        state_hard_positive_mult=float(getattr(args, "prefix_state_hard_positive_mult", 2.00)),
                        structural_boundary_loss=bool(getattr(args, "enable_structural_boundary_loss", False)),
                        structural_boundary_weight=float(getattr(args, "structural_boundary_weight", 0.0)),
                        structural_boundary_hop=int(getattr(args, "structural_boundary_hop", 2)),
                        structural_neg_topk=int(getattr(args, "structural_neg_topk", 64)),
                        size_loss_weight=float(args.size_loss_weight) if bool(args.enable_size_loss) else 0.0,
                        size_over_tolerance=float(args.size_over_tolerance),
                    )
                    views = []
                    if args.prefix_on in {"raw", "both"}:
                        views.append(forward.raw_embedding)
                    if args.prefix_on in {"refined", "both"}:
                        views.append(forward.refined_embedding)
                    p_losses = []
                    p_reports = []
                    for emb_view in views:
                        p_loss, p_report = compute_prefix_boundary_loss(
                            emb_view,
                            entry["query_sets"],
                            entry["gt_sets"],
                            config=p_cfg,
                            edge_index=entry["graph"].edge_index if bool(getattr(args, "enable_structural_boundary_loss", False)) else None,
                        )
                        p_losses.append(p_loss)
                        p_reports.append(p_report.to_dict())
                    if p_losses:
                        prefix_loss = torch.stack(p_losses).mean()
                        total = total + float(args.prefix_loss_weight) * prefix_loss
                        prefix_value = float(prefix_loss.detach().cpu().item())
                        prefix_report = p_reports

                state_boundary_value = None
                state_boundary_report = None
                if bool(getattr(args, "enable_state_conditioned_boundary_loss", False)):
                    sb_loss, sb_report = compute_state_conditioned_boundary_loss(
                        forward.refined_embedding, entry["query_sets"], entry["gt_sets"],
                        margin=float(args.state_boundary_margin),
                        topk_pos=int(args.state_boundary_topk_pos),
                        topk_neg=int(args.state_boundary_topk_neg),
                        under_recall_threshold=float(args.state_boundary_under_recall_threshold),
                        over_precision_threshold=float(args.state_boundary_over_precision_threshold),
                    )
                    total = total + float(args.state_boundary_loss_weight) * sb_loss
                    state_boundary_value=float(sb_loss.detach().cpu().item())
                    state_boundary_report=sb_report.to_dict()
                scpc_value = None
                scpc_report = None
                weighted_scpc_component = None
                if bool(getattr(args, "enable_state_conditioned_community_pic_loss", False)):
                    scpc_loss, scpc_report = compute_state_conditioned_community_pic_loss(
                        forward.refined_embedding, entry["query_sets"], entry["gt_sets"],
                        temperature=float(args.community_pic_temperature),
                        topk_core=int(args.community_pic_topk_core),
                        hard_neg_topk=int(args.community_pic_hard_neg_topk),
                        state_top_fraction=float(args.community_pic_state_top_fraction),
                    )
                    weighted_scpc_component = float(args.community_pic_loss_weight) * scpc_loss
                    total = total + weighted_scpc_component
                    scpc_value = float(scpc_loss.detach().cpu().item())
                    scpc_aggregate["steps"] += 1
                    valid_q = int(scpc_report.get("valid", 0))
                    scpc_aggregate["valid_queries"] += valid_q
                    scpc_aggregate["intra_sum"] += float(scpc_report.get("intra", 0.0)) * valid_q
                    scpc_aggregate["inter_sum"] += float(scpc_report.get("inter", 0.0)) * valid_q
                    scpc_aggregate["ratio_sum"] += float(scpc_report.get("mean_ratio_term", 0.0)) * valid_q
                    scpc_aggregate["separation_sum"] += float(scpc_report.get("mean_separation_term", 0.0)) * valid_q
                    scpc_aggregate["separation_active_sum"] += float(scpc_report.get("separation_active_rate", 0.0)) * valid_q
                    scpc_aggregate["core_nodes_sum"] += float(scpc_report.get("mean_core_count", 0.0)) * valid_q
                    for state_name, count in dict(scpc_report.get("states", {})).items():
                        if state_name in scpc_aggregate["states"]:
                            scpc_aggregate["states"][state_name] += int(count)

                cdss_value = None
                cdss_report = None
                weighted_cdss_component = None
                established_t1 = _timing_mark(stage_timing_enabled, device)
                stage_times["established_losses_before_cdss_seconds"] = float(established_t1 - established_t0)
                cdss_t0 = established_t1
                if bool(getattr(args, "enable_controlled_error_state_synthesis_loss", False)):
                    cdss_config = cdss_config_static
                    if cdss_config is None:
                        raise V5ProtocolError("CDSS enabled but static config was not initialized")
                    reference_embedding = entry.get("cdss_reference_embedding")
                    if reference_embedding is None:
                        raise V5ProtocolError(f"missing frozen CDSS reference embedding for {entry['name']}")
                    cdss_loss, cdss_report_obj = compute_controlled_community_error_state_loss(
                        forward.refined_embedding,
                        reference_embedding,
                        entry["query_sets"],
                        entry["gt_sets"],
                        config=cdss_config,
                        schedule_step=int(global_step),
                        graph_name=str(entry["name"]),
                        selection_plan=entry.get("cdss_selection_plan"),
                    )
                    weighted_cdss_component = float(args.cdss_loss_weight) * cdss_loss
                    total = total + weighted_cdss_component
                    cdss_value = float(cdss_loss.detach().cpu().item())
                    cdss_report = cdss_report_obj.to_dict()
                    valid_cdss = int(cdss_report.get("valid_queries", 0))
                    cdss_aggregate["steps"] += 1
                    cdss_aggregate["valid_queries"] += valid_cdss
                    cdss_aggregate["skipped_queries"] += int(cdss_report.get("skipped_queries", 0))
                    cdss_aggregate["positive_distance_sum"] += float(cdss_report.get("mean_positive_distance", 0.0)) * valid_cdss
                    cdss_aggregate["negative_distance_sum"] += float(cdss_report.get("mean_negative_distance", 0.0)) * valid_cdss
                    cdss_aggregate["ratio_sum"] += float(cdss_report.get("mean_ratio", 0.0)) * valid_cdss
                    cdss_aggregate["separation_sum"] += float(cdss_report.get("mean_separation", 0.0)) * valid_cdss
                    cdss_aggregate["separation_active_sum"] += float(cdss_report.get("separation_active_rate", 0.0)) * valid_cdss
                    cdss_aggregate["core_nodes_sum"] += float(cdss_report.get("mean_core_nodes", 0.0)) * valid_cdss
                    cdss_aggregate["positive_nodes_sum"] += float(cdss_report.get("mean_selected_positive_nodes", 0.0)) * valid_cdss
                    cdss_aggregate["negative_nodes_sum"] += float(cdss_report.get("mean_selected_negative_nodes", 0.0)) * valid_cdss
                    graph_agg = cdss_aggregate["per_graph"].setdefault(
                        str(entry["name"]),
                        {"valid_queries": 0, "states": {"under": 0, "moderate": 0, "boundary": 0, "stable": 0}},
                    )
                    graph_agg["valid_queries"] += valid_cdss
                    for state_name, count in dict(cdss_report.get("state_counts", {})).items():
                        if state_name in cdss_aggregate["states"]:
                            cdss_aggregate["states"][state_name] += int(count)
                            graph_agg["states"][state_name] += int(count)
                cdss_t1 = _timing_mark(stage_timing_enabled, device)
                stage_times["cdss_geometry_seconds"] = float(cdss_t1 - cdss_t0)
                post_cdss_t0 = cdss_t1
                core_proto_value=None
                core_proto_report=None
                if bool(getattr(args, "enable_core_prototype_consistency_loss", False)):
                    cp_loss, cp_report = compute_core_prototype_consistency_loss(
                        forward.refined_embedding, entry["query_sets"], entry["gt_sets"],
                        margin=float(args.core_prototype_margin),
                        topk=int(args.core_prototype_topk),
                        hard_neg_topk=int(args.core_prototype_hard_neg_topk),
                    )
                    total = total + float(args.core_prototype_loss_weight) * cp_loss
                    core_proto_value=float(cp_loss.detach().cpu().item())
                    core_proto_report=cp_report

                preservation_value = None
                preservation_report = None
                if bool(getattr(args, "enable_ranking_preservation_loss", False)) and float(getattr(args, "ranking_preservation_weight", 0.0)) > 0.0:
                    raw_detached = forward.raw_embedding.detach()
                    refined = forward.refined_embedding
                    mode = str(getattr(args, "ranking_preservation_mode", "cosine"))
                    losses = []
                    if mode in {"cosine", "cosine_l2"}:
                        raw_n = F.normalize(raw_detached, p=2, dim=-1, eps=1e-12)
                        ref_n = F.normalize(refined, p=2, dim=-1, eps=1e-12)
                        losses.append((1.0 - (raw_n * ref_n).sum(dim=-1)).mean())
                    if mode in {"l2", "cosine_l2"}:
                        losses.append(F.mse_loss(refined, raw_detached))
                    if losses:
                        preservation_loss = torch.stack(losses).mean()
                        total = total + float(args.ranking_preservation_weight) * preservation_loss
                        preservation_value = float(preservation_loss.detach().cpu().item())
                        preservation_report = {
                            "mode": mode,
                            "weight": float(args.ranking_preservation_weight),
                            "source_only": True,
                            "unseen_training_used": False,
                        }

                post_cdss_t1 = _timing_mark(stage_timing_enabled, device)
                stage_times["other_auxiliary_losses_seconds"] = float(post_cdss_t1 - post_cdss_t0)

                alignment_value = None
                alignment_report = None
                ratio_guidance_value = None
                ratio_guidance_report = None
                target_forward = None
                target_entry = None
                ratio_t0 = post_cdss_t1
                if bool(args.enable_density_ratio_guidance) and target_loaded:
                    target_index = (global_step % len(target_loaded)) if str(args.target_guidance_schedule) == "round_robin" else ((global_step + epoch_step_count) % len(target_loaded))
                    target_entry = target_loaded[target_index]
                    tt = target_entry["tokens_cpu"]
                    requested_nodes = max(
                        int(args.ratio_guidance_max_nodes) if bool(args.enable_density_ratio_guidance) else 0,
                        int(args.alignment_max_nodes) if bool(args.enable_target_alignment_loss) else 0,
                    )
                    tt_batch = _target_token_batch(
                        tt,
                        max_nodes=max(1, requested_nodes),
                        device=device,
                        generator=ratio_cpu_generator,
                    )
                    target_forward = model(None, tt_batch, compute_diffusion_loss=False, disable_diffusion_for_embedding=False)
                    if ratio_guidance_pair is not None and ratio_guidance_optimizer is not None:
                        _rg_loss, _rg_report = ratio_guidance_pair.train_step(
                            source_node=forward.node_tensor,
                            source_neighbor=forward.neighbor_tensor,
                            target_node=target_forward.node_tensor,
                            target_neighbor=target_forward.neighbor_tensor,
                            refiner_node=model.diffusion_node,
                            refiner_neighbor=model.diffusion_neighbor,
                            optimizer=ratio_guidance_optimizer,
                            max_nodes=int(args.ratio_guidance_max_nodes),
                            generator=ratio_device_generator,
                        )
                        ratio_guidance_value = float(_rg_loss.detach().cpu().item())
                        ratio_guidance_report = {"target_graph": target_entry["name"], **_rg_report.to_dict()}
                if bool(getattr(args, "enable_target_alignment_loss", False)) and target_loaded:
                    if target_forward is None:
                        target_index = (global_step % len(target_loaded)) if str(args.target_guidance_schedule) == "round_robin" else ((global_step + epoch_step_count) % len(target_loaded))
                        target_entry = target_loaded[target_index]
                        tt = target_entry["tokens_cpu"]
                        tt_batch = _target_token_batch(
                            tt,
                            max_nodes=int(args.alignment_max_nodes),
                            device=device,
                            generator=ratio_cpu_generator,
                        )
                        target_forward = model(None, tt_batch, compute_diffusion_loss=False, disable_diffusion_for_embedding=False)
                    align_loss, align_report = compute_source_target_alignment_loss(
                        forward.refined_embedding,
                        target_forward.refined_embedding,
                        max_nodes=int(args.alignment_max_nodes),
                        coral_weight=float(args.alignment_coral_weight),
                    )
                    total = total + float(args.alignment_loss_weight) * align_loss
                    alignment_value = float(align_loss.detach().cpu().item())
                    alignment_report = {"target_graph": target_entry["name"] if target_entry else None, **align_report.to_dict()}
                ratio_t1 = _timing_mark(stage_timing_enabled, device)
                stage_times["target_ratio_and_alignment_seconds"] = float(ratio_t1 - ratio_t0)
                gradient_diagnostics = None
                grad_every = int(getattr(args, "loss_gradient_diagnostics_every", 0))
                if grad_every > 0 and (int(global_step) % grad_every == 0):
                    # The diagnostic base is the complete established objective
                    # excluding diffusion, SCPC and CDSS, which are reported as
                    # separate weighted components below.
                    base_objective_for_diag = total
                    for excluded_component in (
                        weighted_diffusion_component,
                        weighted_scpc_component,
                        weighted_cdss_component,
                    ):
                        if excluded_component is not None:
                            base_objective_for_diag = base_objective_for_diag - excluded_component
                    gradient_diagnostics = compute_loss_gradient_diagnostics(
                        model=model,
                        base_loss=base_objective_for_diag,
                        diffusion_loss=weighted_diffusion_component,
                        scpc_loss=weighted_scpc_component,
                        cdss_loss=weighted_cdss_component,
                    )
                    gradient_diagnostics_events.append({
                        "epoch": int(epoch),
                        "graph": str(entry["name"]),
                        "global_step_before_update": int(global_step),
                        **gradient_diagnostics,
                    })
                backward_t0 = _timing_mark(stage_timing_enabled, device)
                total.backward()
                grad_norm_value = None
                if float(args.grad_clip_norm) > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
                    grad_norm_value = float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                opt.step()
                backward_t1 = _timing_mark(stage_timing_enabled, device)
                stage_times["backward_and_optimizer_seconds"] = float(backward_t1 - backward_t0)
                stage_times["step_total_seconds"] = float(backward_t1 - step_t0)
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
                loss_value = float(total.detach().cpu().item())
                per_graph_loss[entry["name"]].append(loss_value)
                epoch_loss_sum += loss_value
                epoch_step_count += 1
                step_event = {
                    "epoch": int(epoch),
                    "graph": entry["name"],
                    "step_in_epoch": int(epoch_step_count),
                    "total_loss": loss_value,
                    "senior_legacy_loss": float(senior_loss.detach().cpu().item()),
                    "diffusion_loss": diffusion_value,
                    "prefix_loss": prefix_value,
                    "state_boundary_loss": state_boundary_value,
                    "state_boundary_report": state_boundary_report,
                    "community_pic_loss": scpc_value,
                    "community_pic_weighted_loss": None if weighted_scpc_component is None else float(weighted_scpc_component.detach().cpu().item()),
                    "community_pic_report": scpc_report,
                    "controlled_error_state_loss": cdss_value,
                    "controlled_error_state_weighted_loss": None if weighted_cdss_component is None else float(weighted_cdss_component.detach().cpu().item()),
                    "controlled_error_state_report": cdss_report,
                    "loss_gradient_diagnostics": gradient_diagnostics,
                    "core_prototype_loss": core_proto_value,
                    "core_prototype_report": core_proto_report,
                    "prefix_report": prefix_report,
                    "ranking_preservation_loss": preservation_value,
                    "ranking_preservation_report": preservation_report,
                    "alignment_loss": alignment_value,
                    "alignment_report": alignment_report,
                    "density_ratio_guidance_loss": ratio_guidance_value,
                    "density_ratio_guidance_report": ratio_guidance_report,
                    "enable_link_loss": bool(args.enable_link_loss),
                    "uses_query_boundary_loss": bool(args.enable_prefix_loss),
                    "uses_prefix_boundary_loss": bool(args.enable_prefix_loss),
                    "uses_size_loss": bool(args.enable_size_loss),
                    "uses_score_aware_mining": False,
                    "uses_learned_target_density_ratio": bool(args.enable_density_ratio_guidance),
                    "source_only": not bool(
                        args.enable_density_ratio_guidance
                        or args.enable_target_alignment_loss
                    ),
                    "unseen_training_used": bool(
                        args.enable_density_ratio_guidance
                        or args.enable_target_alignment_loss
                    ),
                    "full_graph_autograd_used": True,
                    "dense_link_loss_used": bool(args.enable_link_loss),
                    "num_nodes": int(entry["n"]),
                    "global_step": int(global_step),
                    "lr": float(opt.param_groups[0].get("lr", 0.0)),
                    "grad_norm_before_clip": grad_norm_value,
                    "grad_clip_norm": float(args.grad_clip_norm),
                    "target_guidance_schedule": str(args.target_guidance_schedule),
                    "paper_stage_timing_enabled": bool(stage_timing_enabled),
                    "paper_stage_timing": stage_times if stage_timing_enabled else None,
                }
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(step_event, ensure_ascii=False, sort_keys=True)+"\n")
        epoch_loss = float(epoch_loss_sum / max(1, epoch_step_count))
        event = {
            "epoch": int(epoch),
            "epoch_avg_loss": epoch_loss,
            "epoch_steps": int(epoch_step_count),
            "per_graph_avg_loss": {k: float(sum(v)/len(v)) if v else None for k, v in per_graph_loss.items()},
            "source_weighting": str(args.source_weighting),
            "source_steps_per_epoch": {x["name"]: int(x["repeats"]) for x in loaded},
        }
        print(json.dumps(event, ensure_ascii=False, sort_keys=True))
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            _save_ckpt(best_path, model=model, epoch=epoch, metric="joint_training_loss", value=epoch_loss, args=args, extra_state={"selection_note": "joint training-loss checkpoint; paper-table runner may override by target best"}, ratio_guidance_pair=ratio_guidance_pair)
        _save_ckpt(last_path, model=model, epoch=epoch, metric="joint_training_loss_last", value=epoch_loss, args=args, ratio_guidance_pair=ratio_guidance_pair)
        if args.save_every_epoch:
            _save_ckpt(epoch_dir / f"epoch_{epoch:04d}.pt", model=model, epoch=epoch, metric="joint_training_loss_epoch", value=epoch_loss, args=args, ratio_guidance_pair=ratio_guidance_pair)

    if gradient_diagnostics_events:
        diagnostics_path = out_dir / "loss_gradient_diagnostics.jsonl"
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            for row in gradient_diagnostics_events:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "protocol": "OFA-CS-V5-SeniorExact-joint-source-training",
        "source_graphs": source_graphs,
        "source_weighting": str(args.source_weighting),
        "source_repeat_weights": dict(explicit_repeats),
        "target_graphs_for_unlabeled_alignment": target_graphs,
        "strict_one_for_all_source_training": not bool(
            args.enable_density_ratio_guidance or args.enable_target_alignment_loss
        ),
        "enable_prefix_loss": bool(args.enable_prefix_loss),
        "prefix_loss_weight": float(args.prefix_loss_weight),
        "prefix_on": str(args.prefix_on),
        "prefix_fp_topk": int(getattr(args, "prefix_fp_topk", 64)),
        "prefix_precision_loss_weight": float(getattr(args, "prefix_precision_loss_weight", 0.0)),
        "prefix_state_aware_loss": bool(getattr(args, "prefix_state_aware_loss", False)),
        "prefix_stratified_query_sampling": bool(getattr(args, "prefix_stratified_query_sampling", False)),
        "prefix_state_candidate_limit": int(getattr(args, "prefix_state_candidate_limit", 128)),
        "enable_structural_boundary_loss": bool(getattr(args, "enable_structural_boundary_loss", False)),
        "structural_boundary_weight": float(getattr(args, "structural_boundary_weight", 0.0)),
        "structural_boundary_hop": int(getattr(args, "structural_boundary_hop", 2)),
        "structural_neg_topk": int(getattr(args, "structural_neg_topk", 64)),
        "enable_size_loss": bool(args.enable_size_loss),
        "enable_ranking_preservation_loss": bool(getattr(args, "enable_ranking_preservation_loss", False)),
        "ranking_preservation_weight": float(getattr(args, "ranking_preservation_weight", 0.0)),
        "ranking_preservation_mode": str(getattr(args, "ranking_preservation_mode", "cosine")),
        "enable_density_ratio_guidance": bool(args.enable_density_ratio_guidance),
        "ratio_guidance_weight": float(getattr(args, "ratio_guidance_weight", 0.05)),
        "ratio_guidance_clip": float(getattr(args, "ratio_guidance_clip", 5.0)),
        "ratio_guidance_hidden_dim": int(getattr(args, "ratio_guidance_hidden_dim", 256)),
        "ratio_guidance_time_emb_dim": int(getattr(args, "ratio_guidance_time_emb_dim", 64)),
        "ratio_guidance_sampling_seed": None if args.ratio_guidance_sampling_seed is None else int(args.ratio_guidance_sampling_seed),
        "target_guidance_schedule": str(args.target_guidance_schedule),
        "grad_clip_norm": float(args.grad_clip_norm),
        "enable_state_conditioned_community_pic_loss": bool(args.enable_state_conditioned_community_pic_loss),
        "community_pic_loss_weight": float(args.community_pic_loss_weight),
        "community_pic_temperature": float(args.community_pic_temperature),
        "community_pic_topk_core": int(args.community_pic_topk_core),
        "community_pic_hard_neg_topk": int(args.community_pic_hard_neg_topk),
        "community_pic_state_top_fraction": float(args.community_pic_state_top_fraction),
        "community_pic_aggregate": {
            **scpc_aggregate,
            "intra_mean": scpc_aggregate["intra_sum"] / max(1, scpc_aggregate["valid_queries"]),
            "inter_mean": scpc_aggregate["inter_sum"] / max(1, scpc_aggregate["valid_queries"]),
            "ratio_mean": scpc_aggregate["ratio_sum"] / max(1, scpc_aggregate["valid_queries"]),
            "separation_mean": scpc_aggregate["separation_sum"] / max(1, scpc_aggregate["valid_queries"]),
            "separation_active_rate": scpc_aggregate["separation_active_sum"] / max(1, scpc_aggregate["valid_queries"]),
            "core_nodes_mean": scpc_aggregate["core_nodes_sum"] / max(1, scpc_aggregate["valid_queries"]),
        },
        "enable_controlled_error_state_synthesis_loss": bool(args.enable_controlled_error_state_synthesis_loss),
        "cdss_loss_weight": float(args.cdss_loss_weight),
        "cdss_config": {
            "temperature": float(args.cdss_temperature),
            "core_topk": int(args.cdss_core_topk),
            "boundary_pos_topk": int(args.cdss_boundary_pos_topk),
            "hard_neg_topk": int(args.cdss_hard_neg_topk),
            "under_recall": float(args.cdss_under_recall),
            "moderate_recall": float(args.cdss_moderate_recall),
            "boundary_precision": float(args.cdss_boundary_precision),
            "stable_recall": float(args.cdss_stable_recall),
            "enabled_states": str(args.cdss_enabled_states),
            "aggregation": str(args.cdss_aggregation),
            "balance_min": float(args.cdss_balance_min),
            "balance_max": float(args.cdss_balance_max),
            "distance_mode": str(args.cdss_distance_mode),
            "separation_mode": str(args.cdss_separation_mode),
            "absolute_margin": float(args.cdss_absolute_margin),
            "absolute_separation_cap": float(args.cdss_absolute_separation_cap),
            "relative_margin": float(args.cdss_relative_margin),
            "reference_stage": "pre_optimization_raw_embedding",
        },
        "cdss_reference_report": cdss_reference_report,
        "cdss_aggregate": {
            **cdss_aggregate,
            "positive_distance_mean": cdss_aggregate["positive_distance_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "negative_distance_mean": cdss_aggregate["negative_distance_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "ratio_mean": cdss_aggregate["ratio_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "separation_mean": cdss_aggregate["separation_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "separation_active_rate": cdss_aggregate["separation_active_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "core_nodes_mean": cdss_aggregate["core_nodes_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "selected_positive_nodes_mean": cdss_aggregate["positive_nodes_sum"] / max(1, cdss_aggregate["valid_queries"]),
            "selected_negative_nodes_mean": cdss_aggregate["negative_nodes_sum"] / max(1, cdss_aggregate["valid_queries"]),
        },
        "loss_gradient_diagnostics_every": int(args.loss_gradient_diagnostics_every),
        "loss_gradient_diagnostics": gradient_diagnostics_events,
        "enable_target_alignment_loss": bool(getattr(args, "enable_target_alignment_loss", False)),
        "alignment_loss_weight": float(args.alignment_loss_weight),
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_updates": int(args.warmup_updates),
        "tot_updates": int(args.tot_updates),
        "epochs": int(args.epochs),
        "seed": None if args.seed is None else int(args.seed),
        "deterministic": bool(args.deterministic),
        "deterministic_strict": bool(args.deterministic_strict),
        "best_training_loss": best_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epoch_checkpoint_dir": str(epoch_dir) if args.save_every_epoch else None,
        "checkpoint_selection_from_unseen": False,
        "unseen_metrics_used": False,
        "input_status": input_status,
    }
    (out_dir / "senior_exact_joint_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    profile_event(runtime_jsonl, "senior_exact_joint_train_end", best_training_loss=best_loss, best_checkpoint=str(best_path))


if __name__ == "__main__":
    main()
