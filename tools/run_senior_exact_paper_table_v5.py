#!/usr/bin/env python3
"""Run SeniorExact paper-table subset/full setting.

This runner is for the senior paper-style setting where one shared model is
trained jointly on the paper source graphs (e.g. texas,photo,dblp), and every
training epoch is evaluated on each target graph.  For each target it records
both target-best-F1 and target-best-JAC rows, matching the senior paper's
reported target-best protocol.

It is intentionally not an all-pairs runner and not a wrapper that trains one
checkpoint per source.  It trains exactly one joint-source model.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ofacs.offline_eval import file_sha256
from ofacs.v55_presets import V55_PRESET_CHOICES, apply_v55_preset
from ofacs.preset_override import apply_preset_overrides, namespace_jsonable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--source-graphs", required=True, help="comma-separated joint sources, e.g. texas,photo,dblp")
    p.add_argument("--target-graphs", required=True, help="comma-separated targets, e.g. wisconsin,cora,citeseer,cs")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=15)
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
    p.add_argument("--lr", type=float, default=1e-3, help="base/peak LR; senior original uses 0.01 with polynomial schedule")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--lr-scheduler", choices=["none", "polynomial"], default="none")
    p.add_argument("--warmup-updates", type=int, default=400)
    p.add_argument("--tot-updates", type=int, default=1000)
    p.add_argument("--end-lr", type=float, default=1e-9)
    p.add_argument("--lr-power", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--source-weighting", choices=["step_weighted", "uniform", "explicit"], default="step_weighted", help="final mainline uses senior-faithful step weighting; explicit is kept only for ablations")
    p.add_argument("--source-repeat-weights", default="texas=1,photo=1,dblp=3", help="for explicit source weighting, e.g. texas=1,photo=1,dblp=3")
    p.add_argument("--enable-link-loss", action="store_true")
    p.add_argument("--max-link-loss-nodes", type=int, default=10000)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--t-eval", type=int, default=None, help="reverse denoise start step for fixed export; useful for guided_refined cost control")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-start-epoch", type=int, default=1)
    p.add_argument("--eval-before-training", action="store_true", help="also evaluate epoch_0000 initial checkpoint before training")
    p.add_argument("--eval-end-epoch", type=int, default=None)
    p.add_argument("--run-offline-eval", action="store_true", default=True)
    p.add_argument("--allow-unseen-ground-truth-eval", action="store_true")
    p.add_argument("--keep-epoch-predictions", action="store_true", help="keep every epoch prediction dir; default keeps pred/json/status but removes heavy embedding shards")
    p.add_argument("--score-embedding", choices=["raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"], default="guided_refined", help="final mainline uses DiffGDA density-ratio guided refined embeddings")
    p.add_argument("--score-calibration", choices=["none", "center", "zscore", "local_center", "temperature", "precision_profile", "recall_profile"], default="none", help="label-free per-query score/search profile before senior GlobalSearch")
    p.add_argument("--multi-view-search", action=argparse.BooleanOptionalAction, default=False, help="label-free query-wise candidate selection among multiple score views")
    p.add_argument("--candidate-views", default="guided_refined:none,adaptive_raw_plus_guided_refined:none,guided_refined:local_center,guided_refined:temperature,guided_refined:precision_profile,guided_refined:recall_profile")
    p.add_argument("--candidate-selection", choices=["senior_density", "hybrid", "diversity_hybrid", "structure_quality", "canonical_guard", "no_temperature", "temperature_guard", "recall_safe", "balanced_size", "citeseer_safe_unsupervised", "structure_secondary"], default="diversity_hybrid")
    p.add_argument("--candidate-compactness-weight", type=float, default=0.12)
    p.add_argument("--candidate-boundary-gap-weight", type=float, default=0.08)
    p.add_argument("--candidate-size-penalty-weight", type=float, default=0.12)
    p.add_argument("--candidate-structural-quality-weight", type=float, default=0.18)
    p.add_argument("--candidate-lcc-weight", type=float, default=0.08)
    p.add_argument("--candidate-conductance-weight", type=float, default=0.08)
    p.add_argument("--candidate-raw-agreement-weight", type=float, default=0.05)
    p.add_argument("--candidate-diversity-bonus-weight", type=float, default=0.05)
    p.add_argument("--candidate-size-reference-power", type=float, default=0.5)
    p.add_argument("--candidate-canonical-view", default="auto")
    p.add_argument("--candidate-duplicate-jaccard-threshold", type=float, default=0.98)
    p.add_argument("--candidate-topk-overlap-k", type=int, default=100)
    p.add_argument("--candidate-temperature-value", type=float, default=0.20)
    p.add_argument("--candidate-local-center-top-fraction", type=float, default=0.10)
    p.add_argument("--candidate-precision-size-strength", type=float, default=0.08)
    p.add_argument("--candidate-recall-size-bonus", type=float, default=0.04)
    p.add_argument("--candidate-canonical-switch-margin", type=float, default=0.08)
    p.add_argument("--candidate-alternative-min-size-ratio", type=float, default=0.85)
    p.add_argument("--candidate-temperature-min-size-ratio", type=float, default=0.85)
    p.add_argument("--candidate-temperature-entropy-threshold", type=float, default=0.82)
    p.add_argument("--candidate-balanced-size-low", type=float, default=0.85)
    p.add_argument("--candidate-balanced-size-high", type=float, default=1.25)
    p.add_argument("--candidate-under-size-penalty-weight", type=float, default=0.30)
    p.add_argument("--candidate-temperature-penalty-weight", type=float, default=0.35)
    p.add_argument("--candidate-recall-view-bonus", type=float, default=0.08)
    p.add_argument("--candidate-canonical-dominance-guard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-candidate-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v55-preset", choices=V55_PRESET_CHOICES, default="off", help="compact V5.5 preset; keeps experiments interpretable and avoids sweeping low-level alpha weights")
    p.add_argument("--preset-override-json", default="", help="JSON object, JSON file path, or @file applied after --v55-preset; unknown fields fail")
    p.add_argument("--enable-state-conditioned-community-pic-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--community-pic-loss-weight", type=float, default=0.05)
    p.add_argument("--community-pic-temperature", type=float, default=0.1)
    p.add_argument("--community-pic-topk-core", type=int, default=64)
    p.add_argument("--community-pic-hard-neg-topk", type=int, default=64)
    p.add_argument("--community-pic-state-top-fraction", type=float, default=0.10)
    p.add_argument("--enable-controlled-error-state-synthesis-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cdss-loss-weight", type=float, default=0.0125)
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
    p.add_argument("--loss-gradient-diagnostics-every", type=int, default=0)
    p.add_argument("--dual-branch-score", action=argparse.BooleanOptionalAction, default=False, help="V5.5 dual-branch preservation/adaptation scoring")
    p.add_argument("--dual-preserved-score-embedding", help=argparse.SUPPRESS, choices=["raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"], default="raw")
    p.add_argument("--dual-adapted-score-embedding", help=argparse.SUPPRESS, choices=["raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"], default="guided_refined")
    p.add_argument("--dual-preserved-checkpoint-source", choices=["current", "epoch0"], default="current", help=argparse.SUPPRESS)
    p.add_argument("--dual-alpha-mode", help=argparse.SUPPRESS, choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--dual-fixed-alpha", help=argparse.SUPPRESS, type=float, default=0.50)
    p.add_argument("--dual-min-alpha", help=argparse.SUPPRESS, type=float, default=0.10)
    p.add_argument("--dual-max-alpha", help=argparse.SUPPRESS, type=float, default=0.90)
    p.add_argument("--dual-base-alpha", help=argparse.SUPPRESS, type=float, default=0.50)
    p.add_argument("--dual-over-preserved-weight", help=argparse.SUPPRESS, type=float, default=0.18)
    p.add_argument("--dual-adapted-density-weight", help=argparse.SUPPRESS, type=float, default=0.12)
    p.add_argument("--dual-entropy-preserve-weight", help=argparse.SUPPRESS, type=float, default=0.16)
    p.add_argument("--dual-disagreement-preserve-weight", help=argparse.SUPPRESS, type=float, default=0.18)
    p.add_argument("--dual-adapted-undersize-weight", help=argparse.SUPPRESS, type=float, default=0.22)
    p.add_argument("--dual-adapted-oversize-weight", help=argparse.SUPPRESS, type=float, default=0.08)
    p.add_argument("--dual-boundary-gap-adapt-weight", help=argparse.SUPPRESS, type=float, default=0.08)
    p.add_argument("--dual-residual-gain", help=argparse.SUPPRESS, type=float, default=1.0)
    p.add_argument("--dual-search-profile", help=argparse.SUPPRESS, choices=["senior", "balanced_size", "entropy_aware", "precision_recall", "target_state_recovery"], default="senior")
    p.add_argument("--dual-balanced-size-low", help=argparse.SUPPRESS, type=float, default=0.85)
    p.add_argument("--dual-balanced-size-high", help=argparse.SUPPRESS, type=float, default=1.25)
    p.add_argument("--dual-entropy-threshold", help=argparse.SUPPRESS, type=float, default=0.82)
    p.add_argument("--dual-high-entropy-recall-bonus", help=argparse.SUPPRESS, type=float, default=0.04)
    p.add_argument("--dual-target-state-gate", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-graph-bias-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-graph-bias-strength", help=argparse.SUPPRESS, type=float, default=0.18)
    p.add_argument("--dual-graph-bias-clip", help=argparse.SUPPRESS, type=float, default=0.18)
    p.add_argument("--dual-graph-over-preserved-weight", help=argparse.SUPPRESS, type=float, default=0.10)
    p.add_argument("--dual-graph-adapted-density-weight", help=argparse.SUPPRESS, type=float, default=0.08)
    p.add_argument("--dual-graph-adapted-undersize-weight", help=argparse.SUPPRESS, type=float, default=0.16)
    p.add_argument("--dual-graph-disagreement-preserve-weight", help=argparse.SUPPRESS, type=float, default=0.06)
    p.add_argument("--dual-graph-entropy-preserve-weight", help=argparse.SUPPRESS, type=float, default=0.04)
    p.add_argument("--dual-query-undersize-guard-weight", help=argparse.SUPPRESS, type=float, default=0.18)
    p.add_argument("--dual-query-oversize-guard-weight", help=argparse.SUPPRESS, type=float, default=0.10)
    p.add_argument("--dual-density-gain-tolerance", help=argparse.SUPPRESS, type=float, default=0.015)
    p.add_argument("--dual-search-consensus-bonus", help=argparse.SUPPRESS, type=float, default=0.025)
    p.add_argument("--dual-guarded-floor-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-guarded-floor-ratio", help=argparse.SUPPRESS, type=float, default=0.85)
    p.add_argument("--dual-guarded-floor-guard-metric", help=argparse.SUPPRESS, choices=["quality", "density", "either", "both"], default="quality")
    p.add_argument("--dual-guarded-floor-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="strict_abs")
    p.add_argument("--dual-guarded-floor-density-tolerance", help=argparse.SUPPRESS, type=float, default=0.020)
    p.add_argument("--dual-guarded-floor-quality-tolerance", help=argparse.SUPPRESS, type=float, default=0.020)
    p.add_argument("--dual-guarded-floor-rel-density-tolerance", help=argparse.SUPPRESS, type=float, default=0.050)
    p.add_argument("--dual-guarded-floor-rel-quality-tolerance", help=argparse.SUPPRESS, type=float, default=0.050)
    p.add_argument("--dual-guarded-floor-target-trigger", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-guarded-floor-target-under-rate-threshold", help=argparse.SUPPRESS, type=float, default=0.35)
    p.add_argument("--dual-guarded-floor-probe-ratio", help=argparse.SUPPRESS, type=float, default=0.85)
    p.add_argument("--dual-guarded-floor-router-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-guarded-floor-router-strong-under-rate-threshold", help=argparse.SUPPRESS, type=float, default=0.35)
    p.add_argument("--dual-guarded-floor-router-moderate-under-rate-threshold", help=argparse.SUPPRESS, type=float, default=0.12)
    p.add_argument("--dual-guarded-floor-router-quality-drop-threshold", help=argparse.SUPPRESS, type=float, default=0.10)
    p.add_argument("--dual-guarded-floor-router-density-drop-threshold", help=argparse.SUPPRESS, type=float, default=1.00)
    p.add_argument("--dual-guarded-floor-router-moderate-max-branch-jaccard", help=argparse.SUPPRESS, type=float, default=1.00)
    p.add_argument("--dual-guarded-floor-router-strong-ratio", help=argparse.SUPPRESS, type=float, default=1.00)
    p.add_argument("--dual-guarded-floor-router-moderate-ratio", help=argparse.SUPPRESS, type=float, default=0.90)
    p.add_argument("--dual-guarded-floor-router-strong-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="force")
    p.add_argument("--dual-guarded-floor-router-moderate-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="relative_quality")
    p.add_argument("--dual-state-profile-router-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-state-recognition-v2-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-moderate-size-ratio-min", help=argparse.SUPPRESS, type=float, default=1.08)
    p.add_argument("--dual-state-profile-moderate-branch-jaccard-max", help=argparse.SUPPRESS, type=float, default=0.62)
    p.add_argument("--dual-state-profile-moderate-over-rate-min", help=argparse.SUPPRESS, type=float, default=0.25)
    p.add_argument("--dual-state-profile-moderate-density-delta-min", help=argparse.SUPPRESS, type=float, default=-1.20)
    p.add_argument("--dual-state-profile-moderate-entropy-min", help=argparse.SUPPRESS, type=float, default=0.90)
    p.add_argument("--dual-state-profile-stable-size-ratio-low", help=argparse.SUPPRESS, type=float, default=0.90)
    p.add_argument("--dual-state-profile-stable-size-ratio-high", help=argparse.SUPPRESS, type=float, default=1.20)
    p.add_argument("--dual-state-profile-stable-branch-jaccard-min", help=argparse.SUPPRESS, type=float, default=0.70)
    p.add_argument("--dual-state-profile-stable-tight-size-ratio-low", help=argparse.SUPPRESS, type=float, default=0.95)
    p.add_argument("--dual-state-profile-stable-tight-size-ratio-high", help=argparse.SUPPRESS, type=float, default=1.08)
    p.add_argument("--dual-state-profile-stable-tight-branch-jaccard-min", help=argparse.SUPPRESS, type=float, default=0.65)
    p.add_argument("--dual-state-profile-stable-entropy-min", help=argparse.SUPPRESS, type=float, default=0.99)
    p.add_argument("--dual-state-profile-stable-density-delta-abs-max", help=argparse.SUPPRESS, type=float, default=0.04)
    p.add_argument("--dual-state-profile-strong-size-ratio-min", help=argparse.SUPPRESS, type=float, default=1.35)
    p.add_argument("--dual-state-profile-strong-branch-jaccard-max", help=argparse.SUPPRESS, type=float, default=0.72)
    p.add_argument("--dual-state-profile-shift-density-delta-min", help=argparse.SUPPRESS, type=float, default=0.04)
    p.add_argument("--dual-state-profile-shift-branch-jaccard-min", help=argparse.SUPPRESS, type=float, default=0.70)
    p.add_argument("--dual-state-profile-alpha-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-search-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-core-proto-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-structure-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-strong-alpha-bias", help=argparse.SUPPRESS, type=float, default=0.08)
    p.add_argument("--dual-state-profile-moderate-alpha-bias", help=argparse.SUPPRESS, type=float, default=0.02)
    p.add_argument("--dual-state-profile-boundary-alpha-bias", help=argparse.SUPPRESS, type=float, default=-0.10)
    p.add_argument("--dual-state-profile-shift-alpha-bias", help=argparse.SUPPRESS, type=float, default=0.08)
    p.add_argument("--dual-state-profile-stable-alpha-bias", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-state-profile-core-weight-strong", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-state-profile-core-weight-moderate", help=argparse.SUPPRESS, type=float, default=0.12)
    p.add_argument("--dual-state-profile-core-weight-boundary", help=argparse.SUPPRESS, type=float, default=0.18)
    p.add_argument("--dual-state-profile-core-weight-shift", help=argparse.SUPPRESS, type=float, default=0.05)
    p.add_argument("--dual-state-profile-core-weight-stable", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-state-profile-core-max-nodes", help=argparse.SUPPRESS, type=int, default=64)
    p.add_argument("--dual-state-profile-core-source-mode", help=argparse.SUPPRESS, choices=["profile_top", "agreement_top", "agreement_then_profile"], default="profile_top")
    p.add_argument("--dual-state-profile-core-agreement-topk", help=argparse.SUPPRESS, type=int, default=160)
    p.add_argument("--dual-state-profile-core-source-by-state-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-core-source-mode-strong", help=argparse.SUPPRESS, choices=["profile_top", "agreement_top", "agreement_then_profile"], default="profile_top")
    p.add_argument("--dual-state-profile-core-source-mode-moderate", help=argparse.SUPPRESS, choices=["profile_top", "agreement_top", "agreement_then_profile"], default="agreement_then_profile")
    p.add_argument("--dual-state-profile-core-source-mode-boundary", help=argparse.SUPPRESS, choices=["profile_top", "agreement_top", "agreement_then_profile"], default="agreement_then_profile")
    p.add_argument("--dual-state-profile-core-source-mode-shift", help=argparse.SUPPRESS, choices=["profile_top", "agreement_top", "agreement_then_profile"], default="profile_top")
    p.add_argument("--dual-state-profile-core-source-mode-stable", help=argparse.SUPPRESS, choices=["profile_top", "agreement_top", "agreement_then_profile"], default="agreement_top")
    p.add_argument("--dual-state-profile-core-strict-agreement", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-core-min-agreement-nodes", help=argparse.SUPPRESS, type=int, default=4)
    p.add_argument("--dual-state-profile-score-source-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-action-isolation-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-action-isolation-scope", help=argparse.SUPPRESS, choices=["moderate_only", "profile_specific"], default="moderate_only")
    p.add_argument("--dual-state-profile-score-mode-strong", help=argparse.SUPPRESS, choices=["default_fusion", "adapt_preferred", "weak_adapt_residual", "preserve_consensus"], default="adapt_preferred")
    p.add_argument("--dual-state-profile-score-mode-moderate", help=argparse.SUPPRESS, choices=["default_fusion", "adapt_preferred", "weak_adapt_residual", "preserve_consensus"], default="weak_adapt_residual")
    p.add_argument("--dual-state-profile-score-mode-boundary", help=argparse.SUPPRESS, choices=["default_fusion", "adapt_preferred", "weak_adapt_residual", "preserve_consensus"], default="preserve_consensus")
    p.add_argument("--dual-state-profile-score-mode-shift", help=argparse.SUPPRESS, choices=["default_fusion", "adapt_preferred", "weak_adapt_residual", "preserve_consensus"], default="adapt_preferred")
    p.add_argument("--dual-state-profile-score-mode-stable", help=argparse.SUPPRESS, choices=["default_fusion", "adapt_preferred", "weak_adapt_residual", "preserve_consensus"], default="default_fusion")
    p.add_argument("--dual-state-profile-score-residual-scale-strong", help=argparse.SUPPRESS, type=float, default=1.05)
    p.add_argument("--dual-state-profile-score-residual-scale-moderate", help=argparse.SUPPRESS, type=float, default=0.70)
    p.add_argument("--dual-state-profile-score-residual-scale-boundary", help=argparse.SUPPRESS, type=float, default=0.55)
    p.add_argument("--dual-state-profile-score-residual-scale-shift", help=argparse.SUPPRESS, type=float, default=1.15)
    p.add_argument("--dual-state-profile-score-residual-scale-stable", help=argparse.SUPPRESS, type=float, default=1.00)
    p.add_argument("--dual-state-profile-score-recall-weight-moderate", help=argparse.SUPPRESS, type=float, default=0.040)
    p.add_argument("--dual-state-profile-score-consensus-weight-boundary", help=argparse.SUPPRESS, type=float, default=0.050)
    p.add_argument("--dual-state-profile-score-adapt-bonus-shift", help=argparse.SUPPRESS, type=float, default=0.030)
    p.add_argument("--dual-hierarchical-boundary-state-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-boundary-plateau-relative-quality-threshold", help=argparse.SUPPRESS, type=float, default=1.0e-6)
    p.add_argument("--dual-boundary-plateau-precision-extra-penalty", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-boundary-plateau-ceiling-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-boundary-plateau-ceiling-ratio", help=argparse.SUPPRESS, type=float, default=0.95)
    p.add_argument("--dual-boundary-plateau-ceiling-rel-quality-tolerance", help=argparse.SUPPRESS, type=float, default=0.01)
    p.add_argument("--dual-boundary-plateau-ceiling-rel-density-tolerance", help=argparse.SUPPRESS, type=float, default=0.01)
    p.add_argument("--dual-state-profile-core-reliability-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-core-reliability-min", help=argparse.SUPPRESS, type=float, default=0.25)
    p.add_argument("--dual-state-profile-core-reliability-min-nodes", help=argparse.SUPPRESS, type=int, default=8)
    p.add_argument("--dual-state-profile-core-reliability-agreement-weight", help=argparse.SUPPRESS, type=float, default=0.50)
    p.add_argument("--dual-state-profile-core-reliability-count-weight", help=argparse.SUPPRESS, type=float, default=0.30)
    p.add_argument("--dual-state-profile-core-reliability-query-sim-weight", help=argparse.SUPPRESS, type=float, default=0.20)
    p.add_argument("--dual-state-profile-structure-weight-moderate", help=argparse.SUPPRESS, type=float, default=0.020)
    p.add_argument("--dual-state-profile-structure-weight-boundary", help=argparse.SUPPRESS, type=float, default=0.035)
    p.add_argument("--dual-state-profile-structure-weight-strong", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-state-profile-structure-weight-shift", help=argparse.SUPPRESS, type=float, default=0.010)
    p.add_argument("--dual-state-profile-structure-weight-stable", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-state-profile-structure-core-topk", help=argparse.SUPPRESS, type=int, default=24)
    p.add_argument("--dual-state-profile-moderate-recall-bonus", help=argparse.SUPPRESS, type=float, default=0.030)
    p.add_argument("--dual-state-profile-boundary-quality-penalty", help=argparse.SUPPRESS, type=float, default=0.040)
    p.add_argument("--dual-state-profile-shift-adapt-bonus", help=argparse.SUPPRESS, type=float, default=0.020)
    p.add_argument("--dual-state-profile-alpha-bounds-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-strong-alpha-min", help=argparse.SUPPRESS, type=float, default=0.55)
    p.add_argument("--dual-state-profile-moderate-alpha-min", help=argparse.SUPPRESS, type=float, default=0.45)
    p.add_argument("--dual-state-profile-moderate-alpha-max", help=argparse.SUPPRESS, type=float, default=0.72)
    p.add_argument("--dual-state-profile-boundary-alpha-max", help=argparse.SUPPRESS, type=float, default=0.45)
    p.add_argument("--dual-state-profile-shift-alpha-min", help=argparse.SUPPRESS, type=float, default=0.60)
    p.add_argument("--dual-state-profile-floor-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-state-profile-strong-floor-ratio", help=argparse.SUPPRESS, type=float, default=1.00)
    p.add_argument("--dual-state-profile-moderate-floor-ratio", help=argparse.SUPPRESS, type=float, default=0.90)
    p.add_argument("--dual-state-profile-boundary-floor-ratio", help=argparse.SUPPRESS, type=float, default=0.00)
    p.add_argument("--dual-state-profile-shift-floor-ratio", help=argparse.SUPPRESS, type=float, default=0.00)
    p.add_argument("--dual-state-profile-stable-floor-ratio", help=argparse.SUPPRESS, type=float, default=0.00)
    p.add_argument("--dual-state-profile-strong-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="relative_quality")
    p.add_argument("--dual-state-profile-moderate-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="relative_quality")
    p.add_argument("--dual-state-profile-boundary-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="relative_quality")
    p.add_argument("--dual-state-profile-shift-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="relative_quality")
    p.add_argument("--dual-state-profile-stable-accept-mode", help=argparse.SUPPRESS, choices=["strict_abs", "force", "relative_quality", "relative_quality_density"], default="relative_quality")
    p.add_argument("--write-dual-branch-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ratio-guidance-weight", type=float, default=0.01, help="reverse denoise coefficient for grad_z log q_T/p_S")
    p.add_argument("--ratio-guidance-clip", type=float, default=5.0)
    p.add_argument("--adaptive-ratio-guidance", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ratio-guidance-confidence-scale", type=float, default=1.0)
    p.add_argument("--ratio-guidance-min-mult", type=float, default=0.25)
    p.add_argument("--ratio-guidance-max-mult", type=float, default=1.25)
    p.add_argument(
        "--target-evidence-guidance-mode",
        choices=["off", "diagonal_gaussian"],
        default="off",
        help="parameter-free unlabeled target evidence fitted only during fixed inference",
    )
    p.add_argument("--target-evidence-fit-chunk-size", type=int, default=8192)
    p.add_argument("--target-evidence-variance-floor", type=float, default=1e-4)
    p.add_argument("--target-evidence-variance-shrinkage", type=float, default=0.05)
    p.add_argument("--target-evidence-confidence-scale", type=float, default=2.0)
    p.add_argument("--target-evidence-max-relative-update", type=float, default=0.05)
    p.add_argument("--adaptive-fusion", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fusion-min-guided", type=float, default=0.25)
    p.add_argument("--fusion-max-guided", type=float, default=0.85)
    p.add_argument("--fusion-confidence-scale", type=float, default=8.0)
    p.add_argument("--ratio-guidance-lr", type=float, default=1e-3)
    p.add_argument("--ratio-guidance-hidden-dim", type=int, default=256)
    p.add_argument("--ratio-guidance-time-emb-dim", type=int, default=64)
    p.add_argument("--ratio-guidance-max-nodes", type=int, default=4096)
    p.add_argument("--ratio-guidance-sampling-seed", type=int, default=None, help="optional fixed seed for target/guidance subsampling, independent of model seed")
    p.add_argument("--target-guidance-schedule", choices=["legacy_double_step", "round_robin"], default="legacy_double_step", help="target graph schedule for unlabeled guidance")
    p.add_argument("--grad-clip-norm", type=float, default=0.0, help="optional main-model gradient clipping; 0 disables")
    p.add_argument("--enable-prefix-loss", action=argparse.BooleanOptionalAction, default=True, help="source-only GlobalSearch-prefix boundary loss")
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
    p.add_argument("--prefix-state-aware-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--prefix-stratified-query-sampling", action=argparse.BooleanOptionalAction, default=False)
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
    p.add_argument("--enable-size-loss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--size-loss-weight", type=float, default=0.01)
    p.add_argument("--size-over-tolerance", type=float, default=0.15)
    p.add_argument("--enable-ranking-preservation-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ranking-preservation-weight", type=float, default=0.0)
    p.add_argument("--ranking-preservation-mode", choices=["cosine", "l2", "cosine_l2"], default="cosine")
    p.add_argument("--enable-density-ratio-guidance", action=argparse.BooleanOptionalAction, default=True, help="train/save DiffGDA-style target-unlabeled density-ratio reverse guidance")
    p.add_argument("--enable-target-alignment-loss", action="store_true", help="optional MMD/CORAL proxy; separate from density-ratio reverse guidance")
    p.add_argument("--alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--alignment-max-nodes", type=int, default=4096)
    p.add_argument("--alignment-coral-weight", type=float, default=0.1)
    p.add_argument("--target-name-alias", default="", help="optional aliases like cs=CoCS for reports only")
    p.add_argument("--seed", type=int, default=None, help="fixed seed injected into training and fixed export")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--deterministic-strict", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--stage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="forward synchronized training-stage timing to the joint trainer",
    )
    args = p.parse_args(argv)
    apply_v55_preset(args)
    apply_preset_overrides(
        args,
        args.preset_override_json,
        forbidden_keys={"dataset_root", "source_graphs", "target_graphs", "output_dir", "allow_unseen_ground_truth_eval"},
    )
    return args


def _parse_list(s: str) -> list[str]:
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    if not out:
        raise SystemExit("empty graph list")
    return out


def _parse_alias(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad --target-name-alias item: {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _training_targets(args: argparse.Namespace, evaluation_targets: list[str]) -> list[str]:
    """Return graphs exposed to training, separately from fixed evaluation."""

    if bool(args.enable_density_ratio_guidance) or bool(args.enable_target_alignment_loss):
        return list(evaluation_targets)
    return []


def run(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def _read_eval_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    if not records:
        raise RuntimeError(f"offline eval json has no records: {path}")
    return dict(records[0])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "target", "target_display", "best_by", "epoch", "selection_value", "checkpoint", "checkpoint_sha256",
        "official_precision", "official_recall", "official_f1", "official_jaccard",
        "senior_micro_precision", "senior_micro_recall", "senior_micro_f1", "senior_nmi", "senior_ari", "senior_jaccard",
        "official_metric_protocol", "prediction_sha256", "query_sha256", "gt_sha256",
        "macro_precision", "macro_recall", "macro_f1", "macro_jaccard",
        "avg_pred_size", "avg_gt_size", "avg_pred_over_gt", "endpoint_hit_rate", "near_endpoint_hit_rate", "query_missing_rows",
        "prediction_dir", "prediction_file", "offline_eval_json", "offline_eval_csv", "fixed_export_command_json", "fixed_export_command_sha256",
    ]
    keys = preferred + sorted(k for row in rows for k in row.keys() if k not in preferred)
    seen: list[str] = []
    for k in keys:
        if k not in seen:
            seen.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=seen)
        w.writeheader()
        w.writerows(rows)


def _metric(record: dict[str, Any], key: str) -> float:
    return float(record[key])


def _export_and_eval(args: argparse.Namespace, *, target: str, ckpt: Path, epoch: int, out: Path, aliases: dict[str, str]) -> dict[str, Any]:
    py = sys.executable
    base = out / "target_runs" / target / f"epoch_{epoch:04d}"
    pred_dir = base / "predictions" / target
    eval_dir = base / "offline_eval"
    log_prefix = base / "logs" / f"{target}_epoch_{epoch:04d}"

    export_cmd = [
        py, "tools/evaluate_fixed_checkpoint_ofacs.py",
        "--graph-name", target,
        "--checkpoint", str(ckpt),
        "--dataset-root", args.dataset_root,
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--chunk-size", str(args.chunk_size),
        "--embedding-export-format", "shards",
        "--dims", str(args.dims),
        "--hops", str(args.hops),
        "--local-topk", str(args.local_topk),
        "--pos-count", str(args.pos_count),
        "--neg-count", str(args.neg_count),
        "--senior-cache-mode", "required",
        "--feature-projection-mode", "identity",
        "--score-embedding", args.score_embedding,
        "--ratio-guidance-weight", str(args.ratio_guidance_weight),
        "--ratio-guidance-clip", str(args.ratio_guidance_clip),
        "--score-calibration", str(args.score_calibration),
        "--candidate-views", str(args.candidate_views),
        "--candidate-selection", str(args.candidate_selection),
        "--candidate-compactness-weight", str(args.candidate_compactness_weight),
        "--candidate-boundary-gap-weight", str(args.candidate_boundary_gap_weight),
        "--candidate-size-penalty-weight", str(args.candidate_size_penalty_weight),
        "--candidate-structural-quality-weight", str(args.candidate_structural_quality_weight),
        "--candidate-lcc-weight", str(args.candidate_lcc_weight),
        "--candidate-conductance-weight", str(args.candidate_conductance_weight),
        "--candidate-raw-agreement-weight", str(args.candidate_raw_agreement_weight),
        "--candidate-diversity-bonus-weight", str(args.candidate_diversity_bonus_weight),
        "--candidate-size-reference-power", str(args.candidate_size_reference_power),
        "--candidate-canonical-view", str(args.candidate_canonical_view),
        "--candidate-duplicate-jaccard-threshold", str(args.candidate_duplicate_jaccard_threshold),
        "--candidate-topk-overlap-k", str(args.candidate_topk_overlap_k),
        "--candidate-temperature-value", str(args.candidate_temperature_value),
        "--candidate-local-center-top-fraction", str(args.candidate_local_center_top_fraction),
        "--candidate-precision-size-strength", str(args.candidate_precision_size_strength),
        "--candidate-recall-size-bonus", str(args.candidate_recall_size_bonus),
        "--candidate-canonical-switch-margin", str(args.candidate_canonical_switch_margin),
        "--candidate-alternative-min-size-ratio", str(args.candidate_alternative_min_size_ratio),
        "--candidate-temperature-min-size-ratio", str(args.candidate_temperature_min_size_ratio),
        "--candidate-temperature-entropy-threshold", str(args.candidate_temperature_entropy_threshold),
        "--candidate-balanced-size-low", str(args.candidate_balanced_size_low),
        "--candidate-balanced-size-high", str(args.candidate_balanced_size_high),
        "--candidate-under-size-penalty-weight", str(args.candidate_under_size_penalty_weight),
        "--candidate-temperature-penalty-weight", str(args.candidate_temperature_penalty_weight),
        "--candidate-recall-view-bonus", str(args.candidate_recall_view_bonus),
        "--ratio-guidance-confidence-scale", str(args.ratio_guidance_confidence_scale),
        "--ratio-guidance-min-mult", str(args.ratio_guidance_min_mult),
        "--ratio-guidance-max-mult", str(args.ratio_guidance_max_mult),
        "--target-evidence-guidance-mode", str(args.target_evidence_guidance_mode),
        "--target-evidence-fit-chunk-size", str(args.target_evidence_fit_chunk_size),
        "--target-evidence-variance-floor", str(args.target_evidence_variance_floor),
        "--target-evidence-variance-shrinkage", str(args.target_evidence_variance_shrinkage),
        "--target-evidence-confidence-scale", str(args.target_evidence_confidence_scale),
        "--target-evidence-max-relative-update", str(args.target_evidence_max_relative_update),
        "--fusion-min-guided", str(args.fusion_min_guided),
        "--fusion-max-guided", str(args.fusion_max_guided),
        "--fusion-confidence-scale", str(args.fusion_confidence_scale),
        "--dual-preserved-score-embedding", str(args.dual_preserved_score_embedding),
        "--dual-adapted-score-embedding", str(args.dual_adapted_score_embedding),
        "--dual-alpha-mode", str(args.dual_alpha_mode),
        "--dual-fixed-alpha", str(args.dual_fixed_alpha),
        "--dual-min-alpha", str(args.dual_min_alpha),
        "--dual-max-alpha", str(args.dual_max_alpha),
        "--dual-base-alpha", str(args.dual_base_alpha),
        "--dual-over-preserved-weight", str(args.dual_over_preserved_weight),
        "--dual-adapted-density-weight", str(args.dual_adapted_density_weight),
        "--dual-entropy-preserve-weight", str(args.dual_entropy_preserve_weight),
        "--dual-disagreement-preserve-weight", str(args.dual_disagreement_preserve_weight),
        "--dual-adapted-undersize-weight", str(args.dual_adapted_undersize_weight),
        "--dual-adapted-oversize-weight", str(args.dual_adapted_oversize_weight),
        "--dual-boundary-gap-adapt-weight", str(args.dual_boundary_gap_adapt_weight),
        "--dual-residual-gain", str(args.dual_residual_gain),
        "--dual-search-profile", str(args.dual_search_profile),
        "--dual-balanced-size-low", str(args.dual_balanced_size_low),
        "--dual-balanced-size-high", str(args.dual_balanced_size_high),
        "--dual-entropy-threshold", str(args.dual_entropy_threshold),
        "--dual-high-entropy-recall-bonus", str(args.dual_high_entropy_recall_bonus),
        "--dual-graph-bias-strength", str(args.dual_graph_bias_strength),
        "--dual-graph-bias-clip", str(args.dual_graph_bias_clip),
        "--dual-graph-over-preserved-weight", str(args.dual_graph_over_preserved_weight),
        "--dual-graph-adapted-density-weight", str(args.dual_graph_adapted_density_weight),
        "--dual-graph-adapted-undersize-weight", str(args.dual_graph_adapted_undersize_weight),
        "--dual-graph-disagreement-preserve-weight", str(args.dual_graph_disagreement_preserve_weight),
        "--dual-graph-entropy-preserve-weight", str(args.dual_graph_entropy_preserve_weight),
        "--dual-query-undersize-guard-weight", str(args.dual_query_undersize_guard_weight),
        "--dual-query-oversize-guard-weight", str(args.dual_query_oversize_guard_weight),
        "--dual-density-gain-tolerance", str(args.dual_density_gain_tolerance),
        "--dual-search-consensus-bonus", str(args.dual_search_consensus_bonus),
        "--dual-guarded-floor-ratio", str(args.dual_guarded_floor_ratio),
        "--dual-guarded-floor-guard-metric", str(args.dual_guarded_floor_guard_metric),
        "--dual-guarded-floor-accept-mode", str(args.dual_guarded_floor_accept_mode),
        "--dual-guarded-floor-density-tolerance", str(args.dual_guarded_floor_density_tolerance),
        "--dual-guarded-floor-quality-tolerance", str(args.dual_guarded_floor_quality_tolerance),
        "--dual-guarded-floor-rel-density-tolerance", str(args.dual_guarded_floor_rel_density_tolerance),
        "--dual-guarded-floor-rel-quality-tolerance", str(args.dual_guarded_floor_rel_quality_tolerance),
        "--dual-guarded-floor-target-under-rate-threshold", str(args.dual_guarded_floor_target_under_rate_threshold),
        "--dual-guarded-floor-probe-ratio", str(args.dual_guarded_floor_probe_ratio),
        "--dual-guarded-floor-router-strong-under-rate-threshold", str(args.dual_guarded_floor_router_strong_under_rate_threshold),
        "--dual-guarded-floor-router-moderate-under-rate-threshold", str(args.dual_guarded_floor_router_moderate_under_rate_threshold),
        "--dual-guarded-floor-router-quality-drop-threshold", str(args.dual_guarded_floor_router_quality_drop_threshold),
        "--dual-guarded-floor-router-density-drop-threshold", str(args.dual_guarded_floor_router_density_drop_threshold),
        "--dual-guarded-floor-router-moderate-max-branch-jaccard", str(args.dual_guarded_floor_router_moderate_max_branch_jaccard),
        "--dual-guarded-floor-router-strong-ratio", str(args.dual_guarded_floor_router_strong_ratio),
        "--dual-guarded-floor-router-moderate-ratio", str(args.dual_guarded_floor_router_moderate_ratio),
        "--dual-guarded-floor-router-strong-accept-mode", str(args.dual_guarded_floor_router_strong_accept_mode),
        "--dual-guarded-floor-router-moderate-accept-mode", str(args.dual_guarded_floor_router_moderate_accept_mode),
        "--dual-state-profile-state-recognition-v2-enabled" if bool(args.dual_state_profile_state_recognition_v2_enabled) else "--no-dual-state-profile-state-recognition-v2-enabled",
        "--dual-state-profile-moderate-size-ratio-min", str(args.dual_state_profile_moderate_size_ratio_min),
        "--dual-state-profile-moderate-branch-jaccard-max", str(args.dual_state_profile_moderate_branch_jaccard_max),
        "--dual-state-profile-moderate-over-rate-min", str(args.dual_state_profile_moderate_over_rate_min),
        "--dual-state-profile-moderate-density-delta-min", str(args.dual_state_profile_moderate_density_delta_min),
        "--dual-state-profile-moderate-entropy-min", str(args.dual_state_profile_moderate_entropy_min),
        "--dual-state-profile-stable-size-ratio-low", str(args.dual_state_profile_stable_size_ratio_low),
        "--dual-state-profile-stable-size-ratio-high", str(args.dual_state_profile_stable_size_ratio_high),
        "--dual-state-profile-stable-branch-jaccard-min", str(args.dual_state_profile_stable_branch_jaccard_min),
        "--dual-state-profile-stable-tight-size-ratio-low", str(args.dual_state_profile_stable_tight_size_ratio_low),
        "--dual-state-profile-stable-tight-size-ratio-high", str(args.dual_state_profile_stable_tight_size_ratio_high),
        "--dual-state-profile-stable-tight-branch-jaccard-min", str(args.dual_state_profile_stable_tight_branch_jaccard_min),
        "--dual-state-profile-stable-entropy-min", str(args.dual_state_profile_stable_entropy_min),
        "--dual-state-profile-stable-density-delta-abs-max", str(args.dual_state_profile_stable_density_delta_abs_max),
        "--dual-state-profile-strong-size-ratio-min", str(args.dual_state_profile_strong_size_ratio_min),
        "--dual-state-profile-strong-branch-jaccard-max", str(args.dual_state_profile_strong_branch_jaccard_max),
        "--dual-state-profile-shift-density-delta-min", str(args.dual_state_profile_shift_density_delta_min),
        "--dual-state-profile-shift-branch-jaccard-min", str(args.dual_state_profile_shift_branch_jaccard_min),
        "--dual-state-profile-strong-alpha-bias", str(args.dual_state_profile_strong_alpha_bias),
        "--dual-state-profile-moderate-alpha-bias", str(args.dual_state_profile_moderate_alpha_bias),
        "--dual-state-profile-boundary-alpha-bias", str(args.dual_state_profile_boundary_alpha_bias),
        "--dual-state-profile-shift-alpha-bias", str(args.dual_state_profile_shift_alpha_bias),
        "--dual-state-profile-stable-alpha-bias", str(args.dual_state_profile_stable_alpha_bias),
        "--dual-state-profile-core-weight-strong", str(args.dual_state_profile_core_weight_strong),
        "--dual-state-profile-core-weight-moderate", str(args.dual_state_profile_core_weight_moderate),
        "--dual-state-profile-core-weight-boundary", str(args.dual_state_profile_core_weight_boundary),
        "--dual-state-profile-core-weight-shift", str(args.dual_state_profile_core_weight_shift),
        "--dual-state-profile-core-weight-stable", str(args.dual_state_profile_core_weight_stable),
        "--dual-state-profile-core-max-nodes", str(args.dual_state_profile_core_max_nodes),
        "--dual-state-profile-core-source-mode", str(args.dual_state_profile_core_source_mode),
        "--dual-state-profile-core-agreement-topk", str(args.dual_state_profile_core_agreement_topk),
        "--dual-state-profile-core-source-mode-strong", str(args.dual_state_profile_core_source_mode_strong),
        "--dual-state-profile-core-source-mode-moderate", str(args.dual_state_profile_core_source_mode_moderate),
        "--dual-state-profile-core-source-mode-boundary", str(args.dual_state_profile_core_source_mode_boundary),
        "--dual-state-profile-core-source-mode-shift", str(args.dual_state_profile_core_source_mode_shift),
        "--dual-state-profile-core-source-mode-stable", str(args.dual_state_profile_core_source_mode_stable),
        "--dual-state-profile-core-min-agreement-nodes", str(args.dual_state_profile_core_min_agreement_nodes),
        "--dual-state-profile-score-source-enabled" if bool(args.dual_state_profile_score_source_enabled) else "--no-dual-state-profile-score-source-enabled",
        "--dual-state-profile-action-isolation-enabled" if bool(args.dual_state_profile_action_isolation_enabled) else "--no-dual-state-profile-action-isolation-enabled",
        "--dual-state-profile-action-isolation-scope", str(args.dual_state_profile_action_isolation_scope),
        "--dual-state-profile-score-mode-strong", str(args.dual_state_profile_score_mode_strong),
        "--dual-state-profile-score-mode-moderate", str(args.dual_state_profile_score_mode_moderate),
        "--dual-state-profile-score-mode-boundary", str(args.dual_state_profile_score_mode_boundary),
        "--dual-state-profile-score-mode-shift", str(args.dual_state_profile_score_mode_shift),
        "--dual-state-profile-score-mode-stable", str(args.dual_state_profile_score_mode_stable),
        "--dual-state-profile-score-residual-scale-strong", str(args.dual_state_profile_score_residual_scale_strong),
        "--dual-state-profile-score-residual-scale-moderate", str(args.dual_state_profile_score_residual_scale_moderate),
        "--dual-state-profile-score-residual-scale-boundary", str(args.dual_state_profile_score_residual_scale_boundary),
        "--dual-state-profile-score-residual-scale-shift", str(args.dual_state_profile_score_residual_scale_shift),
        "--dual-state-profile-score-residual-scale-stable", str(args.dual_state_profile_score_residual_scale_stable),
        "--dual-state-profile-score-recall-weight-moderate", str(args.dual_state_profile_score_recall_weight_moderate),
        "--dual-state-profile-score-consensus-weight-boundary", str(args.dual_state_profile_score_consensus_weight_boundary),
        "--dual-state-profile-score-adapt-bonus-shift", str(args.dual_state_profile_score_adapt_bonus_shift),
        "--dual-hierarchical-boundary-state-enabled" if bool(args.dual_hierarchical_boundary_state_enabled) else "--no-dual-hierarchical-boundary-state-enabled",
        "--dual-boundary-plateau-relative-quality-threshold", str(args.dual_boundary_plateau_relative_quality_threshold),
        "--dual-boundary-plateau-precision-extra-penalty", str(args.dual_boundary_plateau_precision_extra_penalty),
        "--dual-boundary-plateau-ceiling-enabled" if bool(args.dual_boundary_plateau_ceiling_enabled) else "--no-dual-boundary-plateau-ceiling-enabled",
        "--dual-boundary-plateau-ceiling-ratio", str(args.dual_boundary_plateau_ceiling_ratio),
        "--dual-boundary-plateau-ceiling-rel-quality-tolerance", str(args.dual_boundary_plateau_ceiling_rel_quality_tolerance),
        "--dual-boundary-plateau-ceiling-rel-density-tolerance", str(args.dual_boundary_plateau_ceiling_rel_density_tolerance),
        "--dual-state-profile-core-reliability-enabled" if bool(args.dual_state_profile_core_reliability_enabled) else "--no-dual-state-profile-core-reliability-enabled",
        "--dual-state-profile-core-reliability-min", str(args.dual_state_profile_core_reliability_min),
        "--dual-state-profile-core-reliability-min-nodes", str(args.dual_state_profile_core_reliability_min_nodes),
        "--dual-state-profile-core-reliability-agreement-weight", str(args.dual_state_profile_core_reliability_agreement_weight),
        "--dual-state-profile-core-reliability-count-weight", str(args.dual_state_profile_core_reliability_count_weight),
        "--dual-state-profile-core-reliability-query-sim-weight", str(args.dual_state_profile_core_reliability_query_sim_weight),
        "--dual-state-profile-structure-weight-moderate", str(args.dual_state_profile_structure_weight_moderate),
        "--dual-state-profile-structure-weight-boundary", str(args.dual_state_profile_structure_weight_boundary),
        "--dual-state-profile-structure-weight-strong", str(args.dual_state_profile_structure_weight_strong),
        "--dual-state-profile-structure-weight-shift", str(args.dual_state_profile_structure_weight_shift),
        "--dual-state-profile-structure-weight-stable", str(args.dual_state_profile_structure_weight_stable),
        "--dual-state-profile-structure-core-topk", str(args.dual_state_profile_structure_core_topk),
        "--dual-state-profile-moderate-recall-bonus", str(args.dual_state_profile_moderate_recall_bonus),
        "--dual-state-profile-boundary-quality-penalty", str(args.dual_state_profile_boundary_quality_penalty),
        "--dual-state-profile-shift-adapt-bonus", str(args.dual_state_profile_shift_adapt_bonus),
        "--dual-state-profile-strong-alpha-min", str(args.dual_state_profile_strong_alpha_min),
        "--dual-state-profile-moderate-alpha-min", str(args.dual_state_profile_moderate_alpha_min),
        "--dual-state-profile-moderate-alpha-max", str(args.dual_state_profile_moderate_alpha_max),
        "--dual-state-profile-boundary-alpha-max", str(args.dual_state_profile_boundary_alpha_max),
        "--dual-state-profile-shift-alpha-min", str(args.dual_state_profile_shift_alpha_min),
        "--dual-state-profile-strong-floor-ratio", str(args.dual_state_profile_strong_floor_ratio),
        "--dual-state-profile-moderate-floor-ratio", str(args.dual_state_profile_moderate_floor_ratio),
        "--dual-state-profile-boundary-floor-ratio", str(args.dual_state_profile_boundary_floor_ratio),
        "--dual-state-profile-shift-floor-ratio", str(args.dual_state_profile_shift_floor_ratio),
        "--dual-state-profile-stable-floor-ratio", str(args.dual_state_profile_stable_floor_ratio),
        "--dual-state-profile-strong-accept-mode", str(args.dual_state_profile_strong_accept_mode),
        "--dual-state-profile-moderate-accept-mode", str(args.dual_state_profile_moderate_accept_mode),
        "--dual-state-profile-boundary-accept-mode", str(args.dual_state_profile_boundary_accept_mode),
        "--dual-state-profile-shift-accept-mode", str(args.dual_state_profile_shift_accept_mode),
        "--dual-state-profile-stable-accept-mode", str(args.dual_state_profile_stable_accept_mode),
    ]
    if args.multi_view_search:
        export_cmd.append("--multi-view-search")
    if args.dual_branch_score:
        export_cmd.append("--dual-branch-score")
        if args.dual_target_state_gate:
            export_cmd.append("--dual-target-state-gate")
        if args.dual_graph_bias_enabled:
            export_cmd.append("--dual-graph-bias-enabled")
        if args.dual_guarded_floor_enabled:
            export_cmd.append("--dual-guarded-floor-enabled")
        if args.dual_guarded_floor_target_trigger:
            export_cmd.append("--dual-guarded-floor-target-trigger")
        if args.dual_guarded_floor_router_enabled:
            export_cmd.append("--dual-guarded-floor-router-enabled")
        if args.dual_state_profile_router_enabled:
            export_cmd.append("--dual-state-profile-router-enabled")
        if args.dual_state_profile_alpha_enabled:
            export_cmd.append("--dual-state-profile-alpha-enabled")
        if args.dual_state_profile_search_enabled:
            export_cmd.append("--dual-state-profile-search-enabled")
        if args.dual_state_profile_alpha_bounds_enabled:
            export_cmd.append("--dual-state-profile-alpha-bounds-enabled")
        if args.dual_state_profile_floor_enabled:
            export_cmd.append("--dual-state-profile-floor-enabled")
        if args.dual_state_profile_core_proto_enabled:
            export_cmd.append("--dual-state-profile-core-proto-enabled")
        if args.dual_state_profile_core_source_by_state_enabled:
            export_cmd.append("--dual-state-profile-core-source-by-state-enabled")
        if args.dual_state_profile_core_strict_agreement:
            export_cmd.append("--dual-state-profile-core-strict-agreement")
        if args.dual_state_profile_structure_enabled:
            export_cmd.append("--dual-state-profile-structure-enabled")
        if str(args.dual_preserved_checkpoint_source) == "epoch0":
            epoch0_ckpt = Path(args.output_dir) / "joint_source_train" / "epoch_checkpoints" / "epoch_0000.pt"
            if not epoch0_ckpt.exists():
                raise SystemExit(f"--dual-preserved-checkpoint-source epoch0 requires {epoch0_ckpt}; use --eval-before-training")
            export_cmd.extend(["--dual-preserved-checkpoint", str(epoch0_ckpt)])
    if not args.write_dual_branch_diagnostics:
        export_cmd.append("--no-write-dual-branch-diagnostics")
    if not args.candidate_canonical_dominance_guard:
        export_cmd.append("--no-candidate-canonical-dominance-guard")
    if not args.write_candidate_diagnostics:
        export_cmd.append("--no-write-candidate-diagnostics")
    if args.seed is not None:
        export_cmd.extend(["--seed", str(args.seed)])
    if args.deterministic:
        export_cmd.append("--deterministic")
    if args.adaptive_ratio_guidance:
        export_cmd.append("--adaptive-ratio-guidance")
    if args.adaptive_fusion:
        export_cmd.append("--adaptive-fusion")
    if args.t_eval is not None:
        export_cmd.extend(["--t-eval", str(args.t_eval)])
    cmd_json = log_prefix.with_suffix(".fixed_export_command.json")
    cmd_json.parent.mkdir(parents=True, exist_ok=True)
    cmd_json.write_text(json.dumps({
        "protocol": "official-fixed-export-command-v1",
        "target": target,
        "epoch": int(epoch),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": file_sha256(ckpt) if ckpt.exists() else None,
        "cwd": str(PROJECT_ROOT),
        "command": export_cmd,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rc = run(export_cmd, cwd=PROJECT_ROOT, log_path=log_prefix.with_suffix(".fixed_export.log"))
    if rc != 0:
        raise SystemExit(f"fixed export failed target={target} epoch={epoch} rc={rc}; see {log_prefix.with_suffix('.fixed_export.log')}")

    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_json = eval_dir / "offline_eval.json"
    eval_csv = eval_dir / "offline_eval.csv"
    eval_cmd = [
        py, "tools/offline_evaluate_predictions_ofacs.py",
        "--dataset-root", args.dataset_root,
        "--graphs", target,
        "--predictions-root", str(pred_dir.parent),
        "--output-json", str(eval_json),
        "--output-csv", str(eval_csv),
        "--allow-unseen-ground-truth-eval",
    ]
    rc_eval = run(eval_cmd, cwd=PROJECT_ROOT, log_path=log_prefix.with_suffix(".offline_eval.log"))
    if rc_eval != 0:
        raise SystemExit(f"offline eval failed target={target} epoch={epoch} rc={rc_eval}; see {log_prefix.with_suffix('.offline_eval.log')}")

    rec = _read_eval_json(eval_json)
    rec.update({
        "target": target,
        "target_display": aliases.get(target, target),
        "epoch": int(epoch),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": file_sha256(ckpt) if ckpt.exists() else None,
        "prediction_dir": str(pred_dir),
        "offline_eval_json": str(eval_json),
        "offline_eval_csv": str(eval_csv),
        "fixed_export_command_json": str(cmd_json),
        "fixed_export_command_sha256": file_sha256(cmd_json),
    })
    if not args.keep_epoch_predictions:
        shards = pred_dir / "fixed_embedding_shards"
        if shards.exists():
            shutil.rmtree(shards, ignore_errors=True)
    return rec


def _cdss_train_cli_args(args: argparse.Namespace) -> list[str]:
    """Resolve every CDSS-related trainer argument after preset overrides.

    Keeping the mapping in one place prevents a paper-table run from silently
    omitting fields that are present in the runner namespace.
    """
    values = [
        "--cdss-loss-weight", str(args.cdss_loss_weight),
        "--cdss-temperature", str(args.cdss_temperature),
        "--cdss-core-topk", str(args.cdss_core_topk),
        "--cdss-boundary-pos-topk", str(args.cdss_boundary_pos_topk),
        "--cdss-hard-neg-topk", str(args.cdss_hard_neg_topk),
        "--cdss-under-recall", str(args.cdss_under_recall),
        "--cdss-moderate-recall", str(args.cdss_moderate_recall),
        "--cdss-boundary-precision", str(args.cdss_boundary_precision),
        "--cdss-stable-recall", str(args.cdss_stable_recall),
        "--cdss-enabled-states", str(args.cdss_enabled_states),
        "--cdss-aggregation", str(args.cdss_aggregation),
        "--cdss-balance-min", str(args.cdss_balance_min),
        "--cdss-balance-max", str(args.cdss_balance_max),
        "--cdss-distance-mode", str(args.cdss_distance_mode),
        "--cdss-separation-mode", str(args.cdss_separation_mode),
        "--cdss-absolute-margin", str(args.cdss_absolute_margin),
        "--cdss-absolute-separation-cap", str(args.cdss_absolute_separation_cap),
        "--cdss-relative-margin", str(args.cdss_relative_margin),
        "--loss-gradient-diagnostics-every", str(args.loss_gradient_diagnostics_every),
    ]
    if bool(args.enable_controlled_error_state_synthesis_loss):
        values.append("--enable-controlled-error-state-synthesis-loss")
    return values


def main() -> None:
    args = parse_args()
    if not args.allow_unseen_ground_truth_eval:
        raise SystemExit("senior paper target-best protocol requires --allow-unseen-ground-truth-eval")
    sources = _parse_list(args.source_graphs)
    targets = _parse_list(args.target_graphs)
    aliases = _parse_alias(args.target_name_alias)
    guided_modes_requested = (
        args.score_embedding in {"guided_refined", "adaptive_guided_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}
        or (
            bool(args.dual_branch_score)
            and (
                args.dual_preserved_score_embedding in {"guided_refined", "adaptive_guided_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}
                or args.dual_adapted_score_embedding in {"guided_refined", "adaptive_guided_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}
            )
        )
    )
    if guided_modes_requested and not (
        bool(args.enable_density_ratio_guidance)
        or str(args.target_evidence_guidance_mode) != "off"
    ):
        raise SystemExit(
            "guided score embeddings require either learned density-ratio guidance "
            "or strict parameter-free target evidence guidance"
        )
    if str(args.target_evidence_guidance_mode) != "off" and bool(args.enable_density_ratio_guidance):
        raise SystemExit(
            "strict target evidence guidance and learned target density-ratio guidance "
            "are mutually exclusive"
        )
    if str(args.target_evidence_guidance_mode) != "off" and bool(args.enable_target_alignment_loss):
        raise SystemExit(
            "strict target evidence guidance forbids target-side alignment training"
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_resolved_args.json").write_text(
        json.dumps(namespace_jsonable(args), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "paper_table_run_config.json").write_text(json.dumps({
        "protocol": "OFA-CS-V5-SeniorExact-official-paper-table-joint-targetbest-v1",
        "official_metric_contract": "official_f1=senior_micro_f1; official_jaccard=senior_jaccard; macro metrics are diagnostics only",
        "source_graphs": sources,
        "target_graphs": targets,
        "training_target_graphs": _training_targets(args, targets),
        "target_aliases": aliases,
        "source_training_mode": "joint",
        "source_weighting": args.source_weighting,
        "source_repeat_weights": args.source_repeat_weights,
        "optimizer": args.optimizer,
        "lr": float(args.lr),
        "grad_clip_norm": float(args.grad_clip_norm),
        "target_guidance_schedule": str(args.target_guidance_schedule),
        "ratio_guidance_sampling_seed": None if args.ratio_guidance_sampling_seed is None else int(args.ratio_guidance_sampling_seed),
        "lr_scheduler": args.lr_scheduler,
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "attention_dropout": float(args.attention_dropout),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "diffusion_T": int(args.diffusion_T),
        "gamma": float(args.gamma),
        "senior_contrastive_weight": float(args.senior_contrastive_weight),
        "diffusion_weight": float(args.diffusion_weight),
        "senior_cache_required": True,
        "eval_target_every_epoch": True,
        "reports_best_by_f1_and_jac": True,
        "eval_before_training": bool(args.eval_before_training),
        "score_embedding": str(args.score_embedding),
        "score_calibration": str(args.score_calibration),
        "multi_view_search": bool(args.multi_view_search),
        "candidate_views": str(args.candidate_views),
        "candidate_selection": str(args.candidate_selection),
        "dual_branch_score": bool(args.dual_branch_score),
        "dual_preserved_score_embedding": str(args.dual_preserved_score_embedding),
        "dual_adapted_score_embedding": str(args.dual_adapted_score_embedding),
        "dual_preserved_checkpoint_source": str(args.dual_preserved_checkpoint_source),
        "dual_alpha_mode": str(args.dual_alpha_mode),
        "dual_fixed_alpha": float(args.dual_fixed_alpha),
        "dual_search_profile": str(args.dual_search_profile),
        "candidate_structural_quality_weight": float(args.candidate_structural_quality_weight),
        "candidate_canonical_view": str(args.candidate_canonical_view),
        "candidate_canonical_dominance_guard": bool(args.candidate_canonical_dominance_guard),
        "seed": None if args.seed is None else int(args.seed),
        "adaptive_ratio_guidance": bool(args.adaptive_ratio_guidance),
        "adaptive_fusion": bool(args.adaptive_fusion),
        "t_eval": None if args.t_eval is None else int(args.t_eval),
        "enable_prefix_loss": bool(args.enable_prefix_loss),
        "prefix_loss_weight": float(args.prefix_loss_weight),
        "prefix_on": str(args.prefix_on),
        "prefix_fp_topk": int(args.prefix_fp_topk),
        "prefix_precision_loss_weight": float(args.prefix_precision_loss_weight),
        "prefix_adaptive_precision": bool(args.prefix_adaptive_precision),
        "prefix_precision_over_ratio": float(args.prefix_precision_over_ratio),
        "prefix_pairwise_boundary_loss": bool(args.prefix_pairwise_boundary_loss),
        "prefix_fn_loss_weight": float(args.prefix_fn_loss_weight),
        "prefix_state_aware_loss": bool(args.prefix_state_aware_loss),
        "prefix_stratified_query_sampling": bool(args.prefix_stratified_query_sampling),
        "enable_structural_boundary_loss": bool(args.enable_structural_boundary_loss),
        "structural_boundary_weight": float(args.structural_boundary_weight),
        "structural_boundary_hop": int(args.structural_boundary_hop),
        "structural_neg_topk": int(args.structural_neg_topk),
        "enable_size_loss": bool(args.enable_size_loss),
        "enable_ranking_preservation_loss": bool(args.enable_ranking_preservation_loss),
        "ranking_preservation_weight": float(args.ranking_preservation_weight),
        "ranking_preservation_mode": str(args.ranking_preservation_mode),
        "enable_density_ratio_guidance": bool(args.enable_density_ratio_guidance),
        "ratio_guidance_weight": float(args.ratio_guidance_weight),
        "ratio_guidance_clip": float(args.ratio_guidance_clip),
        "ratio_guidance_sampling_seed": None if args.ratio_guidance_sampling_seed is None else int(args.ratio_guidance_sampling_seed),
        "target_evidence_guidance_mode": str(args.target_evidence_guidance_mode),
        "target_evidence_fit_chunk_size": int(args.target_evidence_fit_chunk_size),
        "target_evidence_variance_floor": float(args.target_evidence_variance_floor),
        "target_evidence_variance_shrinkage": float(args.target_evidence_variance_shrinkage),
        "target_evidence_confidence_scale": float(args.target_evidence_confidence_scale),
        "target_evidence_max_relative_update": float(args.target_evidence_max_relative_update),
        "strict_one_for_all_source_training": bool(
            str(args.target_evidence_guidance_mode) != "off"
        ),
        "preset_override_json": str(args.preset_override_json),
        "preset_overrides_applied": dict(getattr(args, "preset_overrides_applied", {})),
        "deterministic": bool(args.deterministic),
        "deterministic_strict": bool(args.deterministic_strict),
        "enable_state_conditioned_community_pic_loss": bool(args.enable_state_conditioned_community_pic_loss),
        "community_pic_loss_weight": float(args.community_pic_loss_weight),
        "community_pic_state_top_fraction": float(args.community_pic_state_top_fraction),
        "enable_target_alignment_loss": bool(args.enable_target_alignment_loss),
        "alignment_loss_weight": float(args.alignment_loss_weight),
    }, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")

    py = sys.executable
    train_dir = out / "joint_source_train"
    # The strict one-for-all path evaluates the requested target graphs after
    # training but never exposes them to the trainer.  Legacy learned guidance
    # remains available only for historical reproduction.
    training_targets = _training_targets(args, targets)
    train_cmd = [
        py, "tools/train_senior_exact_joint_ofacs.py",
        "--dataset-root", args.dataset_root,
        "--source-graphs", ",".join(sources),
        "--target-graphs", ",".join(training_targets),
        "--output-dir", str(train_dir),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--dims", str(args.dims),
        "--hops", str(args.hops),
        "--local-topk", str(args.local_topk),
        "--pos-count", str(args.pos_count),
        "--neg-count", str(args.neg_count),
        "--hidden-dim", str(args.hidden_dim),
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads),
        "--dropout", str(args.dropout),
        "--attention-dropout", str(args.attention_dropout),
        "--readout", str(args.readout),
        "--diffusion-T", str(args.diffusion_T),
        "--gamma", str(args.gamma),
        "--senior-contrastive-weight", str(args.senior_contrastive_weight),
        "--diffusion-weight", str(args.diffusion_weight),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--optimizer", str(args.optimizer),
        "--target-guidance-schedule", str(args.target_guidance_schedule),
        "--grad-clip-norm", str(args.grad_clip_norm),
        "--lr-scheduler", str(args.lr_scheduler),
        "--warmup-updates", str(args.warmup_updates),
        "--tot-updates", str(args.tot_updates),
        "--end-lr", str(args.end_lr),
        "--lr-power", str(args.lr_power),
        "--batch-size", str(args.batch_size),
        "--source-weighting", args.source_weighting,
        "--source-repeat-weights", args.source_repeat_weights,
        "--max-link-loss-nodes", str(args.max_link_loss_nodes),
        "--prefix-loss-weight", str(args.prefix_loss_weight),
        "--prefix-on", str(args.prefix_on),
        "--prefix-margin", str(args.prefix_margin),
        "--prefix-query-limit", str(args.prefix_query_limit),
        "--prefix-neg-topk", str(args.prefix_neg_topk),
        "--prefix-pos-bottomk", str(args.prefix_pos_bottomk),
        "--prefix-fp-topk", str(args.prefix_fp_topk),
        "--prefix-precision-loss-weight", str(args.prefix_precision_loss_weight),
        "--prefix-precision-over-ratio", str(args.prefix_precision_over_ratio),
        "--prefix-fn-loss-weight", str(args.prefix_fn_loss_weight),
        "--prefix-state-candidate-limit", str(args.prefix_state_candidate_limit),
        "--prefix-state-over-ratio", str(args.prefix_state_over_ratio),
        "--prefix-state-under-fn-ratio", str(args.prefix_state_under_fn_ratio),
        "--prefix-state-mixed-gap", str(args.prefix_state_mixed_gap),
        "--prefix-state-precision-mult", str(args.prefix_state_precision_mult),
        "--prefix-state-size-mult", str(args.prefix_state_size_mult),
        "--prefix-state-fn-mult", str(args.prefix_state_fn_mult),
        "--prefix-state-pairwise-mult", str(args.prefix_state_pairwise_mult),
        "--prefix-state-hard-positive-mult", str(args.prefix_state_hard_positive_mult),
        "--structural-boundary-weight", str(args.structural_boundary_weight),
        "--structural-boundary-hop", str(args.structural_boundary_hop),
        "--structural-neg-topk", str(args.structural_neg_topk),
        "--size-loss-weight", str(args.size_loss_weight),
        "--size-over-tolerance", str(args.size_over_tolerance),
        "--ranking-preservation-weight", str(args.ranking_preservation_weight),
        "--ranking-preservation-mode", str(args.ranking_preservation_mode),
        "--ratio-guidance-weight", str(args.ratio_guidance_weight),
        "--ratio-guidance-clip", str(args.ratio_guidance_clip),
        "--ratio-guidance-lr", str(args.ratio_guidance_lr),
        "--ratio-guidance-hidden-dim", str(args.ratio_guidance_hidden_dim),
        "--ratio-guidance-time-emb-dim", str(args.ratio_guidance_time_emb_dim),
        "--ratio-guidance-max-nodes", str(args.ratio_guidance_max_nodes),
        "--alignment-loss-weight", str(args.alignment_loss_weight),
        "--alignment-max-nodes", str(args.alignment_max_nodes),
        "--alignment-coral-weight", str(args.alignment_coral_weight),
        "--community-pic-loss-weight", str(args.community_pic_loss_weight),
        "--community-pic-temperature", str(args.community_pic_temperature),
        "--community-pic-topk-core", str(args.community_pic_topk_core),
        "--community-pic-hard-neg-topk", str(args.community_pic_hard_neg_topk),
        "--community-pic-state-top-fraction", str(args.community_pic_state_top_fraction),
        "--save-every-epoch",
    ]
    train_cmd.extend(_cdss_train_cli_args(args))
    if args.ratio_guidance_sampling_seed is not None:
        train_cmd.extend(["--ratio-guidance-sampling-seed", str(args.ratio_guidance_sampling_seed)])
    if args.seed is not None:
        train_cmd.extend(["--seed", str(args.seed)])
    if args.deterministic:
        train_cmd.append("--deterministic")
    if args.deterministic_strict:
        train_cmd.append("--deterministic-strict")
    if bool(args.stage_timing):
        train_cmd.append("--stage-timing")
    if args.enable_link_loss:
        train_cmd.append("--enable-link-loss")
    if args.enable_prefix_loss:
        train_cmd.append("--enable-prefix-loss")
    if args.prefix_adaptive_precision:
        train_cmd.append("--prefix-adaptive-precision")
    if args.prefix_pairwise_boundary_loss:
        train_cmd.append("--prefix-pairwise-boundary-loss")
    if args.prefix_state_aware_loss:
        train_cmd.append("--prefix-state-aware-loss")
    if args.prefix_stratified_query_sampling:
        train_cmd.append("--prefix-stratified-query-sampling")
    if args.enable_structural_boundary_loss:
        train_cmd.append("--enable-structural-boundary-loss")
    if args.enable_size_loss:
        train_cmd.append("--enable-size-loss")
    if args.enable_ranking_preservation_loss:
        train_cmd.append("--enable-ranking-preservation-loss")
    if args.enable_density_ratio_guidance:
        train_cmd.append("--enable-density-ratio-guidance")
    if args.enable_target_alignment_loss:
        train_cmd.append("--enable-target-alignment-loss")
    if args.enable_state_conditioned_community_pic_loss:
        train_cmd.append("--enable-state-conditioned-community-pic-loss")
    if args.eval_before_training:
        train_cmd.append("--save-initial-checkpoint")
    rc_train = run(train_cmd, cwd=PROJECT_ROOT, log_path=out / "logs" / "joint_train.log")
    if rc_train != 0:
        raise SystemExit(f"joint train failed rc={rc_train}; see {out/'logs'/'joint_train.log'}")

    epoch_dir = train_dir / "epoch_checkpoints"
    end_epoch = int(args.eval_end_epoch or args.epochs)
    eval_every = max(1, int(args.eval_every))
    epoch_metrics: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    start_epoch = 0 if args.eval_before_training else max(1, int(args.eval_start_epoch))
    for epoch in range(start_epoch, end_epoch + 1):
        if (epoch - int(args.eval_start_epoch)) % eval_every != 0:
            continue
        ckpt = epoch_dir / f"epoch_{epoch:04d}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing epoch checkpoint: {ckpt}")
        for target in targets:
            rec = _export_and_eval(args, target=target, ckpt=ckpt, epoch=epoch, out=out, aliases=aliases)
            epoch_metrics[target].append(rec)
            print(json.dumps({"target": target, "epoch": epoch, "official_f1": rec.get("official_f1"), "official_jaccard": rec.get("official_jaccard"), "senior_micro_f1": rec.get("senior_micro_f1"), "senior_jaccard": rec.get("senior_jaccard")}, ensure_ascii=False, sort_keys=True))

    best_f1_rows: list[dict[str, Any]] = []
    best_jac_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for target, rows in epoch_metrics.items():
        if not rows:
            continue
        for r in rows:
            all_rows.append(r)
        best_f1 = max(rows, key=lambda r: _metric(r, "official_f1"))
        best_jac = max(rows, key=lambda r: _metric(r, "official_jaccard"))
        b1 = dict(best_f1)
        b1.update({"best_by": "target_f1", "selection_value": float(best_f1["senior_micro_f1"])})
        b2 = dict(best_jac)
        b2.update({"best_by": "target_jaccard", "selection_value": float(best_jac["senior_jaccard"])})
        best_f1_rows.append(b1)
        best_jac_rows.append(b2)
        # Copy best checkpoints for convenience.
        for label, row in [("f1", b1), ("jac", b2)]:
            src = Path(row["checkpoint"])
            dst = out / "best_checkpoints" / target / f"best_by_{label}_epoch_{int(row['epoch']):04d}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    _write_csv(out / "paper_table_epoch_metrics.csv", all_rows)
    _write_csv(out / "paper_table_best_by_f1.csv", best_f1_rows)
    _write_csv(out / "paper_table_best_by_jac.csv", best_jac_rows)

    summary = {
        "protocol": "OFA-CS-V5-SeniorExact-official-paper-table-joint-targetbest-v1",
        "official_metric_contract": "official_f1=senior_micro_f1; official_jaccard=senior_jaccard; macro metrics are diagnostics only",
        "source_graphs": sources,
        "target_graphs": targets,
        "target_aliases": aliases,
        "best_by_f1": best_f1_rows,
        "best_by_jaccard": best_jac_rows,
        "joint_train_dir": str(train_dir),
        "checkpoint_selection_from_target": True,
        "senior_paper_protocol": True,
        "reports_best_by_f1_and_jac": True,
        "eval_before_training": bool(args.eval_before_training),
        "score_embedding": str(args.score_embedding),
        "multi_view_search": bool(args.multi_view_search),
        "candidate_views": str(args.candidate_views),
        "candidate_selection": str(args.candidate_selection),
        "dual_branch_score": bool(args.dual_branch_score),
        "dual_preserved_score_embedding": str(args.dual_preserved_score_embedding),
        "dual_adapted_score_embedding": str(args.dual_adapted_score_embedding),
        "dual_preserved_checkpoint_source": str(args.dual_preserved_checkpoint_source),
        "dual_alpha_mode": str(args.dual_alpha_mode),
        "dual_fixed_alpha": float(args.dual_fixed_alpha),
        "dual_search_profile": str(args.dual_search_profile),
        "t_eval": None if args.t_eval is None else int(args.t_eval),
        "enable_prefix_loss": bool(args.enable_prefix_loss),
        "prefix_loss_weight": float(args.prefix_loss_weight),
        "prefix_on": str(args.prefix_on),
        "prefix_fp_topk": int(args.prefix_fp_topk),
        "prefix_precision_loss_weight": float(args.prefix_precision_loss_weight),
        "prefix_adaptive_precision": bool(args.prefix_adaptive_precision),
        "prefix_precision_over_ratio": float(args.prefix_precision_over_ratio),
        "prefix_pairwise_boundary_loss": bool(args.prefix_pairwise_boundary_loss),
        "prefix_fn_loss_weight": float(args.prefix_fn_loss_weight),
        "enable_size_loss": bool(args.enable_size_loss),
        "enable_ranking_preservation_loss": bool(args.enable_ranking_preservation_loss),
        "ranking_preservation_weight": float(args.ranking_preservation_weight),
        "ranking_preservation_mode": str(args.ranking_preservation_mode),
        "enable_density_ratio_guidance": bool(args.enable_density_ratio_guidance),
        "ratio_guidance_weight": float(args.ratio_guidance_weight),
        "ratio_guidance_clip": float(args.ratio_guidance_clip),
        "enable_target_alignment_loss": bool(args.enable_target_alignment_loss),
        "alignment_loss_weight": float(args.alignment_loss_weight),
    }
    (out / "paper_table_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    status = {"return_code": 0, "output_dir": str(out), "best_by_f1_csv": str(out/"paper_table_best_by_f1.csv"), "best_by_jac_csv": str(out/"paper_table_best_by_jac.csv")}
    (out / "paper_table_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
