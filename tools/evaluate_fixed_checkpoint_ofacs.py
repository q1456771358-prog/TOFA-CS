#!/usr/bin/env python3
"""Fixed-checkpoint prediction export for V5.

This entry point writes prediction files only and refuses metric/gt arguments by
design.  It can either consume a fixed embedding tensor or run a fixed V5 model
checkpoint on a senior-style graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from senior_core.model_v5 import PretrainModelV5
from ofacs.domain_guidance import load_domain_guidance_state, make_structure_condition, node_domain_statistics
from ofacs.embedding_store import save_embedding_chunk_stream
from ofacs.errors import V5DataLeakageError, V5ProtocolError
from ofacs.fixed_checkpoint import load_fixed_checkpoint
from ofacs.inference_loop import (
    run_fixed_inference_from_embedding_shards,
    run_fixed_inference_from_embeddings,
    run_fixed_inference_from_multiview_embedding_shards,
    run_fixed_inference_from_dual_branch_embedding_shards,
)
from ofacs.dual_branch_scoring import DualBranchGateConfig
from ofacs.query_chunk_training import encode_embedding_in_chunks, encode_refined_in_chunks, iter_embedding_chunks, iter_refined_embedding_chunks
from ofacs.multiview_search import MultiViewSelectionConfig
from ofacs.density_ratio_guidance import load_density_ratio_guidance_pair
from ofacs.target_evidence_guidance import fit_target_moment_guidance_pair
from ofacs.resource_preflight import assert_resource_preflight_gate_passed, build_resource_preflight_gate_report, load_resource_preflight_payload
from ofacs.runtime_profile import profile_event
from ofacs.ratio_guidance_diagnostics import RatioGuidanceDiagnostics, RatioGuidanceRuntimeTimer
from ofacs.randomness import set_global_seed
from ofacs.v55_presets import V55_PRESET_CHOICES, apply_v55_preset
from ofacs.preset_override import apply_preset_overrides, namespace_jsonable
from ofacs.senior_data import load_senior_graph_data
from ofacs.senior_compat import load_senior_query_file


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graph-name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embedding-pt", default=None, help="fixed node embeddings [N,H]; no labels")
    p.add_argument("--dataset-root", default=None, help="senior dataset root for fixed model forward")
    p.add_argument("--query-file", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--embedding-export-format", choices=["memory", "shards"], default="shards")
    p.add_argument("--hops", type=int, default=None)
    p.add_argument("--dims", type=int, default=64)
    p.add_argument("--unified-input-dim", type=int, default=None, help="optional explicit guard; fixed export uses checkpoint unified_input_dim by default")
    p.add_argument("--feature-projection-mode", choices=["signed_feature_hash", "identity"], default=None)
    p.add_argument("--feature-projection-hash-seed", type=int, default=None)
    p.add_argument("--local-topk", type=int, default=10)
    p.add_argument("--pos-count", type=int, default=10)
    p.add_argument("--neg-count", type=int, default=10)
    p.add_argument("--no-preprocessed", action="store_true")
    p.add_argument("--senior-cache-mode", choices=["auto", "required", "disabled"], default="auto")
    p.add_argument("--t-eval", type=int, default=None)
    p.add_argument("--score-embedding", choices=["raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"], default="refined", help="embedding view used for query cosine scoring; refined preserves previous default")
    p.add_argument("--score-calibration", choices=["none", "center", "zscore", "local_center", "temperature", "precision_profile", "recall_profile"], default="none", help="label-free per-query score/search profile before senior GlobalSearch")
    p.add_argument("--multi-view-search", action=argparse.BooleanOptionalAction, default=False, help="label-free per-query selection among multiple GlobalSearch score views")
    p.add_argument("--candidate-views", default="guided_refined:none,adaptive_raw_plus_guided_refined:none,guided_refined:local_center,guided_refined:temperature,guided_refined:precision_profile,guided_refined:recall_profile", help="comma list of mode[:profile] candidates for --multi-view-search")
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
    # V5.5 dual-branch preservation/adaptation score path.  This is label-free
    # inference only; target gt remains forbidden here and is used only by the
    # external offline evaluator when explicitly allowed by the paper-table runner.
    p.add_argument("--v55-preset", choices=V55_PRESET_CHOICES, default="off", help="compact V5.5 preset; keeps experiments interpretable and avoids sweeping low-level alpha weights")
    p.add_argument("--preset-override-json", default="", help="JSON object, JSON file path, or @file applied after --v55-preset")
    p.add_argument("--dual-branch-score", action=argparse.BooleanOptionalAction, default=False, help="V5.5 dual-branch preserved/adapted score fusion before GlobalSearch")
    p.add_argument("--dual-preserved-score-embedding", help=argparse.SUPPRESS, choices=["raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"], default="raw")
    p.add_argument("--dual-adapted-score-embedding", help=argparse.SUPPRESS, choices=["raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"], default="guided_refined")
    p.add_argument("--dual-preserved-checkpoint", default=None, help=argparse.SUPPRESS)
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
    p.add_argument("--dual-precision-recall-over-penalty", help=argparse.SUPPRESS, type=float, default=0.05)
    p.add_argument("--dual-overexpansion-router-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-overexpansion-ratio-threshold", help=argparse.SUPPRESS, type=float, default=1.10)
    p.add_argument("--dual-overexpansion-rate-threshold", help=argparse.SUPPRESS, type=float, default=0.35)
    p.add_argument("--dual-overexpansion-precision-extra-penalty", help=argparse.SUPPRESS, type=float, default=0.0)
    p.add_argument("--dual-guarded-ceiling-enabled", help=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-guarded-ceiling-ratio", help=argparse.SUPPRESS, type=float, default=0.92)
    p.add_argument("--dual-guarded-ceiling-rel-quality-tolerance", help=argparse.SUPPRESS, type=float, default=0.01)
    p.add_argument("--dual-guarded-ceiling-rel-density-tolerance", help=argparse.SUPPRESS, type=float, default=0.01)
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
    p.add_argument("--dual-state-profile-strong-accept-mode", help=argparse.SUPPRESS, type=str, default="relative_quality")
    p.add_argument("--dual-state-profile-moderate-accept-mode", help=argparse.SUPPRESS, type=str, default="relative_quality")
    p.add_argument("--dual-state-profile-boundary-accept-mode", help=argparse.SUPPRESS, type=str, default="relative_quality")
    p.add_argument("--dual-state-profile-shift-accept-mode", help=argparse.SUPPRESS, type=str, default="relative_quality")
    p.add_argument("--dual-state-profile-stable-accept-mode", help=argparse.SUPPRESS, type=str, default="relative_quality")
    p.add_argument("--write-dual-branch-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ratio-guidance-weight", type=float, default=None, help="override checkpoint density-ratio reverse guidance weight for guided score modes")
    p.add_argument("--ratio-guidance-clip", type=float, default=None, help="override checkpoint density-ratio gradient clip for guided score modes")
    p.add_argument(
        "--density-ratio-guidance-state",
        default="",
        help="optional target-specific unlabeled density-ratio sidecar; overrides the checkpoint estimator only",
    )
    p.add_argument("--adaptive-ratio-guidance", action=argparse.BooleanOptionalAction, default=False, help="node-adaptive density-ratio guidance strength during reverse denoise")
    p.add_argument("--ratio-guidance-confidence-scale", type=float, default=1.0)
    p.add_argument("--ratio-guidance-min-mult", type=float, default=0.25)
    p.add_argument("--ratio-guidance-max-mult", type=float, default=1.25)
    p.add_argument(
        "--target-evidence-guidance-mode",
        choices=["off", "diagonal_gaussian"],
        default="off",
        help=(
            "strict one-for-all inference guidance; diagonal_gaussian fits a "
            "parameter-free potential from unlabeled target raw embeddings"
        ),
    )
    p.add_argument("--target-evidence-fit-chunk-size", type=int, default=8192)
    p.add_argument("--target-evidence-variance-floor", type=float, default=1e-4)
    p.add_argument("--target-evidence-variance-shrinkage", type=float, default=0.05)
    p.add_argument("--target-evidence-confidence-scale", type=float, default=2.0)
    p.add_argument("--target-evidence-max-relative-update", type=float, default=0.05)
    p.add_argument(
        "--write-ratio-guidance-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write detached density-ratio gradient diagnostics; does not alter inference",
    )
    p.add_argument("--adaptive-fusion", action=argparse.BooleanOptionalAction, default=False, help="adaptive raw/guided fusion for raw_plus guided modes")
    p.add_argument("--fusion-min-guided", type=float, default=0.25)
    p.add_argument("--fusion-max-guided", type=float, default=0.85)
    p.add_argument("--fusion-confidence-scale", type=float, default=8.0)
    p.add_argument("--preflight-report-json", default=None)
    p.add_argument("--require-preflight-pass", action="store_true")
    p.add_argument("--token-cache-dir", default=None)
    p.add_argument("--reuse-token-cache", action="store_true")
    p.add_argument("--no-save-token-cache", action="store_true")
    p.add_argument("--runtime-jsonl", default=None)
    p.add_argument(
        "--profile-stage-synchronize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="synchronize CUDA at named paper-efficiency stage boundaries; use only for the dedicated timing run",
    )
    p.add_argument("--seed", type=int, default=None, help="fixed random seed for reproducible fixed export diagnostics")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--metric", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    apply_v55_preset(args)
    apply_preset_overrides(
        args,
        args.preset_override_json,
        forbidden_keys={"graph_name", "checkpoint", "dataset_root", "output_dir", "gt", "metric"},
    )
    return args


def _paper_stage_event(path: str | Path | None, event: str, args: argparse.Namespace, **extra: object) -> None:
    """Emit a high-resolution stage marker for the paper efficiency audit.

    Synchronization is opt-in so ordinary fixed evaluation remains unchanged.
    The timing-only runner enables it to prevent asynchronous CUDA kernels from
    being charged to a later stage.
    """

    if bool(getattr(args, "profile_stage_synchronize", False)) and torch.cuda.is_available():
        device_text = str(getattr(args, "device", ""))
        if device_text.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device_text))
    profile_event(path, event, **extra)


def _parse_candidate_views(spec: str, *, fallback_mode: str, fallback_calibration: str) -> list[tuple[str, str, str]]:
    raw = str(spec or "").strip()
    if not raw:
        raw = f"{fallback_mode}:{fallback_calibration}"
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for idx, part in enumerate(raw.split(",")):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            mode, cal = part.split(":", 1)
        else:
            mode, cal = part, "none"
        mode = mode.strip()
        cal = cal.strip() or "none"
        if mode not in {"raw", "refined", "guided_refined", "adaptive_guided_refined", "one_step", "raw_plus_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}:
            raise V5ProtocolError(f"unsupported multi-view score mode: {mode}")
        if cal not in {"none", "center", "zscore", "local_center", "temperature", "precision_profile", "recall_profile"}:
            raise V5ProtocolError(f"unsupported multi-view score/search profile: {cal}")
        name = f"{mode}__{cal}"
        if name in seen:
            name = f"{name}__{idx}"
        seen.add(name)
        out.append((name, mode, cal))
    if not out:
        raise V5ProtocolError("--candidate-views produced no candidates")
    return out


def _checkpoint_unified_input_dim(ckpt: dict[str, object]) -> int:
    cfg = dict(ckpt.get("config", {}))
    value = ckpt.get("unified_input_dim", None)
    if value is None:
        value = cfg.get("unified_input_dim", None)
    if value is None:
        value = ckpt.get("input_dim", None)
    if value is None:
        value = cfg.get("input_dim", None)
    if value is None:
        raise V5ProtocolError("checkpoint missing unified_input_dim/input_dim; refusing cross-graph fixed export")
    return int(value)


def _validate_state_input_dim(ckpt: dict[str, object], *, expected_input_dim: int) -> None:
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise V5ProtocolError("checkpoint model_state_dict must be a dict")
    weight = state.get("encoder.att_embeddings_nope.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise V5ProtocolError("checkpoint missing encoder.att_embeddings_nope.weight for input-dim validation")
    state_input_dim = int(weight.shape[1])
    if state_input_dim != int(expected_input_dim):
        raise V5ProtocolError(
            f"checkpoint input layer dim={state_input_dim} does not match unified_input_dim={int(expected_input_dim)}"
        )


def _score_mode_needs_guidance(mode: str) -> bool:
    return str(mode) in {"guided_refined", "adaptive_guided_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}


def _write_ratio_diagnostics(args: argparse.Namespace, collector: RatioGuidanceDiagnostics | None) -> None:
    if collector is None:
        return
    path = Path(args.output_dir) / "density_ratio_guidance_diagnostics.json"
    report = collector.to_dict()
    report.update(
        {
            "graph_name": str(args.graph_name),
            "score_embedding": str(args.dual_adapted_score_embedding if args.dual_branch_score else args.score_embedding),
            "ratio_guidance_weight": args.ratio_guidance_weight,
            "ratio_guidance_clip": args.ratio_guidance_clip,
        }
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_views_need_guidance(spec: str) -> bool:
    guided = {"guided_refined", "adaptive_guided_refined", "raw_plus_guided_refined", "adaptive_raw_plus_guided_refined"}
    for part in str(spec or "").split(","):
        mode = part.strip().split(":", 1)[0].strip()
        if mode in guided:
            return True
    return False


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=bool(args.deterministic))
    if args.gt is not None or args.metric is not None:
        raise V5DataLeakageError("fixed inference refuses gt/metric arguments")
    if (not bool(args.multi_view_search)) and str(args.score_calibration) not in {"none", "center", "zscore"}:
        raise V5ProtocolError("local_center/temperature/precision_profile/recall_profile are multi-view search profiles; use --multi-view-search")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "fixed_inference_resolved_args.json").write_text(
        json.dumps(namespace_jsonable(args), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_jsonl = args.runtime_jsonl or str(Path(args.output_dir) / "runtime_profile.jsonl")
    profile_event(runtime_jsonl, "fixed_inference_start", graph_name=args.graph_name, checkpoint=args.checkpoint, score_embedding=args.score_embedding)
    _maybe_check_resource_preflight(args, runtime_jsonl=runtime_jsonl)
    _paper_stage_event(runtime_jsonl, "paper_stage_checkpoint_load_start", args, graph_name=args.graph_name)
    ckpt = load_fixed_checkpoint(args.checkpoint, map_location="cpu")
    _paper_stage_event(runtime_jsonl, "paper_stage_checkpoint_load_end", args, graph_name=args.graph_name)
    global_node_ids = None
    if args.embedding_pt:
        emb_obj = torch.load(args.embedding_pt, map_location="cpu")
        if isinstance(emb_obj, dict):
            emb = None
            for key in ("embeddings", "node_embeddings", "refined_embedding", "embedding"):
                value = emb_obj.get(key)
                if isinstance(value, torch.Tensor):
                    emb = value
                    break
        else:
            emb = emb_obj
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
            raise RuntimeError("embedding-pt must contain a [N,H] tensor")
        if not args.query_file:
            raise RuntimeError("--query-file is required when using --embedding-pt")
        qbatch = load_senior_query_file(args.query_file, num_nodes=emb.shape[0], graph_name=args.graph_name)
    elif args.dataset_root:
        cfg_dict = dict(ckpt.get("config", {}))
        ckpt_input_dim = _checkpoint_unified_input_dim(ckpt)
        if args.unified_input_dim is not None and int(args.unified_input_dim) != int(ckpt_input_dim):
            raise V5ProtocolError(
                f"--unified-input-dim={int(args.unified_input_dim)} does not match checkpoint unified_input_dim={ckpt_input_dim}"
            )
        hops = int(args.hops if args.hops is not None else cfg_dict.get("hops", 4))
        projection_cfg = cfg_dict.get("feature_projection") if isinstance(cfg_dict.get("feature_projection"), dict) else {}
        projection_mode = args.feature_projection_mode or str(projection_cfg.get("mode", "signed_feature_hash"))
        if projection_mode == "identity" and int(projection_cfg.get("raw_feature_dim", ckpt_input_dim)) != int(ckpt_input_dim):
            # Checkpoints trained with a hashed unified space should not be exported with identity target raw features.
            projection_mode = "signed_feature_hash"
        projection_seed = int(args.feature_projection_hash_seed if args.feature_projection_hash_seed is not None else projection_cfg.get("hash_seed", 1729))
        _paper_stage_event(runtime_jsonl, "paper_stage_graph_and_token_load_start", args, graph_name=args.graph_name)
        graph = load_senior_graph_data(
            args.dataset_root,
            args.graph_name,
            split="unseen_inference",
            purpose="fixed_prediction_export",
            dims=int(args.dims),
            hops=hops,
            local_topk=int(args.local_topk),
            pos_count=int(args.pos_count),
            neg_count=int(args.neg_count),
            require_labels=False,
            load_queries=True,
            prefer_preprocessed=not args.no_preprocessed,
            senior_cache_mode=args.senior_cache_mode,
            token_cache_dir=args.token_cache_dir,
            reuse_token_cache=bool(args.reuse_token_cache),
            save_token_cache_enabled=not bool(args.no_save_token_cache),
            runtime_jsonl=runtime_jsonl,
            unified_input_dim=int(ckpt_input_dim),
            feature_projection_mode=projection_mode,
            feature_projection_hash_seed=projection_seed,
        )
        _paper_stage_event(
            runtime_jsonl,
            "paper_stage_graph_and_token_load_end",
            args,
            graph_name=args.graph_name,
            num_nodes=int(graph.global_token_seqs.shape[0]),
        )
        if graph.query_batch is None:
            raise RuntimeError("fixed inference requires a senior .query file")
        if int(graph.global_token_seqs.shape[-1]) != int(ckpt_input_dim):
            raise V5ProtocolError(
                f"checkpoint unified_input_dim={ckpt_input_dim} but target graph token dim={int(graph.global_token_seqs.shape[-1])}; "
                "refusing silent fallback"
            )
        _validate_state_input_dim(ckpt, expected_input_dim=int(ckpt_input_dim))
        model_cfg = SimpleNamespace(
            hidden_dim=int(cfg_dict.get("hidden_dim", 512)),
            hops=hops,
            n_layers=int(cfg_dict.get("n_layers", cfg_dict.get("n-layers", 1))),
            n_heads=int(cfg_dict.get("n_heads", cfg_dict.get("n-heads", 8))),
            dropout=float(cfg_dict.get("dropout", 0.4)),
            attention_dropout=float(cfg_dict.get("attention_dropout", cfg_dict.get("attention-dropout", 0.4))),
            readout=str(cfg_dict.get("readout", "mean")),
            diffusion_T=int(cfg_dict.get("diffusion_T", cfg_dict.get("diffusion-T", 200))),
            condition_dim=16,
        )
        device = torch.device(args.device)
        _paper_stage_event(runtime_jsonl, "paper_stage_model_restore_start", args, graph_name=args.graph_name)
        model = PretrainModelV5(input_dim=int(ckpt_input_dim), config=model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _paper_stage_event(runtime_jsonl, "paper_stage_model_restore_end", args, graph_name=args.graph_name)
        ratio_guidance_pair = None
        ratio_diagnostics = RatioGuidanceDiagnostics() if bool(args.write_ratio_guidance_diagnostics) else None
        ratio_runtime_timer = RatioGuidanceRuntimeTimer(
            synchronize_cuda=bool(args.profile_stage_synchronize)
        ) if bool(args.profile_stage_synchronize) else None
        ratio_guidance_weight = 0.0
        ratio_guidance_clip = 5.0
        needs_ratio_guidance = (
            _score_mode_needs_guidance(args.score_embedding)
            or (bool(args.multi_view_search) and _candidate_views_need_guidance(args.candidate_views))
            or (bool(args.dual_branch_score) and (_score_mode_needs_guidance(args.dual_preserved_score_embedding) or _score_mode_needs_guidance(args.dual_adapted_score_embedding)))
        )
        if needs_ratio_guidance:
            target_evidence_mode = str(args.target_evidence_guidance_mode)
            if target_evidence_mode != "off":
                if str(args.density_ratio_guidance_state or "").strip():
                    raise V5ProtocolError(
                        "strict target evidence guidance forbids a learned target-specific sidecar"
                    )
                _paper_stage_event(
                    runtime_jsonl,
                    "paper_stage_target_evidence_fit_start",
                    args,
                    graph_name=args.graph_name,
                )
                ratio_guidance_pair, target_evidence_report = fit_target_moment_guidance_pair(
                    model=model,
                    global_token_seqs=graph.global_token_seqs.cpu(),
                    device=device,
                    chunk_size=int(args.target_evidence_fit_chunk_size),
                    variance_floor=float(args.target_evidence_variance_floor),
                    variance_shrinkage=float(args.target_evidence_variance_shrinkage),
                    confidence_scale=float(args.target_evidence_confidence_scale),
                    max_relative_update=float(args.target_evidence_max_relative_update),
                )
                ratio_report = target_evidence_report.to_dict()
                ratio_guidance_weight = float(
                    0.03 if args.ratio_guidance_weight is None else args.ratio_guidance_weight
                )
                ratio_guidance_clip = float(
                    5.0 if args.ratio_guidance_clip is None else args.ratio_guidance_clip
                )
                report_path = Path(args.output_dir) / "target_evidence_guidance_report.json"
                report_path.write_text(
                    json.dumps(
                        {
                            **ratio_report,
                            "graph_name": str(args.graph_name),
                            "guidance_weight": ratio_guidance_weight,
                            "guidance_clip": ratio_guidance_clip,
                            "learned_target_estimator_loaded": False,
                            "checkpoint_density_ratio_state_ignored": bool(
                                "density_ratio_guidance" in ckpt
                            ),
                            "strict_one_for_all": True,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                profile_event(
                    runtime_jsonl,
                    "target_evidence_guidance_fixed_ready",
                    graph_name=args.graph_name,
                    score_embedding=str(args.score_embedding),
                    guidance_weight=ratio_guidance_weight,
                    guidance_clip=ratio_guidance_clip,
                    report=ratio_report,
                    target_optimizer_steps=0,
                    target_trainable_parameters=0,
                    labels_used=False,
                    gt_file_read=False,
                    metrics_computed=False,
                )
                _paper_stage_event(
                    runtime_jsonl,
                    "paper_stage_target_evidence_fit_end",
                    args,
                    graph_name=args.graph_name,
                )
            else:
                _paper_stage_event(runtime_jsonl, "paper_stage_density_ratio_state_load_start", args, graph_name=args.graph_name)
                ratio_state = None
                ratio_state_origin = "checkpoint"
                if str(args.density_ratio_guidance_state or "").strip():
                    sidecar_path = Path(args.density_ratio_guidance_state).expanduser().resolve()
                    if not sidecar_path.is_file():
                        raise V5ProtocolError(f"density-ratio sidecar not found: {sidecar_path}")
                    sidecar_obj = torch.load(sidecar_path, map_location="cpu")
                    if isinstance(sidecar_obj, dict) and isinstance(sidecar_obj.get("density_ratio_guidance"), dict):
                        ratio_state = sidecar_obj["density_ratio_guidance"]
                    elif isinstance(sidecar_obj, dict):
                        ratio_state = sidecar_obj
                    else:
                        raise V5ProtocolError("density-ratio sidecar must contain a mapping")
                    expected_target = str(ratio_state.get("target_graph", "")).lower()
                    actual_target = str(args.graph_name).lower()
                    if expected_target and expected_target != actual_target:
                        raise V5ProtocolError(
                            f"density-ratio sidecar target mismatch: state={expected_target}, requested={actual_target}"
                        )
                    expected_backbone_sha = str(ratio_state.get("backbone_sha256", "")).strip().lower()
                    if not expected_backbone_sha:
                        raise V5ProtocolError("density-ratio sidecar is missing backbone_sha256")
                    actual_backbone_sha = _sha256_file(args.checkpoint)
                    if expected_backbone_sha != actual_backbone_sha:
                        raise V5ProtocolError(
                            "density-ratio sidecar/checkpoint mismatch: "
                            f"state={expected_backbone_sha}, checkpoint={actual_backbone_sha}"
                        )
                    sidecar_max_timestep = ratio_state.get("max_timestep")
                    if sidecar_max_timestep is None:
                        raise V5ProtocolError("density-ratio sidecar is missing max_timestep")
                    if args.t_eval is not None and int(sidecar_max_timestep) != int(args.t_eval):
                        raise V5ProtocolError(
                            "density-ratio sidecar/t_eval mismatch: "
                            f"state={int(sidecar_max_timestep)}, requested={int(args.t_eval)}"
                        )
                    ratio_state_origin = str(sidecar_path)
                elif "density_ratio_guidance" in ckpt:
                    ratio_state = ckpt["density_ratio_guidance"]
                if ratio_state is None:
                    raise V5ProtocolError("guided score embedding requested but no density-ratio guidance state is available")
                ratio_guidance_pair, ratio_report = load_density_ratio_guidance_pair(ratio_state, map_location=device)
                ratio_guidance_weight = float(args.ratio_guidance_weight if args.ratio_guidance_weight is not None else ratio_report.get("guidance_weight", 0.0))
                ratio_guidance_clip = float(args.ratio_guidance_clip if args.ratio_guidance_clip is not None else ratio_report.get("guidance_clip", 5.0))
                profile_event(
                    runtime_jsonl,
                    "density_ratio_guidance_fixed_loaded",
                    graph_name=args.graph_name,
                    score_embedding=str(args.score_embedding),
                    guidance_weight=ratio_guidance_weight,
                    guidance_clip=ratio_guidance_clip,
                    guidance_state_origin=ratio_state_origin,
                    report=ratio_report,
                    labels_used=False,
                    gt_file_read=False,
                    metrics_computed=False,
                )
                _paper_stage_event(runtime_jsonl, "paper_stage_density_ratio_state_load_end", args, graph_name=args.graph_name)
        structure_condition = None
        if "domain_guidance" in ckpt:
            dg_model, dg_mean, dg_std, _dg_cfg, dg_report = load_domain_guidance_state(ckpt["domain_guidance"])
            target_stats = node_domain_statistics(graph.global_token_seqs.cpu(), graph.edge_index.long().cpu())
            structure_condition = make_structure_condition(dg_model, target_stats, mean=dg_mean, std=dg_std, device=device)
            profile_event(
                runtime_jsonl,
                "domain_guidance_fixed_condition_ready",
                graph_name=args.graph_name,
                condition_shape=tuple(int(x) for x in structure_condition.shape),
                condition_norm=float(structure_condition.norm(dim=-1).mean().item()),
                labels_used=False,
                gt_file_read=False,
                metrics_computed=False,
                report=dg_report,
            )
        if args.embedding_export_format == "shards":
            qbatch = graph.query_batch
            global_node_ids = graph.global_node_ids.tolist()
            if bool(args.dual_branch_score):
                preserved_model = model
                preserved_ckpt_path = args.checkpoint
                preserved_ratio_pair = ratio_guidance_pair
                if args.dual_preserved_checkpoint:
                    preserved_ckpt_path = str(args.dual_preserved_checkpoint)
                    p_ckpt = load_fixed_checkpoint(preserved_ckpt_path, map_location="cpu")
                    _validate_state_input_dim(p_ckpt, expected_input_dim=int(ckpt_input_dim))
                    preserved_model = PretrainModelV5(input_dim=int(ckpt_input_dim), config=model_cfg).to(device)
                    preserved_model.load_state_dict(p_ckpt["model_state_dict"])
                    preserved_model.eval()
                    if _score_mode_needs_guidance(args.dual_preserved_score_embedding):
                        if str(args.target_evidence_guidance_mode) != "off":
                            preserved_ratio_pair, _pres_ratio_report = fit_target_moment_guidance_pair(
                                model=preserved_model,
                                global_token_seqs=graph.global_token_seqs.cpu(),
                                device=device,
                                chunk_size=int(args.target_evidence_fit_chunk_size),
                                variance_floor=float(args.target_evidence_variance_floor),
                                variance_shrinkage=float(args.target_evidence_variance_shrinkage),
                                confidence_scale=float(args.target_evidence_confidence_scale),
                                max_relative_update=float(args.target_evidence_max_relative_update),
                            )
                        else:
                            if "density_ratio_guidance" not in p_ckpt:
                                raise V5ProtocolError("dual preserved guided branch requested but preserved checkpoint has no density_ratio_guidance state")
                            preserved_ratio_pair, _pres_ratio_report = load_density_ratio_guidance_pair(p_ckpt["density_ratio_guidance"], map_location=device)
                    else:
                        preserved_ratio_pair = None
                _paper_stage_event(runtime_jsonl, "paper_stage_preserved_embedding_start", args, graph_name=args.graph_name)
                preserved_manifest = save_embedding_chunk_stream(
                    chunks=iter_embedding_chunks(
                        model=preserved_model,
                        global_token_seqs=graph.global_token_seqs.cpu(),
                        device=device,
                        chunk_size=int(args.chunk_size),
                        runtime_jsonl=runtime_jsonl,
                        graph_name=args.graph_name,
                        t_eval=args.t_eval,
                        structure_condition=structure_condition,
                        score_embedding=str(args.dual_preserved_score_embedding),
                        ratio_guidance_pair=preserved_ratio_pair,
                        ratio_guidance_weight=ratio_guidance_weight if _score_mode_needs_guidance(args.dual_preserved_score_embedding) else 0.0,
                        ratio_guidance_clip=ratio_guidance_clip,
                        adaptive_ratio_guidance=bool(args.adaptive_ratio_guidance or args.dual_preserved_score_embedding == "adaptive_guided_refined"),
                        ratio_guidance_confidence_scale=float(args.ratio_guidance_confidence_scale),
                        ratio_guidance_min_mult=float(args.ratio_guidance_min_mult),
                        ratio_guidance_max_mult=float(args.ratio_guidance_max_mult),
                        adaptive_fusion=bool(args.adaptive_fusion or args.dual_preserved_score_embedding == "adaptive_raw_plus_guided_refined"),
                        fusion_min_guided=float(args.fusion_min_guided),
                        fusion_max_guided=float(args.fusion_max_guided),
                        fusion_confidence_scale=float(args.fusion_confidence_scale),
                    ),
                    output_dir=Path(args.output_dir) / "fixed_embedding_shards" / f"dual_preserved_{args.dual_preserved_score_embedding}",
                    graph_name=f"{args.graph_name}_dual_preserved_{args.dual_preserved_score_embedding}",
                    embedding_dim=int(model_cfg.hidden_dim) * 2,
                    num_nodes=int(graph.global_token_seqs.shape[0]),
                    dtype="torch.float32",
                    metadata={
                        "fixed_inference": True,
                        "labels_used": False,
                        "ground_truth_used": False,
                        "metrics_computed": False,
                        "checkpoint": str(preserved_ckpt_path),
                        "score_embedding": str(args.dual_preserved_score_embedding),
                        "dual_branch": "preserved",
                    },
                    runtime_jsonl=runtime_jsonl,
                )
                _paper_stage_event(runtime_jsonl, "paper_stage_preserved_embedding_end", args, graph_name=args.graph_name)
                _paper_stage_event(runtime_jsonl, "paper_stage_adapted_embedding_start", args, graph_name=args.graph_name)
                adapted_manifest = save_embedding_chunk_stream(
                    chunks=iter_embedding_chunks(
                        model=model,
                        global_token_seqs=graph.global_token_seqs.cpu(),
                        device=device,
                        chunk_size=int(args.chunk_size),
                        runtime_jsonl=runtime_jsonl,
                        graph_name=args.graph_name,
                        t_eval=args.t_eval,
                        structure_condition=structure_condition,
                        score_embedding=str(args.dual_adapted_score_embedding),
                        ratio_guidance_pair=ratio_guidance_pair,
                        ratio_guidance_weight=ratio_guidance_weight if _score_mode_needs_guidance(args.dual_adapted_score_embedding) else 0.0,
                        ratio_guidance_clip=ratio_guidance_clip,
                        adaptive_ratio_guidance=bool(args.adaptive_ratio_guidance or args.dual_adapted_score_embedding == "adaptive_guided_refined"),
                        ratio_guidance_confidence_scale=float(args.ratio_guidance_confidence_scale),
                        ratio_guidance_min_mult=float(args.ratio_guidance_min_mult),
                        ratio_guidance_max_mult=float(args.ratio_guidance_max_mult),
                        adaptive_fusion=bool(args.adaptive_fusion or args.dual_adapted_score_embedding == "adaptive_raw_plus_guided_refined"),
                        fusion_min_guided=float(args.fusion_min_guided),
                        fusion_max_guided=float(args.fusion_max_guided),
                        fusion_confidence_scale=float(args.fusion_confidence_scale),
                        guidance_diagnostics=ratio_diagnostics,
                        guidance_runtime_timer=ratio_runtime_timer,
                    ),
                    output_dir=Path(args.output_dir) / "fixed_embedding_shards" / f"dual_adapted_{args.dual_adapted_score_embedding}",
                    graph_name=f"{args.graph_name}_dual_adapted_{args.dual_adapted_score_embedding}",
                    embedding_dim=int(model_cfg.hidden_dim) * 2,
                    num_nodes=int(graph.global_token_seqs.shape[0]),
                    dtype="torch.float32",
                    metadata={
                        "fixed_inference": True,
                        "labels_used": False,
                        "ground_truth_used": False,
                        "metrics_computed": False,
                        "checkpoint": str(args.checkpoint),
                        "score_embedding": str(args.dual_adapted_score_embedding),
                        "dual_branch": "adapted",
                    },
                    runtime_jsonl=runtime_jsonl,
                )
                _paper_stage_event(runtime_jsonl, "paper_stage_adapted_embedding_end", args, graph_name=args.graph_name)
                if ratio_runtime_timer is not None:
                    _paper_stage_event(
                        runtime_jsonl,
                        "paper_stage_density_ratio_guidance_aggregate",
                        args,
                        graph_name=args.graph_name,
                        **ratio_runtime_timer.to_dict(),
                    )
                _write_ratio_diagnostics(args, ratio_diagnostics)
                _paper_stage_event(runtime_jsonl, "paper_stage_state_scoring_search_start", args, graph_name=args.graph_name)
                outputs = run_fixed_inference_from_dual_branch_embedding_shards(
                    graph_name=args.graph_name,
                    preserved_manifest=preserved_manifest,
                    adapted_manifest=adapted_manifest,
                    query_batch=qbatch,
                    output_dir=args.output_dir,
                    checkpoint_path=args.checkpoint,
                    global_node_ids=global_node_ids,
                    chunk_size=args.chunk_size,
                    runtime_jsonl=runtime_jsonl,
                    dual_config=DualBranchGateConfig(
                        enabled=True,
                        alpha_mode=str(args.dual_alpha_mode),
                        fixed_alpha=float(args.dual_fixed_alpha),
                        min_alpha=float(args.dual_min_alpha),
                        max_alpha=float(args.dual_max_alpha),
                        base_alpha=float(args.dual_base_alpha),
                        over_preserved_weight=float(args.dual_over_preserved_weight),
                        adapted_density_weight=float(args.dual_adapted_density_weight),
                        entropy_preserve_weight=float(args.dual_entropy_preserve_weight),
                        disagreement_preserve_weight=float(args.dual_disagreement_preserve_weight),
                        adapted_undersize_weight=float(args.dual_adapted_undersize_weight),
                        adapted_oversize_weight=float(args.dual_adapted_oversize_weight),
                        boundary_gap_adapt_weight=float(args.dual_boundary_gap_adapt_weight),
                        residual_gain=float(args.dual_residual_gain),
                        search_profile=str(args.dual_search_profile),
                        balanced_size_low=float(args.dual_balanced_size_low),
                        balanced_size_high=float(args.dual_balanced_size_high),
                        entropy_threshold=float(args.dual_entropy_threshold),
                        high_entropy_recall_bonus=float(args.dual_high_entropy_recall_bonus),
                        precision_recall_over_penalty=float(args.dual_precision_recall_over_penalty),
                        overexpansion_router_enabled=bool(args.dual_overexpansion_router_enabled),
                        overexpansion_ratio_threshold=float(args.dual_overexpansion_ratio_threshold),
                        overexpansion_rate_threshold=float(args.dual_overexpansion_rate_threshold),
                        overexpansion_precision_extra_penalty=float(args.dual_overexpansion_precision_extra_penalty),
                        guarded_ceiling_enabled=bool(args.dual_guarded_ceiling_enabled),
                        guarded_ceiling_ratio=float(args.dual_guarded_ceiling_ratio),
                        guarded_ceiling_rel_quality_tolerance=float(args.dual_guarded_ceiling_rel_quality_tolerance),
                        guarded_ceiling_rel_density_tolerance=float(args.dual_guarded_ceiling_rel_density_tolerance),
                        target_state_gate=bool(args.dual_target_state_gate),
                        graph_bias_enabled=bool(args.dual_graph_bias_enabled),
                        graph_bias_strength=float(args.dual_graph_bias_strength),
                        graph_bias_clip=float(args.dual_graph_bias_clip),
                        graph_over_preserved_weight=float(args.dual_graph_over_preserved_weight),
                        graph_adapted_density_weight=float(args.dual_graph_adapted_density_weight),
                        graph_adapted_undersize_weight=float(args.dual_graph_adapted_undersize_weight),
                        graph_disagreement_preserve_weight=float(args.dual_graph_disagreement_preserve_weight),
                        graph_entropy_preserve_weight=float(args.dual_graph_entropy_preserve_weight),
                        query_undersize_guard_weight=float(args.dual_query_undersize_guard_weight),
                        query_oversize_guard_weight=float(args.dual_query_oversize_guard_weight),
                        density_gain_tolerance=float(args.dual_density_gain_tolerance),
                        search_consensus_bonus=float(args.dual_search_consensus_bonus),
                        guarded_floor_enabled=bool(args.dual_guarded_floor_enabled),
                        guarded_floor_ratio=float(args.dual_guarded_floor_ratio),
                        guarded_floor_guard_metric=str(args.dual_guarded_floor_guard_metric),
                        guarded_floor_accept_mode=str(args.dual_guarded_floor_accept_mode),
                        guarded_floor_density_tolerance=float(args.dual_guarded_floor_density_tolerance),
                        guarded_floor_quality_tolerance=float(args.dual_guarded_floor_quality_tolerance),
                        guarded_floor_rel_density_tolerance=float(args.dual_guarded_floor_rel_density_tolerance),
                        guarded_floor_rel_quality_tolerance=float(args.dual_guarded_floor_rel_quality_tolerance),
                        guarded_floor_target_trigger=bool(args.dual_guarded_floor_target_trigger),
                        guarded_floor_target_under_rate_threshold=float(args.dual_guarded_floor_target_under_rate_threshold),
                        guarded_floor_probe_ratio=float(args.dual_guarded_floor_probe_ratio),
                        guarded_floor_router_enabled=bool(args.dual_guarded_floor_router_enabled),
                        guarded_floor_router_strong_under_rate_threshold=float(args.dual_guarded_floor_router_strong_under_rate_threshold),
                        guarded_floor_router_moderate_under_rate_threshold=float(args.dual_guarded_floor_router_moderate_under_rate_threshold),
                        guarded_floor_router_quality_drop_threshold=float(args.dual_guarded_floor_router_quality_drop_threshold),
                        guarded_floor_router_density_drop_threshold=float(args.dual_guarded_floor_router_density_drop_threshold),
                        guarded_floor_router_moderate_max_branch_jaccard=float(args.dual_guarded_floor_router_moderate_max_branch_jaccard),
                        guarded_floor_router_strong_ratio=float(args.dual_guarded_floor_router_strong_ratio),
                        guarded_floor_router_moderate_ratio=float(args.dual_guarded_floor_router_moderate_ratio),
                        guarded_floor_router_strong_accept_mode=str(args.dual_guarded_floor_router_strong_accept_mode),
                        guarded_floor_router_moderate_accept_mode=str(args.dual_guarded_floor_router_moderate_accept_mode),
                        state_profile_router_enabled=bool(args.dual_state_profile_router_enabled),
                        state_profile_state_recognition_v2_enabled=bool(args.dual_state_profile_state_recognition_v2_enabled),
                        state_profile_moderate_size_ratio_min=float(args.dual_state_profile_moderate_size_ratio_min),
                        state_profile_moderate_branch_jaccard_max=float(args.dual_state_profile_moderate_branch_jaccard_max),
                        state_profile_moderate_over_rate_min=float(args.dual_state_profile_moderate_over_rate_min),
                        state_profile_moderate_density_delta_min=float(args.dual_state_profile_moderate_density_delta_min),
                        state_profile_moderate_entropy_min=float(args.dual_state_profile_moderate_entropy_min),
                        state_profile_stable_size_ratio_low=float(args.dual_state_profile_stable_size_ratio_low),
                        state_profile_stable_size_ratio_high=float(args.dual_state_profile_stable_size_ratio_high),
                        state_profile_stable_branch_jaccard_min=float(args.dual_state_profile_stable_branch_jaccard_min),
                        state_profile_stable_tight_size_ratio_low=float(args.dual_state_profile_stable_tight_size_ratio_low),
                        state_profile_stable_tight_size_ratio_high=float(args.dual_state_profile_stable_tight_size_ratio_high),
                        state_profile_stable_tight_branch_jaccard_min=float(args.dual_state_profile_stable_tight_branch_jaccard_min),
                        state_profile_stable_entropy_min=float(args.dual_state_profile_stable_entropy_min),
                        state_profile_stable_density_delta_abs_max=float(args.dual_state_profile_stable_density_delta_abs_max),
                        state_profile_strong_size_ratio_min=float(args.dual_state_profile_strong_size_ratio_min),
                        state_profile_strong_branch_jaccard_max=float(args.dual_state_profile_strong_branch_jaccard_max),
                        state_profile_shift_density_delta_min=float(args.dual_state_profile_shift_density_delta_min),
                        state_profile_shift_branch_jaccard_min=float(args.dual_state_profile_shift_branch_jaccard_min),
                        state_profile_alpha_enabled=bool(args.dual_state_profile_alpha_enabled),
                        state_profile_search_enabled=bool(args.dual_state_profile_search_enabled),
                        state_profile_core_proto_enabled=bool(args.dual_state_profile_core_proto_enabled),
                        state_profile_structure_enabled=bool(args.dual_state_profile_structure_enabled),
                        state_profile_strong_alpha_bias=float(args.dual_state_profile_strong_alpha_bias),
                        state_profile_moderate_alpha_bias=float(args.dual_state_profile_moderate_alpha_bias),
                        state_profile_boundary_alpha_bias=float(args.dual_state_profile_boundary_alpha_bias),
                        state_profile_shift_alpha_bias=float(args.dual_state_profile_shift_alpha_bias),
                        state_profile_stable_alpha_bias=float(args.dual_state_profile_stable_alpha_bias),
                        state_profile_core_weight_strong=float(args.dual_state_profile_core_weight_strong),
                        state_profile_core_weight_moderate=float(args.dual_state_profile_core_weight_moderate),
                        state_profile_core_weight_boundary=float(args.dual_state_profile_core_weight_boundary),
                        state_profile_core_weight_shift=float(args.dual_state_profile_core_weight_shift),
                        state_profile_core_weight_stable=float(args.dual_state_profile_core_weight_stable),
                        state_profile_core_max_nodes=int(args.dual_state_profile_core_max_nodes),
                        state_profile_core_source_mode=str(args.dual_state_profile_core_source_mode),
                        state_profile_core_agreement_topk=int(args.dual_state_profile_core_agreement_topk),
                        state_profile_core_source_by_state_enabled=bool(args.dual_state_profile_core_source_by_state_enabled),
                        state_profile_core_source_mode_strong=str(args.dual_state_profile_core_source_mode_strong),
                        state_profile_core_source_mode_moderate=str(args.dual_state_profile_core_source_mode_moderate),
                        state_profile_core_source_mode_boundary=str(args.dual_state_profile_core_source_mode_boundary),
                        state_profile_core_source_mode_shift=str(args.dual_state_profile_core_source_mode_shift),
                        state_profile_core_source_mode_stable=str(args.dual_state_profile_core_source_mode_stable),
                        state_profile_core_strict_agreement=bool(args.dual_state_profile_core_strict_agreement),
                        state_profile_core_min_agreement_nodes=int(args.dual_state_profile_core_min_agreement_nodes),
                        state_profile_score_source_enabled=bool(args.dual_state_profile_score_source_enabled),
                        state_profile_action_isolation_enabled=bool(args.dual_state_profile_action_isolation_enabled),
                        state_profile_action_isolation_scope=str(args.dual_state_profile_action_isolation_scope),
                        state_profile_score_mode_strong=str(args.dual_state_profile_score_mode_strong),
                        state_profile_score_mode_moderate=str(args.dual_state_profile_score_mode_moderate),
                        state_profile_score_mode_boundary=str(args.dual_state_profile_score_mode_boundary),
                        state_profile_score_mode_shift=str(args.dual_state_profile_score_mode_shift),
                        state_profile_score_mode_stable=str(args.dual_state_profile_score_mode_stable),
                        state_profile_score_residual_scale_strong=float(args.dual_state_profile_score_residual_scale_strong),
                        state_profile_score_residual_scale_moderate=float(args.dual_state_profile_score_residual_scale_moderate),
                        state_profile_score_residual_scale_boundary=float(args.dual_state_profile_score_residual_scale_boundary),
                        state_profile_score_residual_scale_shift=float(args.dual_state_profile_score_residual_scale_shift),
                        state_profile_score_residual_scale_stable=float(args.dual_state_profile_score_residual_scale_stable),
                        state_profile_score_recall_weight_moderate=float(args.dual_state_profile_score_recall_weight_moderate),
                        state_profile_score_consensus_weight_boundary=float(args.dual_state_profile_score_consensus_weight_boundary),
                        state_profile_score_adapt_bonus_shift=float(args.dual_state_profile_score_adapt_bonus_shift),
                        hierarchical_boundary_state_enabled=bool(args.dual_hierarchical_boundary_state_enabled),
                        boundary_plateau_relative_quality_threshold=float(args.dual_boundary_plateau_relative_quality_threshold),
                        boundary_plateau_precision_extra_penalty=float(args.dual_boundary_plateau_precision_extra_penalty),
                        boundary_plateau_ceiling_enabled=bool(args.dual_boundary_plateau_ceiling_enabled),
                        boundary_plateau_ceiling_ratio=float(args.dual_boundary_plateau_ceiling_ratio),
                        boundary_plateau_ceiling_rel_quality_tolerance=float(args.dual_boundary_plateau_ceiling_rel_quality_tolerance),
                        boundary_plateau_ceiling_rel_density_tolerance=float(args.dual_boundary_plateau_ceiling_rel_density_tolerance),
                        state_profile_core_reliability_enabled=bool(args.dual_state_profile_core_reliability_enabled),
                        state_profile_core_reliability_min=float(args.dual_state_profile_core_reliability_min),
                        state_profile_core_reliability_min_nodes=int(args.dual_state_profile_core_reliability_min_nodes),
                        state_profile_core_reliability_agreement_weight=float(args.dual_state_profile_core_reliability_agreement_weight),
                        state_profile_core_reliability_count_weight=float(args.dual_state_profile_core_reliability_count_weight),
                        state_profile_core_reliability_query_sim_weight=float(args.dual_state_profile_core_reliability_query_sim_weight),
                        state_profile_structure_weight_moderate=float(args.dual_state_profile_structure_weight_moderate),
                        state_profile_structure_weight_boundary=float(args.dual_state_profile_structure_weight_boundary),
                        state_profile_structure_weight_strong=float(args.dual_state_profile_structure_weight_strong),
                        state_profile_structure_weight_shift=float(args.dual_state_profile_structure_weight_shift),
                        state_profile_structure_weight_stable=float(args.dual_state_profile_structure_weight_stable),
                        state_profile_structure_core_topk=int(args.dual_state_profile_structure_core_topk),
                        state_profile_moderate_recall_bonus=float(args.dual_state_profile_moderate_recall_bonus),
                        state_profile_boundary_quality_penalty=float(args.dual_state_profile_boundary_quality_penalty),
                        state_profile_shift_adapt_bonus=float(args.dual_state_profile_shift_adapt_bonus),
                        state_profile_alpha_bounds_enabled=bool(args.dual_state_profile_alpha_bounds_enabled),
                        state_profile_strong_alpha_min=float(args.dual_state_profile_strong_alpha_min),
                        state_profile_moderate_alpha_min=float(args.dual_state_profile_moderate_alpha_min),
                        state_profile_moderate_alpha_max=float(args.dual_state_profile_moderate_alpha_max),
                        state_profile_boundary_alpha_max=float(args.dual_state_profile_boundary_alpha_max),
                        state_profile_shift_alpha_min=float(args.dual_state_profile_shift_alpha_min),
                        state_profile_floor_enabled=bool(args.dual_state_profile_floor_enabled),
                        state_profile_strong_floor_ratio=float(args.dual_state_profile_strong_floor_ratio),
                        state_profile_moderate_floor_ratio=float(args.dual_state_profile_moderate_floor_ratio),
                        state_profile_boundary_floor_ratio=float(args.dual_state_profile_boundary_floor_ratio),
                        state_profile_shift_floor_ratio=float(args.dual_state_profile_shift_floor_ratio),
                        state_profile_stable_floor_ratio=float(args.dual_state_profile_stable_floor_ratio),
                        state_profile_strong_accept_mode=str(args.dual_state_profile_strong_accept_mode),
                        state_profile_moderate_accept_mode=str(args.dual_state_profile_moderate_accept_mode),
                        state_profile_boundary_accept_mode=str(args.dual_state_profile_boundary_accept_mode),
                        state_profile_shift_accept_mode=str(args.dual_state_profile_shift_accept_mode),
                        state_profile_stable_accept_mode=str(args.dual_state_profile_stable_accept_mode),
                    ),
                    write_dual_diagnostics=bool(args.write_dual_branch_diagnostics),
                    edge_index=graph.edge_index.detach().cpu() if bool(args.dual_state_profile_structure_enabled) and hasattr(graph, "edge_index") and graph.edge_index is not None else None,
                )
                _paper_stage_event(runtime_jsonl, "paper_stage_state_scoring_search_end", args, graph_name=args.graph_name)
                print(f"wrote {outputs.jsonl_path}")
                print(f"wrote {outputs.pred_path}")
                print(f"wrote {outputs.protocol_check_path}")
                print(f"wrote {outputs.status_path}")
                profile_event(runtime_jsonl, "fixed_inference_end", graph_name=args.graph_name, jsonl_path=str(outputs.jsonl_path), record_count=len(outputs.records), dual_branch_score=True)
                return
            elif bool(args.multi_view_search):
                views = _parse_candidate_views(args.candidate_views, fallback_mode=args.score_embedding, fallback_calibration=args.score_calibration)
                manifests: list[tuple[str, Path, str]] = []
                view_profiles = {view_name: (mode, cal) for view_name, mode, cal in views}
                unique_modes = {}
                for view_name, mode, cal in views:
                    unique_modes.setdefault(mode, []).append((view_name, cal))
                for mode, rows in unique_modes.items():
                    manifest = save_embedding_chunk_stream(
                        chunks=iter_embedding_chunks(
                            model=model,
                            global_token_seqs=graph.global_token_seqs.cpu(),
                            device=device,
                            chunk_size=int(args.chunk_size),
                            runtime_jsonl=runtime_jsonl,
                            graph_name=args.graph_name,
                            t_eval=args.t_eval,
                            structure_condition=structure_condition,
                            score_embedding=mode,
                            ratio_guidance_pair=ratio_guidance_pair,
                            ratio_guidance_weight=ratio_guidance_weight,
                            ratio_guidance_clip=ratio_guidance_clip,
                            adaptive_ratio_guidance=bool(args.adaptive_ratio_guidance or mode == "adaptive_guided_refined"),
                            ratio_guidance_confidence_scale=float(args.ratio_guidance_confidence_scale),
                            ratio_guidance_min_mult=float(args.ratio_guidance_min_mult),
                            ratio_guidance_max_mult=float(args.ratio_guidance_max_mult),
                            adaptive_fusion=bool(args.adaptive_fusion or mode == "adaptive_raw_plus_guided_refined"),
                            fusion_min_guided=float(args.fusion_min_guided),
                            fusion_max_guided=float(args.fusion_max_guided),
                            fusion_confidence_scale=float(args.fusion_confidence_scale),
                        ),
                        output_dir=Path(args.output_dir) / "fixed_embedding_shards" / mode,
                        graph_name=f"{args.graph_name}_{mode}",
                        embedding_dim=int(model_cfg.hidden_dim) * 2,
                        num_nodes=int(graph.global_token_seqs.shape[0]),
                        dtype="torch.float32",
                        metadata={
                            "fixed_inference": True,
                            "labels_used": False,
                            "ground_truth_used": False,
                            "metrics_computed": False,
                            "checkpoint": str(args.checkpoint),
                            "score_embedding": str(mode),
                            "multi_view_search": True,
                        },
                        runtime_jsonl=runtime_jsonl,
                    )
                    for view_name, cal in rows:
                        manifests.append((view_name, manifest, cal))
                outputs = run_fixed_inference_from_multiview_embedding_shards(
                    graph_name=args.graph_name,
                    embedding_manifests=manifests,
                    query_batch=qbatch,
                    output_dir=args.output_dir,
                    checkpoint_path=args.checkpoint,
                    global_node_ids=global_node_ids,
                    chunk_size=args.chunk_size,
                    runtime_jsonl=runtime_jsonl,
                    selection_config=MultiViewSelectionConfig(
                        selection=str(args.candidate_selection),
                        compactness_weight=float(args.candidate_compactness_weight),
                        boundary_gap_weight=float(args.candidate_boundary_gap_weight),
                        size_penalty_weight=float(args.candidate_size_penalty_weight),
                        structural_quality_weight=float(args.candidate_structural_quality_weight),
                        lcc_weight=float(args.candidate_lcc_weight),
                        conductance_weight=float(args.candidate_conductance_weight),
                        raw_agreement_weight=float(args.candidate_raw_agreement_weight),
                        diversity_bonus_weight=float(args.candidate_diversity_bonus_weight),
                        size_reference_power=float(args.candidate_size_reference_power),
                        canonical_view=str(args.candidate_canonical_view),
                        duplicate_jaccard_threshold=float(args.candidate_duplicate_jaccard_threshold),
                        topk_overlap_k=int(args.candidate_topk_overlap_k),
                        temperature_value=float(args.candidate_temperature_value),
                        local_center_top_fraction=float(args.candidate_local_center_top_fraction),
                        precision_size_strength=float(args.candidate_precision_size_strength),
                        recall_size_bonus=float(args.candidate_recall_size_bonus),
                        canonical_dominance_guard=bool(args.candidate_canonical_dominance_guard),
                        canonical_switch_margin=float(args.candidate_canonical_switch_margin),
                        alternative_min_size_ratio=float(args.candidate_alternative_min_size_ratio),
                        temperature_min_size_ratio=float(args.candidate_temperature_min_size_ratio),
                        temperature_entropy_threshold=float(args.candidate_temperature_entropy_threshold),
                        balanced_size_low=float(args.candidate_balanced_size_low),
                        balanced_size_high=float(args.candidate_balanced_size_high),
                        under_size_penalty_weight=float(args.candidate_under_size_penalty_weight),
                        temperature_penalty_weight=float(args.candidate_temperature_penalty_weight),
                        recall_view_bonus=float(args.candidate_recall_view_bonus),
                    ),
                    view_profiles=view_profiles,
                    edge_index=graph.edge_index.detach().cpu() if hasattr(graph, "edge_index") and graph.edge_index is not None else None,
                    write_candidate_diagnostics=bool(args.write_candidate_diagnostics),
                )
                print(f"wrote {outputs.jsonl_path}")
                print(f"wrote {outputs.pred_path}")
                print(f"wrote {outputs.protocol_check_path}")
                print(f"wrote {outputs.status_path}")
                profile_event(runtime_jsonl, "fixed_inference_end", graph_name=args.graph_name, jsonl_path=str(outputs.jsonl_path), record_count=len(outputs.records), multi_view_search=True)
                return
            _paper_stage_event(runtime_jsonl, "paper_stage_adapted_embedding_start", args, graph_name=args.graph_name)
            embedding_manifest = save_embedding_chunk_stream(
                chunks=iter_embedding_chunks(
                    model=model,
                    global_token_seqs=graph.global_token_seqs.cpu(),
                    device=device,
                    chunk_size=int(args.chunk_size),
                    runtime_jsonl=runtime_jsonl,
                    graph_name=args.graph_name,
                    t_eval=args.t_eval,
                    structure_condition=structure_condition,
                    score_embedding=args.score_embedding,
                    ratio_guidance_pair=ratio_guidance_pair,
                    ratio_guidance_weight=ratio_guidance_weight,
                    ratio_guidance_clip=ratio_guidance_clip,
                    adaptive_ratio_guidance=bool(args.adaptive_ratio_guidance or args.score_embedding == "adaptive_guided_refined"),
                    ratio_guidance_confidence_scale=float(args.ratio_guidance_confidence_scale),
                    ratio_guidance_min_mult=float(args.ratio_guidance_min_mult),
                    ratio_guidance_max_mult=float(args.ratio_guidance_max_mult),
                    adaptive_fusion=bool(args.adaptive_fusion or args.score_embedding == "adaptive_raw_plus_guided_refined"),
                    fusion_min_guided=float(args.fusion_min_guided),
                    fusion_max_guided=float(args.fusion_max_guided),
                    fusion_confidence_scale=float(args.fusion_confidence_scale),
                    guidance_diagnostics=ratio_diagnostics,
                    guidance_runtime_timer=ratio_runtime_timer,
                ),
                output_dir=Path(args.output_dir) / "fixed_embedding_shards",
                graph_name=args.graph_name,
                embedding_dim=int(model_cfg.hidden_dim) * 2,
                num_nodes=int(graph.global_token_seqs.shape[0]),
                dtype="torch.float32",
                metadata={
                    "fixed_inference": True,
                    "labels_used": False,
                    "ground_truth_used": False,
                    "metrics_computed": False,
                    "checkpoint": str(args.checkpoint),
                    "score_embedding": str(args.score_embedding),
                },
                runtime_jsonl=runtime_jsonl,
            )
            _paper_stage_event(runtime_jsonl, "paper_stage_adapted_embedding_end", args, graph_name=args.graph_name)
            _paper_stage_event(runtime_jsonl, "paper_stage_state_scoring_search_start", args, graph_name=args.graph_name)
            outputs = run_fixed_inference_from_embedding_shards(
                graph_name=args.graph_name,
                embedding_manifest=embedding_manifest,
                query_batch=qbatch,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                global_node_ids=global_node_ids,
                chunk_size=args.chunk_size,
                runtime_jsonl=runtime_jsonl,
                score_calibration=str(args.score_calibration),
            )
            _paper_stage_event(runtime_jsonl, "paper_stage_state_scoring_search_end", args, graph_name=args.graph_name)
            _write_ratio_diagnostics(args, ratio_diagnostics)
            print(f"wrote {outputs.jsonl_path}")
            print(f"wrote {outputs.pred_path}")
            print(f"wrote {outputs.protocol_check_path}")
            print(f"wrote {outputs.status_path}")
            profile_event(runtime_jsonl, "fixed_inference_end", graph_name=args.graph_name, jsonl_path=str(outputs.jsonl_path), record_count=len(outputs.records))
            return
        emb = encode_embedding_in_chunks(
            model=model,
            global_token_seqs=graph.global_token_seqs.cpu(),
            device=device,
            chunk_size=int(args.chunk_size),
            runtime_jsonl=runtime_jsonl,
            graph_name=args.graph_name,
            t_eval=args.t_eval,
            structure_condition=structure_condition,
            score_embedding=args.score_embedding,
            ratio_guidance_pair=ratio_guidance_pair,
            ratio_guidance_weight=ratio_guidance_weight,
            ratio_guidance_clip=ratio_guidance_clip,
            adaptive_ratio_guidance=bool(args.adaptive_ratio_guidance or args.score_embedding == "adaptive_guided_refined"),
            ratio_guidance_confidence_scale=float(args.ratio_guidance_confidence_scale),
            ratio_guidance_min_mult=float(args.ratio_guidance_min_mult),
            ratio_guidance_max_mult=float(args.ratio_guidance_max_mult),
            adaptive_fusion=bool(args.adaptive_fusion or args.score_embedding == "adaptive_raw_plus_guided_refined"),
            fusion_min_guided=float(args.fusion_min_guided),
            fusion_max_guided=float(args.fusion_max_guided),
            fusion_confidence_scale=float(args.fusion_confidence_scale),
            guidance_diagnostics=ratio_diagnostics,
            guidance_runtime_timer=ratio_runtime_timer,
        )
        qbatch = graph.query_batch
        global_node_ids = graph.global_node_ids.tolist()
    else:
        raise RuntimeError("--embedding-pt or --dataset-root is required")

    outputs = run_fixed_inference_from_embeddings(
        graph_name=args.graph_name,
        node_embeddings=emb,
        query_batch=qbatch,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        global_node_ids=global_node_ids,
        chunk_size=args.chunk_size,
        score_calibration=str(args.score_calibration),
    )
    if args.dataset_root:
        _write_ratio_diagnostics(args, ratio_diagnostics)
    print(f"wrote {outputs.jsonl_path}")
    print(f"wrote {outputs.pred_path}")
    print(f"wrote {outputs.protocol_check_path}")
    print(f"wrote {outputs.status_path}")
    profile_event(runtime_jsonl, "fixed_inference_end", graph_name=args.graph_name, jsonl_path=str(outputs.jsonl_path), record_count=len(outputs.records))


def _maybe_check_resource_preflight(args: argparse.Namespace, *, runtime_jsonl: str | None) -> dict[str, object]:
    if not args.preflight_report_json:
        if args.require_preflight_pass:
            raise RuntimeError("--require-preflight-pass requires --preflight-report-json")
        return {"enabled": False, "require_pass": bool(args.require_preflight_pass)}
    payload = load_resource_preflight_payload(args.preflight_report_json)
    report = build_resource_preflight_gate_report(
        payload,
        require_pass=bool(args.require_preflight_pass),
        token_cache_strategy_enabled=bool(args.token_cache_dir),
        embedding_cache_strategy_enabled=str(args.embedding_export_format) == "shards",
        score_chunk_strategy_enabled=True,
    )
    profile_event(runtime_jsonl, "resource_preflight_gate_checked", **report.to_dict())
    assert_resource_preflight_gate_passed(report)
    return report.to_dict()


if __name__ == "__main__":
    main()
