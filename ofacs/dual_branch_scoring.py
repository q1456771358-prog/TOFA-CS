"""V5.5 dual-branch preservation/adaptation scoring.

This module is target-label-free.  It combines a preserved branch (raw/cache or
initial-checkpoint ranking) and an adapted branch (trained / target-guided
DiffGDA ranking) at the *score* level before senior GlobalSearch.  It never
loads target ground truth or metrics; all diagnostics are unsupervised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .chunked_scoring import ChunkedScoreResult, _normalize_query_sets, default_top_k_pool
from .embedding_store import load_embedding_manifest, _load_shard_tensor
from .errors import V5ProtocolError
from .runtime_profile import profile_event
from .senior_compat import SeniorQueryBatch, senior_global_search_from_ranked_scores, subgraph_density_controlled


@dataclass(frozen=True)
class DualBranchGateConfig:
    enabled: bool = False
    alpha_mode: str = "adaptive"  # fixed | adaptive
    fixed_alpha: float = 0.50
    min_alpha: float = 0.10
    max_alpha: float = 0.90
    base_alpha: float = 0.50
    over_preserved_weight: float = 0.18
    adapted_density_weight: float = 0.12
    entropy_preserve_weight: float = 0.16
    disagreement_preserve_weight: float = 0.18
    adapted_undersize_weight: float = 0.22
    adapted_oversize_weight: float = 0.08
    boundary_gap_adapt_weight: float = 0.08
    residual_gain: float = 1.0
    size_reference_power: float = 0.50
    min_size_reference: float = 20.0
    search_profile: str = "senior"  # senior | balanced_size | entropy_aware | precision_recall
    balanced_size_low: float = 0.85
    balanced_size_high: float = 1.25
    entropy_threshold: float = 0.82
    high_entropy_recall_bonus: float = 0.04
    precision_over_penalty: float = 0.08
    recall_under_penalty: float = 0.08
    # B1 endpoint repair.  Keep a separate precision-recall coefficient so
    # the historical target_state_recovery coefficient above stays unchanged.
    # All new actions are disabled by default, preserving the old B1 path.
    precision_recall_over_penalty: float = 0.05
    overexpansion_router_enabled: bool = False
    overexpansion_ratio_threshold: float = 1.10
    overexpansion_rate_threshold: float = 0.35
    overexpansion_precision_extra_penalty: float = 0.0
    guarded_ceiling_enabled: bool = False
    guarded_ceiling_ratio: float = 0.92
    guarded_ceiling_rel_quality_tolerance: float = 0.01
    guarded_ceiling_rel_density_tolerance: float = 0.01
    # V5.6-lite target-state gate. These are label-free graph/query statistics
    # that bias alpha before final score fusion.  They are deliberately mild;
    # standard experiments should use presets rather than sweeping these values.
    target_state_gate: bool = False
    graph_bias_enabled: bool = False
    graph_bias_strength: float = 0.18
    graph_bias_clip: float = 0.18
    graph_over_preserved_weight: float = 0.10
    graph_adapted_density_weight: float = 0.08
    graph_adapted_undersize_weight: float = 0.16
    graph_disagreement_preserve_weight: float = 0.06
    graph_entropy_preserve_weight: float = 0.04
    query_undersize_guard_weight: float = 0.18
    query_oversize_guard_weight: float = 0.10
    density_gain_tolerance: float = 0.015
    search_consensus_bonus: float = 0.025
    # V5.6.1 guarded size-floor endpoint recovery.  This is inference-only
    # and label-free: it checks whether the final prefix is much smaller than
    # the larger branch candidate and, only when the larger prefix quality does
    # not drop beyond tolerance, recovers to that floor prefix.
    guarded_floor_enabled: bool = False
    guarded_floor_ratio: float = 0.85
    guarded_floor_guard_metric: str = "quality"  # quality | density | either | both (legacy absolute guard)
    guarded_floor_accept_mode: str = "strict_abs"  # strict_abs | force | relative_quality | relative_quality_density
    guarded_floor_density_tolerance: float = 0.020
    guarded_floor_quality_tolerance: float = 0.020
    guarded_floor_rel_density_tolerance: float = 0.050
    guarded_floor_rel_quality_tolerance: float = 0.050
    guarded_floor_target_trigger: bool = False
    guarded_floor_target_under_rate_threshold: float = 0.35
    guarded_floor_probe_ratio: float = 0.85
    # V5.6.3 multi-state unsupervised search router.  It still uses only
    # label-free target/query statistics.  The router chooses among strong
    # recovery, mild recovery, boundary-protection, and default/no-recovery
    # states without using target graph names.
    guarded_floor_router_enabled: bool = False
    guarded_floor_router_strong_under_rate_threshold: float = 0.35
    guarded_floor_router_moderate_under_rate_threshold: float = 0.12
    guarded_floor_router_quality_drop_threshold: float = 0.10
    # V5.6.5: medium recovery must also pass a boundary-preservation guard.
    # Quality drop is the primary guard; density drop is optional and disabled
    # by default with a large threshold so older presets remain unchanged.
    guarded_floor_router_density_drop_threshold: float = 1.00
    # Medium recovery should only apply when the two branches disagree enough to
    # make a larger branch boundary informative.  High branch overlap is treated
    # as boundary-sensitive/stable rather than blindly expanded.
    guarded_floor_router_moderate_max_branch_jaccard: float = 1.00
    guarded_floor_router_strong_ratio: float = 1.00
    guarded_floor_router_moderate_ratio: float = 0.90
    guarded_floor_router_strong_accept_mode: str = "force"
    guarded_floor_router_moderate_accept_mode: str = "relative_quality"
    # V5.6.7 state-conditioned parameter profile router.  This is still
    # inference-only and label-free.  It translates historical hyperparameter
    # findings into state profiles instead of target-name-specific rules.
    state_profile_router_enabled: bool = False
    # V5.7.2: revised label-free state recognizer.  Earlier score-profile
    # experiments showed that Citeseer-like moderate-recall targets were often
    # routed to boundary-preserve, so their recall package never actually ran.
    # These thresholds make state recognition itself explicit and diagnostic
    # while preserving the legacy recognizer for older presets.
    state_profile_state_recognition_v2_enabled: bool = False
    state_profile_moderate_size_ratio_min: float = 1.08
    state_profile_moderate_branch_jaccard_max: float = 0.62
    state_profile_moderate_over_rate_min: float = 0.25
    state_profile_moderate_density_delta_min: float = -1.20
    state_profile_moderate_entropy_min: float = 0.90
    state_profile_stable_size_ratio_low: float = 0.90
    state_profile_stable_size_ratio_high: float = 1.20
    state_profile_stable_branch_jaccard_min: float = 0.70
    state_profile_stable_tight_size_ratio_low: float = 0.95
    state_profile_stable_tight_size_ratio_high: float = 1.08
    state_profile_stable_tight_branch_jaccard_min: float = 0.65
    state_profile_stable_entropy_min: float = 0.99
    # V5.7.2 density guard: stable-default should mean size/overlap are stable
    # AND the adapted-vs-preserved density difference is small.  Without this
    # guard, high-shift targets can be swallowed by stable-default before the
    # shift-adaptive package has a chance to run.
    state_profile_stable_density_delta_abs_max: float = 0.04
    state_profile_strong_size_ratio_min: float = 1.35
    state_profile_strong_branch_jaccard_max: float = 0.72
    state_profile_shift_density_delta_min: float = 0.04
    state_profile_shift_branch_jaccard_min: float = 0.70
    state_profile_alpha_enabled: bool = False
    state_profile_search_enabled: bool = False
    state_profile_core_proto_enabled: bool = False
    state_profile_structure_enabled: bool = False
    state_profile_strong_alpha_bias: float = 0.08
    state_profile_moderate_alpha_bias: float = 0.02
    state_profile_boundary_alpha_bias: float = -0.10
    state_profile_shift_alpha_bias: float = 0.08
    state_profile_stable_alpha_bias: float = 0.0
    state_profile_core_weight_strong: float = 0.00
    state_profile_core_weight_moderate: float = 0.12
    state_profile_core_weight_boundary: float = 0.18
    state_profile_core_weight_shift: float = 0.05
    state_profile_core_weight_stable: float = 0.00
    state_profile_core_max_nodes: int = 64
    # V5.6.9 core-profile refinement.  The V5.6.8 experiments showed that
    # high-confidence community cores are the useful part of the state profile
    # package.  The source mode controls whether the core comes from the state
    # preferred branch or from nodes jointly supported by both branches.
    state_profile_core_source_mode: str = "profile_top"  # profile_top | agreement_top | agreement_then_profile
    state_profile_core_agreement_topk: int = 160
    # V5.7.0 state-aware core source routing.  V5.6.9 made core source a
    # preset-level knob; these fields make it a real state-conditioned package:
    # strong/shift states can use the adapted/profile core, boundary/stable
    # states can use branch-agreement cores, and moderate-recall can mix both.
    state_profile_core_source_by_state_enabled: bool = False
    state_profile_core_source_mode_strong: str = "profile_top"
    state_profile_core_source_mode_moderate: str = "agreement_then_profile"
    state_profile_core_source_mode_boundary: str = "agreement_then_profile"
    state_profile_core_source_mode_shift: str = "profile_top"
    state_profile_core_source_mode_stable: str = "agreement_top"
    # If true, agreement cores are restricted to nodes inside both branch search
    # candidates.  If false, the router may also use nodes that are ranked high
    # by both branches but were not selected by both branch searches.
    state_profile_core_strict_agreement: bool = False
    state_profile_core_min_agreement_nodes: int = 4
    # V5.7.1: state-conditioned score source routing and reliability-aware
    # core prototypes.  These are inference-time equivalents of historical
    # hyperparameter findings: weak adaptation for moderate recall states,
    # preservation for boundary-sensitive states, stronger adaptation for
    # shift states, and reliability-gated core prototypes.
    state_profile_score_source_enabled: bool = False
    # V5.7.3 action isolation: score-source / recall actions should only
    # affect the state they are designed for.  In earlier V571/V572 presets,
    # enabling a recall-oriented score formula could also change boundary or
    # stable targets in the same preset.  This flag keeps non-target states on
    # the safe V570-style default fusion while allowing moderate-recall targets
    # to use their recall package.
    state_profile_action_isolation_enabled: bool = False
    # ``moderate_only`` preserves historical V573/V574 behavior exactly.
    # ``profile_specific`` lets each recognized state execute only its own
    # explicitly configured score action.  The new B1 refinement uses this to
    # activate the existing boundary preserve-consensus action without adding
    # a second router or changing strong/shift/stable safe defaults.
    state_profile_action_isolation_scope: str = "moderate_only"
    state_profile_score_mode_strong: str = "adapt_preferred"
    state_profile_score_mode_moderate: str = "weak_adapt_residual"
    state_profile_score_mode_boundary: str = "preserve_consensus"
    state_profile_score_mode_shift: str = "adapt_preferred"
    state_profile_score_mode_stable: str = "default_fusion"
    state_profile_score_residual_scale_strong: float = 1.05
    state_profile_score_residual_scale_moderate: float = 0.70
    state_profile_score_residual_scale_boundary: float = 0.55
    state_profile_score_residual_scale_shift: float = 1.15
    state_profile_score_residual_scale_stable: float = 1.00
    state_profile_score_recall_weight_moderate: float = 0.040
    state_profile_score_consensus_weight_boundary: float = 0.050
    state_profile_score_adapt_bonus_shift: float = 0.030
    # V5.7.5 hierarchical unlabeled retrieval-state refinement.  The first
    # stage keeps the established five-state population recognizer.  Only its
    # boundary fallback is refined after a safe/default endpoint probe.  A
    # numerically saturated endpoint receives the precision-oriented boundary
    # package; a margin-sensitive endpoint stays on the exact B1 safe path.
    # The switch is target-name-free and uses neither labels nor ground truth.
    hierarchical_boundary_state_enabled: bool = False
    boundary_plateau_relative_quality_threshold: float = 1.0e-6
    boundary_plateau_precision_extra_penalty: float = 0.0
    boundary_plateau_ceiling_enabled: bool = False
    boundary_plateau_ceiling_ratio: float = 0.95
    boundary_plateau_ceiling_rel_quality_tolerance: float = 0.01
    boundary_plateau_ceiling_rel_density_tolerance: float = 0.01
    state_profile_core_reliability_enabled: bool = False
    state_profile_core_reliability_min: float = 0.25
    state_profile_core_reliability_min_nodes: int = 8
    state_profile_core_reliability_agreement_weight: float = 0.50
    state_profile_core_reliability_count_weight: float = 0.30
    state_profile_core_reliability_query_sim_weight: float = 0.20
    state_profile_structure_weight_moderate: float = 0.020
    state_profile_structure_weight_boundary: float = 0.035
    state_profile_structure_weight_strong: float = 0.000
    state_profile_structure_weight_shift: float = 0.010
    state_profile_structure_weight_stable: float = 0.000
    state_profile_structure_core_topk: int = 24
    state_profile_moderate_recall_bonus: float = 0.030
    state_profile_boundary_quality_penalty: float = 0.040
    state_profile_shift_adapt_bonus: float = 0.020
    # V5.6.8 complete state-conditioned parameter packages.  These settings
    # make the router choose not only a small alpha/search bias, but also the
    # state-specific branch tendency and floor-recovery package.  This keeps the
    # method single-model and label-free while incorporating historical
    # hyperparameter findings as state profiles rather than target-name rules.
    state_profile_alpha_bounds_enabled: bool = False
    state_profile_strong_alpha_min: float = 0.55
    state_profile_moderate_alpha_min: float = 0.45
    state_profile_moderate_alpha_max: float = 0.72
    state_profile_boundary_alpha_max: float = 0.45
    state_profile_shift_alpha_min: float = 0.60
    state_profile_floor_enabled: bool = False
    state_profile_strong_floor_ratio: float = 1.00
    state_profile_moderate_floor_ratio: float = 0.90
    state_profile_boundary_floor_ratio: float = 0.00
    state_profile_shift_floor_ratio: float = 0.00
    state_profile_stable_floor_ratio: float = 0.00
    state_profile_strong_accept_mode: str = "relative_quality"
    state_profile_moderate_accept_mode: str = "relative_quality"
    state_profile_boundary_accept_mode: str = "relative_quality"
    state_profile_shift_accept_mode: str = "relative_quality"
    state_profile_stable_accept_mode: str = "relative_quality"
    diagnostics: bool = True
    eps: float = 1e-8

    def validate(self) -> "DualBranchGateConfig":
        if self.alpha_mode not in {"fixed", "adaptive"}:
            raise V5ProtocolError(f"unsupported dual-branch alpha_mode={self.alpha_mode!r}")
        if self.search_profile not in {"senior", "balanced_size", "entropy_aware", "precision_recall", "target_state_recovery"}:
            raise V5ProtocolError(f"unsupported dual-branch search_profile={self.search_profile!r}")
        if not (0.0 <= float(self.min_alpha) <= float(self.max_alpha) <= 1.0):
            raise V5ProtocolError("dual branch alpha bounds must satisfy 0 <= min <= max <= 1")
        if float(self.fixed_alpha) < 0.0 or float(self.fixed_alpha) > 1.0:
            raise V5ProtocolError("fixed_alpha must be in [0,1]")
        if float(self.balanced_size_low) <= 0 or float(self.balanced_size_high) <= 0:
            raise V5ProtocolError("balanced size bounds must be positive")
        if float(self.balanced_size_low) > float(self.balanced_size_high):
            raise V5ProtocolError("balanced_size_low must be <= balanced_size_high")
        if float(self.precision_recall_over_penalty) < 0.0:
            raise V5ProtocolError("precision_recall_over_penalty must be non-negative")
        if float(self.overexpansion_ratio_threshold) <= 1.0:
            raise V5ProtocolError("overexpansion_ratio_threshold must be > 1")
        if not (0.0 <= float(self.overexpansion_rate_threshold) <= 1.0):
            raise V5ProtocolError("overexpansion_rate_threshold must be in [0,1]")
        if float(self.overexpansion_precision_extra_penalty) < 0.0:
            raise V5ProtocolError("overexpansion_precision_extra_penalty must be non-negative")
        if not (0.0 < float(self.guarded_ceiling_ratio) < 1.0):
            raise V5ProtocolError("guarded_ceiling_ratio must be in (0,1)")
        if float(self.guarded_ceiling_rel_quality_tolerance) < 0.0:
            raise V5ProtocolError("guarded_ceiling_rel_quality_tolerance must be non-negative")
        if float(self.guarded_ceiling_rel_density_tolerance) < 0.0:
            raise V5ProtocolError("guarded_ceiling_rel_density_tolerance must be non-negative")
        if str(self.guarded_floor_guard_metric) not in {"quality", "density", "either", "both"}:
            raise V5ProtocolError("guarded_floor_guard_metric must be one of quality/density/either/both")
        if str(self.guarded_floor_accept_mode) not in {"strict_abs", "force", "relative_quality", "relative_quality_density"}:
            raise V5ProtocolError("guarded_floor_accept_mode must be one of strict_abs/force/relative_quality/relative_quality_density")
        if float(self.guarded_floor_ratio) <= 0.0:
            raise V5ProtocolError("guarded_floor_ratio must be positive")
        if float(self.guarded_floor_probe_ratio) <= 0.0:
            raise V5ProtocolError("guarded_floor_probe_ratio must be positive")
        if float(self.guarded_floor_density_tolerance) < 0.0 or float(self.guarded_floor_quality_tolerance) < 0.0:
            raise V5ProtocolError("guarded floor tolerances must be non-negative")
        if float(self.guarded_floor_rel_density_tolerance) < 0.0 or float(self.guarded_floor_rel_quality_tolerance) < 0.0:
            raise V5ProtocolError("guarded floor relative tolerances must be non-negative")
        if not (0.0 <= float(self.guarded_floor_target_under_rate_threshold) <= 1.0):
            raise V5ProtocolError("guarded_floor_target_under_rate_threshold must be in [0,1]")
        if str(self.guarded_floor_router_strong_accept_mode) not in {"strict_abs", "force", "relative_quality", "relative_quality_density"}:
            raise V5ProtocolError("guarded_floor_router_strong_accept_mode must be a supported floor accept mode")
        if str(self.guarded_floor_router_moderate_accept_mode) not in {"strict_abs", "force", "relative_quality", "relative_quality_density"}:
            raise V5ProtocolError("guarded_floor_router_moderate_accept_mode must be a supported floor accept mode")
        if not (0.0 <= float(self.guarded_floor_router_moderate_under_rate_threshold) <= 1.0):
            raise V5ProtocolError("guarded_floor_router_moderate_under_rate_threshold must be in [0,1]")
        if not (0.0 <= float(self.guarded_floor_router_strong_under_rate_threshold) <= 1.0):
            raise V5ProtocolError("guarded_floor_router_strong_under_rate_threshold must be in [0,1]")
        if float(self.guarded_floor_router_moderate_under_rate_threshold) > float(self.guarded_floor_router_strong_under_rate_threshold):
            raise V5ProtocolError("router moderate threshold must be <= strong threshold")
        if float(self.guarded_floor_router_quality_drop_threshold) < 0.0:
            raise V5ProtocolError("router quality drop threshold must be non-negative")
        if float(self.guarded_floor_router_density_drop_threshold) < 0.0:
            raise V5ProtocolError("router density drop threshold must be non-negative")
        if not (0.0 <= float(self.guarded_floor_router_moderate_max_branch_jaccard) <= 1.0):
            raise V5ProtocolError("router moderate max branch jaccard must be in [0,1]")
        if float(self.guarded_floor_router_strong_ratio) <= 0.0 or float(self.guarded_floor_router_moderate_ratio) <= 0.0:
            raise V5ProtocolError("router floor ratios must be positive")
        for name in (
            "state_profile_moderate_branch_jaccard_max",
            "state_profile_moderate_over_rate_min",
            "state_profile_stable_branch_jaccard_min",
            "state_profile_stable_tight_branch_jaccard_min",
            "state_profile_stable_entropy_min",
            "state_profile_strong_branch_jaccard_max",
            "state_profile_shift_branch_jaccard_min",
        ):
            val = float(getattr(self, name))
            if val < 0.0 or val > 1.0:
                raise V5ProtocolError(f"{name} must be in [0,1]")
        for name in (
            "state_profile_moderate_size_ratio_min",
            "state_profile_stable_size_ratio_low",
            "state_profile_stable_size_ratio_high",
            "state_profile_stable_tight_size_ratio_low",
            "state_profile_stable_tight_size_ratio_high",
            "state_profile_strong_size_ratio_min",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise V5ProtocolError(f"{name} must be positive")
        if float(self.state_profile_stable_size_ratio_low) > float(self.state_profile_stable_size_ratio_high):
            raise V5ProtocolError("state_profile_stable_size_ratio_low must be <= high")
        if float(self.state_profile_stable_tight_size_ratio_low) > float(self.state_profile_stable_tight_size_ratio_high):
            raise V5ProtocolError("state_profile_stable_tight_size_ratio_low must be <= high")
        if float(self.state_profile_stable_density_delta_abs_max) < 0.0:
            raise V5ProtocolError("state_profile_stable_density_delta_abs_max must be non-negative")
        for name in (
            "state_profile_core_weight_strong",
            "state_profile_core_weight_moderate",
            "state_profile_core_weight_boundary",
            "state_profile_core_weight_shift",
            "state_profile_core_weight_stable",
        ):
            val = float(getattr(self, name))
            if val < 0.0 or val > 1.0:
                raise V5ProtocolError(f"{name} must be in [0,1]")
        if int(self.state_profile_core_max_nodes) <= 0:
            raise V5ProtocolError("state_profile_core_max_nodes must be positive")
        allowed_core_modes = {"profile_top", "agreement_top", "agreement_then_profile"}
        for name in (
            "state_profile_core_source_mode",
            "state_profile_core_source_mode_strong",
            "state_profile_core_source_mode_moderate",
            "state_profile_core_source_mode_boundary",
            "state_profile_core_source_mode_shift",
            "state_profile_core_source_mode_stable",
        ):
            if str(getattr(self, name)) not in allowed_core_modes:
                raise V5ProtocolError(f"{name} must be one of profile_top/agreement_top/agreement_then_profile")
        if int(self.state_profile_core_agreement_topk) <= 0:
            raise V5ProtocolError("state_profile_core_agreement_topk must be positive")
        if int(self.state_profile_core_min_agreement_nodes) < 0:
            raise V5ProtocolError("state_profile_core_min_agreement_nodes must be non-negative")
        allowed_score_modes = {"default_fusion", "adapt_preferred", "weak_adapt_residual", "preserve_consensus"}
        if str(self.state_profile_action_isolation_scope) not in {"moderate_only", "profile_specific"}:
            raise V5ProtocolError(
                "state_profile_action_isolation_scope must be moderate_only or profile_specific"
            )
        for name in (
            "state_profile_score_mode_strong",
            "state_profile_score_mode_moderate",
            "state_profile_score_mode_boundary",
            "state_profile_score_mode_shift",
            "state_profile_score_mode_stable",
        ):
            if str(getattr(self, name)) not in allowed_score_modes:
                raise V5ProtocolError(f"{name} must be one of {sorted(allowed_score_modes)}")
        for name in (
            "state_profile_score_residual_scale_strong",
            "state_profile_score_residual_scale_moderate",
            "state_profile_score_residual_scale_boundary",
            "state_profile_score_residual_scale_shift",
            "state_profile_score_residual_scale_stable",
        ):
            if float(getattr(self, name)) < 0.0:
                raise V5ProtocolError(f"{name} must be non-negative")
        for name in (
            "state_profile_score_recall_weight_moderate",
            "state_profile_score_consensus_weight_boundary",
            "state_profile_score_adapt_bonus_shift",
            "boundary_plateau_relative_quality_threshold",
            "boundary_plateau_precision_extra_penalty",
            "boundary_plateau_ceiling_rel_quality_tolerance",
            "boundary_plateau_ceiling_rel_density_tolerance",
        ):
            if float(getattr(self, name)) < 0.0:
                raise V5ProtocolError(f"{name} must be non-negative")
        if not (0.0 < float(self.boundary_plateau_ceiling_ratio) < 1.0):
            raise V5ProtocolError("boundary_plateau_ceiling_ratio must be in (0,1)")
        if bool(self.hierarchical_boundary_state_enabled) and bool(self.state_profile_structure_enabled):
            raise V5ProtocolError(
                "hierarchical boundary refinement requires structure scoring disabled; "
                "the safe first-pass probe intentionally has no target edge-index dependency"
            )
        if bool(self.hierarchical_boundary_state_enabled) and str(self.search_profile) == "senior":
            raise V5ProtocolError(
                "hierarchical boundary refinement requires a prefix-search profile with endpoint diagnostics"
            )
        if not (0.0 <= float(self.state_profile_core_reliability_min) <= 1.0):
            raise V5ProtocolError("state_profile_core_reliability_min must be in [0,1]")
        if int(self.state_profile_core_reliability_min_nodes) <= 0:
            raise V5ProtocolError("state_profile_core_reliability_min_nodes must be positive")
        for name in (
            "state_profile_core_reliability_agreement_weight",
            "state_profile_core_reliability_count_weight",
            "state_profile_core_reliability_query_sim_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise V5ProtocolError(f"{name} must be non-negative")
        if int(self.state_profile_structure_core_topk) <= 0:
            raise V5ProtocolError("state_profile_structure_core_topk must be positive")
        for name in (
            "state_profile_strong_alpha_min",
            "state_profile_moderate_alpha_min",
            "state_profile_moderate_alpha_max",
            "state_profile_boundary_alpha_max",
            "state_profile_shift_alpha_min",
        ):
            val = float(getattr(self, name))
            if val < 0.0 or val > 1.0:
                raise V5ProtocolError(f"{name} must be in [0,1]")
        if float(self.state_profile_moderate_alpha_min) > float(self.state_profile_moderate_alpha_max):
            raise V5ProtocolError("state_profile_moderate_alpha_min must be <= state_profile_moderate_alpha_max")
        for name in (
            "state_profile_strong_floor_ratio",
            "state_profile_moderate_floor_ratio",
            "state_profile_boundary_floor_ratio",
            "state_profile_shift_floor_ratio",
            "state_profile_stable_floor_ratio",
        ):
            val = float(getattr(self, name))
            if val < 0.0:
                raise V5ProtocolError(f"{name} must be non-negative")
        for name in (
            "state_profile_strong_accept_mode",
            "state_profile_moderate_accept_mode",
            "state_profile_boundary_accept_mode",
            "state_profile_shift_accept_mode",
            "state_profile_stable_accept_mode",
        ):
            if str(getattr(self, name)) not in {"strict_abs", "force", "relative_quality", "relative_quality_density"}:
                raise V5ProtocolError(f"{name} must be a supported floor accept mode")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BranchCandidateStats:
    size: int
    density: float
    boundary_gap: float
    entropy: float
    nodes: tuple[int, ...]


@dataclass
class PrefixSearchDecision:
    best_prefix_k: int
    final_prefix_k: int
    best_size: int
    final_size: int
    best_density: float
    final_density: float
    best_quality: float
    final_quality: float
    floor_prefix_k: int
    floor_size: int
    floor_density: float
    floor_quality: float
    density_drop: float
    quality_drop: float
    rel_density_drop: float
    rel_quality_drop: float
    floor_accept_mode: str
    floor_accept_reason: str
    floor_reject_reason: str
    query_floor_enabled: bool
    max_branch_size: int
    floor_ratio: float
    floor_triggered: bool
    floor_accepted: bool
    floor_reason: str
    target_underfloor_rate: float
    target_floor_enabled: bool
    floor_route_state: str
    floor_route_reason: str
    effective_floor_ratio: float
    effective_accept_mode: str
    target_moderate_underfloor_rate: float
    target_strong_underfloor_rate: float
    target_probe_rel_quality_drop: float
    target_probe_rel_density_drop: float
    target_branch_jaccard_mean: float
    target_moderate_max_branch_jaccard: float
    target_moderate_quality_drop_threshold: float
    target_moderate_density_drop_threshold: float
    floor_router_disagreement_pass: bool
    floor_router_quality_drop_pass: bool
    floor_router_density_drop_pass: bool
    state_profile_id: str
    state_profile_reason: str
    state_profile_alpha_bias: float
    state_profile_core_weight: float
    state_profile_structure_weight: float
    state_profile_search_mode: str
    state_profile_query_proto_enabled: bool
    state_profile_structure_enabled: bool
    search_endpoint: int
    precision_over_coefficient: float = 0.0
    precision_size_penalty: float = 0.0
    target_overexpansion_rate: float = 0.0
    target_overexpansion_action: bool = False
    target_overexpansion_reason: str = "disabled"
    ceiling_prefix_k: int = -1
    ceiling_size: int = 0
    ceiling_density: float = float("nan")
    ceiling_quality: float = float("nan")
    ceiling_rel_density_drop: float = float("nan")
    ceiling_rel_quality_drop: float = float("nan")
    ceiling_triggered: bool = False
    ceiling_accepted: bool = False
    ceiling_reason: str = "disabled"
    final_size_before_ceiling: int = 0
    final_size_after_ceiling: int = 0
    previous_prefix_density: float = float("nan")
    previous_prefix_quality: float = float("nan")
    next_prefix_density: float = float("nan")
    next_prefix_quality: float = float("nan")
    marginal_density_from_previous: float = float("nan")
    marginal_quality_from_previous: float = float("nan")
    marginal_density_to_next: float = float("nan")
    marginal_quality_to_next: float = float("nan")


def _entropy(values: Sequence[float], eps: float = 1e-12) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    xs = torch.tensor([float(v) for v in vals], dtype=torch.float64)
    xs = xs - xs.max()
    p = torch.softmax(xs, dim=0)
    ent = -(p * torch.log(p.clamp_min(float(eps)))).sum().item()
    return float(ent / math.log(max(2, len(vals))))


def _jaccard(a: Sequence[int] | set[int], b: Sequence[int] | set[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _prefix_count(pred: Sequence[int], query_nodes: Sequence[int], ranked_nodes: Sequence[int]) -> int:
    pred_set = set(int(x) for x in pred)
    cur = set(int(x) for x in query_nodes)
    if pred_set <= cur:
        return 0
    for k, node in enumerate(ranked_nodes, start=1):
        cur.add(int(node))
        if pred_set <= cur:
            return k
    return len(ranked_nodes)


def _candidate_stats(result: ChunkedScoreResult, qi: int) -> BranchCandidateStats:
    qset = tuple(int(x) for x in result.query_node_sets[int(qi)])
    ranked_nodes = [int(x) for x in result.top_nodes[int(qi)].tolist()]
    ranked_scores = [float(x) for x in result.top_scores[int(qi)].tolist()]
    q_scores = [float(result.query_seed_scores[int(qi)][int(seed)]) for seed in qset]
    pred = senior_global_search_from_ranked_scores(
        query_nodes=qset,
        query_scores=q_scores,
        ranked_nodes=ranked_nodes,
        ranked_scores=ranked_scores,
        num_nodes=int(result.num_nodes),
        full_score_sum=float(result.full_score_sums[int(qi)].item()),
    )
    prefix_k = _prefix_count(pred, qset, ranked_nodes)
    size = max(1, len(pred))
    cand_sum = float(sum(q_scores) + sum(ranked_scores[:prefix_k]))
    density = subgraph_density_controlled(cand_sum, size, full_score_sum=float(result.full_score_sums[int(qi)].item()), num_nodes=int(result.num_nodes))
    if prefix_k > 0 and prefix_k < len(ranked_scores):
        gap = float(ranked_scores[prefix_k - 1] - ranked_scores[prefix_k])
    else:
        gap = 0.0
    return BranchCandidateStats(
        size=int(size),
        density=float(density),
        boundary_gap=float(gap),
        entropy=_entropy(ranked_scores[: min(256, len(ranked_scores))]),
        nodes=tuple(int(x) for x in pred),
    )


def _score_result_from_branch(
    *,
    manifest_path: str | Path,
    query_node_sets: SeniorQueryBatch,
    chunk_size: int,
    top_k_pool: int | None,
    runtime_jsonl: str | Path | None,
    score_calibration: str = "zscore",
) -> ChunkedScoreResult:
    # Reuse existing shard scorer for branch diagnostics.  Z-score calibration
    # makes preserved/adapted branches comparable and avoids raw scale leakage.
    from .embedding_store import chunked_query_cosine_from_embedding_shards

    return chunked_query_cosine_from_embedding_shards(
        manifest_path=manifest_path,
        query_node_sets=query_node_sets,
        chunk_size=chunk_size,
        top_k_pool=top_k_pool,
        runtime_jsonl=runtime_jsonl,
        score_calibration=score_calibration,
    )


def _load_query_embs(manifest_path: str | Path, qsets: tuple[tuple[int, ...], ...], *, eps: float = 1e-12) -> torch.Tensor:
    manifest_file, manifest = load_embedding_manifest(manifest_path)
    n = int(manifest["num_nodes"])
    _normalize_query_sets(qsets, num_nodes=n)
    seed_to_queries: dict[int, list[int]] = {}
    for qi, qset in enumerate(qsets):
        for seed in qset:
            seed_to_queries.setdefault(int(seed), []).append(qi)
    seed_vectors: dict[int, torch.Tensor] = {}
    for row in manifest.get("shards", []):
        start, end = int(row["start"]), int(row["end"])
        needed = [seed for seed in seed_to_queries if start <= seed < end]
        if not needed:
            continue
        emb = _load_shard_tensor(manifest_file.parent / str(row["path"]), start=start, end=end).float()
        emb = F.normalize(emb, p=2, dim=-1, eps=eps)
        for seed in needed:
            seed_vectors[int(seed)] = emb[int(seed) - start].detach().cpu()
    query_embs = []
    for qset in qsets:
        missing = [int(seed) for seed in qset if int(seed) not in seed_vectors]
        if missing:
            raise V5ProtocolError(f"query seeds missing from embedding manifest: {missing[:5]}")
        query_embs.append(torch.stack([seed_vectors[int(seed)] for seed in qset], dim=0).mean(dim=0))
    return F.normalize(torch.stack(query_embs, dim=0).float(), p=2, dim=-1, eps=eps)


def _branch_stats(
    *,
    manifest_path: str | Path,
    query_embs: torch.Tensor,
    chunk_size: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    manifest_file, manifest = load_embedding_manifest(manifest_path)
    q_count = int(query_embs.shape[0])
    n = int(manifest["num_nodes"])
    sum1 = torch.zeros(q_count, dtype=torch.float64)
    sum2 = torch.zeros(q_count, dtype=torch.float64)
    for row in manifest.get("shards", []):
        start, end = int(row["start"]), int(row["end"])
        shard = _load_shard_tensor(manifest_file.parent / str(row["path"]), start=start, end=end).float()
        for local_start in range(0, int(shard.shape[0]), int(chunk_size)):
            local_end = min(int(shard.shape[0]), local_start + int(chunk_size))
            chunk = F.normalize(shard[local_start:local_end], p=2, dim=-1, eps=eps)
            scores = query_embs @ chunk.t()
            sum1 += scores.double().sum(dim=1)
            sum2 += scores.double().pow(2).sum(dim=1)
    mean = sum1 / float(n)
    std = torch.sqrt(torch.clamp(sum2 / float(n) - mean.pow(2), min=1e-12))
    return mean, std


def _adaptive_alpha(
    *,
    preserved: BranchCandidateStats,
    adapted: BranchCandidateStats,
    num_nodes: int,
    cfg: DualBranchGateConfig,
) -> tuple[float, dict[str, float]]:
    ref_size = max(float(cfg.min_size_reference), float(num_nodes) ** float(cfg.size_reference_power))
    over_pres = max(0.0, math.log((float(preserved.size) + cfg.eps) / (ref_size + cfg.eps)))
    under_adapt_vs_pres = max(0.0, math.log((float(preserved.size) + cfg.eps) / (float(adapted.size) + cfg.eps)))
    over_adapt = max(0.0, math.log((float(adapted.size) + cfg.eps) / (ref_size + cfg.eps)))
    density_delta = float(adapted.density - preserved.density)
    disagreement = 1.0 - _jaccard(preserved.nodes, adapted.nodes)
    entropy = max(float(preserved.entropy), float(adapted.entropy))
    boundary = max(0.0, float(adapted.boundary_gap))
    # Symmetric query-level guard.  The original V5.5 adaptive alpha only
    # discouraged adapted candidates that were much smaller than preserved ones.
    # V5.6-lite keeps that, but also adds a mild over/under-size state gate so
    # that under-selection and over-expansion are both visible in diagnostics.
    query_under_guard = 0.0
    if bool(cfg.target_state_gate) and under_adapt_vs_pres > 0.0 and density_delta < float(cfg.density_gain_tolerance):
        query_under_guard = under_adapt_vs_pres
    query_over_guard = 0.0
    if bool(cfg.target_state_gate) and over_pres > 0.0 and density_delta > -float(cfg.density_gain_tolerance):
        query_over_guard = over_pres
    raw_alpha = (
        float(cfg.base_alpha)
        + float(cfg.over_preserved_weight) * over_pres
        + float(cfg.adapted_density_weight) * density_delta
        + float(cfg.boundary_gap_adapt_weight) * boundary
        + float(cfg.query_oversize_guard_weight) * query_over_guard
        - float(cfg.entropy_preserve_weight) * entropy
        - float(cfg.disagreement_preserve_weight) * disagreement
        - float(cfg.adapted_undersize_weight) * under_adapt_vs_pres
        - float(cfg.adapted_oversize_weight) * over_adapt
        - float(cfg.query_undersize_guard_weight) * query_under_guard
    )
    alpha = max(float(cfg.min_alpha), min(float(cfg.max_alpha), float(raw_alpha)))
    return alpha, {
        "raw_alpha": float(raw_alpha),
        "over_preserved": float(over_pres),
        "adapted_under_size_vs_preserved": float(under_adapt_vs_pres),
        "adapted_over_size": float(over_adapt),
        "density_delta_adapt_minus_preserve": float(density_delta),
        "branch_disagreement": float(disagreement),
        "branch_entropy": float(entropy),
        "adapted_boundary_gap": float(boundary),
        "query_under_guard": float(query_under_guard),
        "query_over_guard": float(query_over_guard),
    }



def _mean_float(rows: Sequence[Mapping[str, object]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0.0)))
        except Exception:
            pass
    return float(sum(vals) / max(1, len(vals)))


def _compute_graph_state_bias(rows: Sequence[Mapping[str, object]], cfg: DualBranchGateConfig) -> tuple[float, dict[str, float]]:
    """Return a label-free graph-level alpha bias.

    Positive bias means the whole target appears to need more adapted ranking;
    negative bias means the target appears fragile or under-selected and should
    preserve more cache/epoch0 ranking.  This is not target-name-specific and
    never reads gt.
    """
    if not bool(cfg.graph_bias_enabled) or not rows:
        return 0.0, {
            "graph_bias_raw": 0.0,
            "graph_bias": 0.0,
            "graph_mean_over_preserved": 0.0,
            "graph_mean_adapted_undersize": 0.0,
            "graph_mean_density_delta": 0.0,
            "graph_mean_disagreement": 0.0,
            "graph_mean_entropy": 0.0,
        }
    mean_over_pres = _mean_float(rows, "over_preserved")
    mean_under = _mean_float(rows, "adapted_under_size_vs_preserved")
    mean_density = _mean_float(rows, "density_delta_adapt_minus_preserve")
    mean_disagree = _mean_float(rows, "branch_disagreement")
    mean_entropy = _mean_float(rows, "branch_entropy")
    raw = (
        float(cfg.graph_over_preserved_weight) * mean_over_pres
        + float(cfg.graph_adapted_density_weight) * mean_density
        - float(cfg.graph_adapted_undersize_weight) * mean_under
        - float(cfg.graph_disagreement_preserve_weight) * mean_disagree
        - float(cfg.graph_entropy_preserve_weight) * mean_entropy
    )
    bias = max(-float(cfg.graph_bias_clip), min(float(cfg.graph_bias_clip), float(cfg.graph_bias_strength) * float(raw)))
    return float(bias), {
        "graph_bias_raw": float(raw),
        "graph_bias": float(bias),
        "graph_mean_over_preserved": float(mean_over_pres),
        "graph_mean_adapted_undersize": float(mean_under),
        "graph_mean_density_delta": float(mean_density),
        "graph_mean_disagreement": float(mean_disagree),
        "graph_mean_entropy": float(mean_entropy),
    }




def _mean_stat(rows: Sequence[Mapping[str, object]], key: str, default: float = 0.0) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row.get(key, default)))
        except Exception:
            pass
    return float(sum(vals) / max(1, len(vals)))


def _select_state_profile_from_branch_rows(rows: Sequence[Mapping[str, object]], cfg: DualBranchGateConfig) -> dict[str, object]:
    """Choose a label-free state-conditioned parameter profile.

    V5.7.2 adds an optional revised recognizer.  The old recognizer often
    collapsed Citeseer-like medium-recall targets into boundary-preserve, so
    score/recovery packages designed for moderate recall were never exercised.
    The revised recognizer separates states with *unsupervised* branch
    statistics only: size-ratio distributions, branch overlap, density shift,
    disagreement, and entropy.  It never checks target graph names or labels.
    """
    def payload(
        profile_id: str,
        reason: str,
        *,
        alpha_bias: float,
        core_weight: float,
        structure_weight: float,
        search_mode: str,
        size_ratio: float,
        branch_jaccard: float,
        density_delta: float,
        disagreement: float,
        entropy: float,
        route_version: str = "legacy",
        adapted_over_rate: float = 0.0,
        adapted_strong_rate: float = 0.0,
        stable_close_rate: float = 0.0,
        moderate_eligible_rate: float = 0.0,
        shift_eligible_rate: float = 0.0,
    ) -> dict[str, object]:
        return {
            "profile_id": str(profile_id),
            "reason": str(reason),
            "alpha_bias": float(alpha_bias),
            "core_weight": float(core_weight),
            "structure_weight": float(structure_weight),
            "search_mode": str(search_mode),
            "target_profile_size_ratio": float(size_ratio),
            "target_profile_branch_jaccard": float(branch_jaccard),
            "target_profile_density_delta": float(density_delta),
            "target_profile_density_delta_abs": float(abs(density_delta)),
            "target_profile_disagreement": float(disagreement),
            "target_profile_entropy": float(entropy),
            "target_profile_route_version": str(route_version),
            "target_profile_adapted_over_rate": float(adapted_over_rate),
            "target_profile_adapted_strong_rate": float(adapted_strong_rate),
            "target_profile_stable_close_rate": float(stable_close_rate),
            "target_profile_moderate_eligible_rate": float(moderate_eligible_rate),
            "target_profile_shift_eligible_rate": float(shift_eligible_rate),
        }

    if not bool(cfg.state_profile_router_enabled) or not rows:
        return payload(
            "disabled",
            "state_profile_router_disabled",
            alpha_bias=0.0,
            core_weight=0.0,
            structure_weight=0.0,
            search_mode="default",
            size_ratio=1.0,
            branch_jaccard=1.0,
            density_delta=0.0,
            disagreement=0.0,
            entropy=0.0,
            route_version="disabled",
        )

    def _safe_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except Exception:
            return float(default)

    def _row_size_ratio(row: Mapping[str, object]) -> float:
        ps = max(_safe_float(row, "preserved_size", 1.0), float(cfg.eps))
        ad = max(_safe_float(row, "adapted_size", 1.0), 0.0)
        return float(ad + cfg.eps) / float(ps)

    def _rate(pred) -> float:
        hit = 0
        total = 0
        for row in rows:
            total += 1
            try:
                if pred(row):
                    hit += 1
            except Exception:
                pass
        return float(hit) / float(max(1, total))

    preserved_size = _mean_stat(rows, "preserved_size", 1.0)
    adapted_size = _mean_stat(rows, "adapted_size", 1.0)
    size_ratio = float(adapted_size + cfg.eps) / float(max(preserved_size, cfg.eps))
    branch_jaccard = _mean_stat(rows, "preserved_adapted_jaccard", 1.0)
    density_delta = _mean_stat(rows, "density_delta_adapt_minus_preserve", 0.0)
    disagreement = _mean_stat(rows, "branch_disagreement", 1.0 - branch_jaccard)
    entropy = _mean_stat(rows, "branch_entropy", 0.0)

    adapted_over_rate = _rate(lambda r: _row_size_ratio(r) >= float(cfg.state_profile_moderate_size_ratio_min))
    adapted_strong_rate = _rate(lambda r: _row_size_ratio(r) >= float(cfg.state_profile_strong_size_ratio_min))
    stable_close_rate = _rate(
        lambda r: float(cfg.state_profile_stable_tight_size_ratio_low)
        <= _row_size_ratio(r)
        <= float(cfg.state_profile_stable_tight_size_ratio_high)
    )
    moderate_eligible_rate = _rate(
        lambda r: _row_size_ratio(r) >= float(cfg.state_profile_moderate_size_ratio_min)
        and _safe_float(r, "preserved_adapted_jaccard", 1.0) <= float(cfg.state_profile_moderate_branch_jaccard_max)
    )
    shift_eligible_rate = _rate(
        lambda r: _safe_float(r, "density_delta_adapt_minus_preserve", 0.0) >= float(cfg.state_profile_shift_density_delta_min)
        and _safe_float(r, "preserved_adapted_jaccard", 1.0) >= float(cfg.state_profile_shift_branch_jaccard_min)
    )

    route_version = "v572" if bool(cfg.state_profile_state_recognition_v2_enabled) else "legacy"

    if bool(cfg.state_profile_state_recognition_v2_enabled):
        if (
            size_ratio >= float(cfg.state_profile_strong_size_ratio_min)
            and branch_jaccard <= float(cfg.state_profile_strong_branch_jaccard_max)
        ) or (
            adapted_strong_rate >= 0.35
            and branch_jaccard <= float(cfg.state_profile_strong_branch_jaccard_max)
        ):
            return payload(
                "strong_recovery_profile",
                (
                    f"v572_strong_size_ratio_{size_ratio:.4f}_ge_{float(cfg.state_profile_strong_size_ratio_min):.4f}_"
                    f"or_strong_rate_{adapted_strong_rate:.4f}_ge_0.35_"
                    f"branch_jaccard_{branch_jaccard:.4f}_le_{float(cfg.state_profile_strong_branch_jaccard_max):.4f}"
                ),
                alpha_bias=float(cfg.state_profile_strong_alpha_bias),
                core_weight=float(cfg.state_profile_core_weight_strong),
                structure_weight=float(cfg.state_profile_structure_weight_strong),
                search_mode="strong_recovery",
                size_ratio=size_ratio,
                branch_jaccard=branch_jaccard,
                density_delta=density_delta,
                disagreement=disagreement,
                entropy=entropy,
                route_version=route_version,
                adapted_over_rate=adapted_over_rate,
                adapted_strong_rate=adapted_strong_rate,
                stable_close_rate=stable_close_rate,
                moderate_eligible_rate=moderate_eligible_rate,
                shift_eligible_rate=shift_eligible_rate,
            )

        stable_high_overlap = (
            float(cfg.state_profile_stable_size_ratio_low) <= size_ratio <= float(cfg.state_profile_stable_size_ratio_high)
            and branch_jaccard >= float(cfg.state_profile_stable_branch_jaccard_min)
            and disagreement <= 0.38
            and abs(density_delta) <= float(cfg.state_profile_stable_density_delta_abs_max)
            and entropy >= float(cfg.state_profile_stable_entropy_min)
        )
        stable_tight_size = (
            float(cfg.state_profile_stable_tight_size_ratio_low) <= size_ratio <= float(cfg.state_profile_stable_tight_size_ratio_high)
            and branch_jaccard >= float(cfg.state_profile_stable_tight_branch_jaccard_min)
            and disagreement <= 0.45
            and abs(density_delta) <= float(cfg.state_profile_stable_density_delta_abs_max)
            and entropy >= float(cfg.state_profile_stable_entropy_min)
        )
        if stable_high_overlap or stable_tight_size:
            return payload(
                "stable_default_profile",
                (
                    f"v572_stable_size_ratio_{size_ratio:.4f}_branch_jaccard_{branch_jaccard:.4f}_"
                    f"entropy_{entropy:.4f}_ge_{float(cfg.state_profile_stable_entropy_min):.4f}_"
                    f"density_abs_{abs(density_delta):.4f}_le_{float(cfg.state_profile_stable_density_delta_abs_max):.4f}_"
                    f"stable_close_rate_{stable_close_rate:.4f}"
                ),
                alpha_bias=float(cfg.state_profile_stable_alpha_bias),
                core_weight=float(cfg.state_profile_core_weight_stable),
                structure_weight=float(cfg.state_profile_structure_weight_stable),
                search_mode="stable_default",
                size_ratio=size_ratio,
                branch_jaccard=branch_jaccard,
                density_delta=density_delta,
                disagreement=disagreement,
                entropy=entropy,
                route_version=route_version,
                adapted_over_rate=adapted_over_rate,
                adapted_strong_rate=adapted_strong_rate,
                stable_close_rate=stable_close_rate,
                moderate_eligible_rate=moderate_eligible_rate,
                shift_eligible_rate=shift_eligible_rate,
            )

        if (
            size_ratio >= float(cfg.state_profile_moderate_size_ratio_min)
            and branch_jaccard <= float(cfg.state_profile_moderate_branch_jaccard_max)
            and adapted_over_rate >= float(cfg.state_profile_moderate_over_rate_min)
            and density_delta >= float(cfg.state_profile_moderate_density_delta_min)
            and entropy >= float(cfg.state_profile_moderate_entropy_min)
        ):
            return payload(
                "moderate_recall_profile",
                (
                    f"v572_moderate_size_ratio_{size_ratio:.4f}_ge_{float(cfg.state_profile_moderate_size_ratio_min):.4f}_"
                    f"branch_jaccard_{branch_jaccard:.4f}_le_{float(cfg.state_profile_moderate_branch_jaccard_max):.4f}_"
                    f"adapted_over_rate_{adapted_over_rate:.4f}_ge_{float(cfg.state_profile_moderate_over_rate_min):.4f}_"
                    f"density_delta_{density_delta:.4f}_ge_{float(cfg.state_profile_moderate_density_delta_min):.4f}"
                ),
                alpha_bias=float(cfg.state_profile_moderate_alpha_bias),
                core_weight=float(cfg.state_profile_core_weight_moderate),
                structure_weight=float(cfg.state_profile_structure_weight_moderate),
                search_mode="moderate_recall",
                size_ratio=size_ratio,
                branch_jaccard=branch_jaccard,
                density_delta=density_delta,
                disagreement=disagreement,
                entropy=entropy,
                route_version=route_version,
                adapted_over_rate=adapted_over_rate,
                adapted_strong_rate=adapted_strong_rate,
                stable_close_rate=stable_close_rate,
                moderate_eligible_rate=moderate_eligible_rate,
                shift_eligible_rate=shift_eligible_rate,
            )

        if density_delta >= float(cfg.state_profile_shift_density_delta_min) and branch_jaccard >= float(cfg.state_profile_shift_branch_jaccard_min):
            return payload(
                "shift_adaptive_profile",
                f"v572_shift_density_delta_{density_delta:.4f}_ge_{float(cfg.state_profile_shift_density_delta_min):.4f}_branch_jaccard_{branch_jaccard:.4f}",
                alpha_bias=float(cfg.state_profile_shift_alpha_bias),
                core_weight=float(cfg.state_profile_core_weight_shift),
                structure_weight=float(cfg.state_profile_structure_weight_shift),
                search_mode="shift_adaptive",
                size_ratio=size_ratio,
                branch_jaccard=branch_jaccard,
                density_delta=density_delta,
                disagreement=disagreement,
                entropy=entropy,
                route_version=route_version,
                adapted_over_rate=adapted_over_rate,
                adapted_strong_rate=adapted_strong_rate,
                stable_close_rate=stable_close_rate,
                moderate_eligible_rate=moderate_eligible_rate,
                shift_eligible_rate=shift_eligible_rate,
            )

        return payload(
            "boundary_preserve_profile",
            (
                f"v572_boundary_fallback_size_ratio_{size_ratio:.4f}_branch_jaccard_{branch_jaccard:.4f}_"
                f"density_delta_{density_delta:.4f}_moderate_eligible_rate_{moderate_eligible_rate:.4f}"
            ),
            alpha_bias=float(cfg.state_profile_boundary_alpha_bias),
            core_weight=float(cfg.state_profile_core_weight_boundary),
            structure_weight=float(cfg.state_profile_structure_weight_boundary),
            search_mode="boundary_preserve",
            size_ratio=size_ratio,
            branch_jaccard=branch_jaccard,
            density_delta=density_delta,
            disagreement=disagreement,
            entropy=entropy,
            route_version=route_version,
            adapted_over_rate=adapted_over_rate,
            adapted_strong_rate=adapted_strong_rate,
            stable_close_rate=stable_close_rate,
            moderate_eligible_rate=moderate_eligible_rate,
        )

    # Legacy V5.6.8--V5.7.1 recognizer, retained so earlier presets remain
    # directly comparable.
    if size_ratio >= 1.35 and branch_jaccard <= 0.72:
        return payload(
            "strong_recovery_profile",
            f"adapted_preserved_size_ratio_{size_ratio:.4f}_ge_1.35_branch_jaccard_{branch_jaccard:.4f}_le_0.72",
            alpha_bias=float(cfg.state_profile_strong_alpha_bias),
            core_weight=float(cfg.state_profile_core_weight_strong),
            structure_weight=float(cfg.state_profile_structure_weight_strong),
            search_mode="strong_recovery",
            size_ratio=size_ratio,
            branch_jaccard=branch_jaccard,
            density_delta=density_delta,
            disagreement=disagreement,
            entropy=entropy,
            route_version=route_version,
            adapted_over_rate=adapted_over_rate,
            adapted_strong_rate=adapted_strong_rate,
            stable_close_rate=stable_close_rate,
            moderate_eligible_rate=moderate_eligible_rate,
        )

    if 0.85 <= size_ratio <= 1.20 and branch_jaccard >= 0.75 and abs(density_delta) < 0.04 and disagreement <= 0.35:
        return payload(
            "stable_default_profile",
            f"stable_size_ratio_{size_ratio:.4f}_branch_jaccard_{branch_jaccard:.4f}_density_delta_{density_delta:.4f}",
            alpha_bias=float(cfg.state_profile_stable_alpha_bias),
            core_weight=float(cfg.state_profile_core_weight_stable),
            structure_weight=float(cfg.state_profile_structure_weight_stable),
            search_mode="stable_default",
            size_ratio=size_ratio,
            branch_jaccard=branch_jaccard,
            density_delta=density_delta,
            disagreement=disagreement,
            entropy=entropy,
            route_version=route_version,
            adapted_over_rate=adapted_over_rate,
            adapted_strong_rate=adapted_strong_rate,
            stable_close_rate=stable_close_rate,
            moderate_eligible_rate=moderate_eligible_rate,
        )

    if density_delta >= 0.04 and branch_jaccard >= 0.70:
        return payload(
            "shift_adaptive_profile",
            f"density_delta_{density_delta:.4f}_ge_0.04_branch_jaccard_{branch_jaccard:.4f}_ge_0.70",
            alpha_bias=float(cfg.state_profile_shift_alpha_bias),
            core_weight=float(cfg.state_profile_core_weight_shift),
            structure_weight=float(cfg.state_profile_structure_weight_shift),
            search_mode="shift_adaptive",
            size_ratio=size_ratio,
            branch_jaccard=branch_jaccard,
            density_delta=density_delta,
            disagreement=disagreement,
            entropy=entropy,
            route_version=route_version,
            adapted_over_rate=adapted_over_rate,
            adapted_strong_rate=adapted_strong_rate,
            stable_close_rate=stable_close_rate,
            moderate_eligible_rate=moderate_eligible_rate,
        )

    if branch_jaccard <= 0.65 and density_delta >= -0.02:
        return payload(
            "moderate_recall_profile",
            f"branch_jaccard_{branch_jaccard:.4f}_le_0.65_density_delta_{density_delta:.4f}_ge_-0.02",
            alpha_bias=float(cfg.state_profile_moderate_alpha_bias),
            core_weight=float(cfg.state_profile_core_weight_moderate),
            structure_weight=float(cfg.state_profile_structure_weight_moderate),
            search_mode="moderate_recall",
            size_ratio=size_ratio,
            branch_jaccard=branch_jaccard,
            density_delta=density_delta,
            disagreement=disagreement,
            entropy=entropy,
            route_version=route_version,
            adapted_over_rate=adapted_over_rate,
            adapted_strong_rate=adapted_strong_rate,
            stable_close_rate=stable_close_rate,
            moderate_eligible_rate=moderate_eligible_rate,
        )

    if branch_jaccard >= 0.65 or density_delta < -0.02:
        return payload(
            "boundary_preserve_profile",
            f"branch_jaccard_{branch_jaccard:.4f}_ge_0.65_or_density_delta_{density_delta:.4f}_lt_-0.02",
            alpha_bias=float(cfg.state_profile_boundary_alpha_bias),
            core_weight=float(cfg.state_profile_core_weight_boundary),
            structure_weight=float(cfg.state_profile_structure_weight_boundary),
            search_mode="boundary_preserve",
            size_ratio=size_ratio,
            branch_jaccard=branch_jaccard,
            density_delta=density_delta,
            disagreement=disagreement,
            entropy=entropy,
            route_version=route_version,
            adapted_over_rate=adapted_over_rate,
            adapted_strong_rate=adapted_strong_rate,
            stable_close_rate=stable_close_rate,
            moderate_eligible_rate=moderate_eligible_rate,
        )

    return payload(
        "stable_default_profile",
        "fallback_stable_default",
        alpha_bias=float(cfg.state_profile_stable_alpha_bias),
        core_weight=float(cfg.state_profile_core_weight_stable),
        structure_weight=float(cfg.state_profile_structure_weight_stable),
        search_mode="stable_default",
        size_ratio=size_ratio,
        branch_jaccard=branch_jaccard,
        density_delta=density_delta,
        disagreement=disagreement,
        entropy=entropy,
        route_version=route_version,
        adapted_over_rate=adapted_over_rate,
        adapted_strong_rate=adapted_strong_rate,
        stable_close_rate=stable_close_rate,
        moderate_eligible_rate=moderate_eligible_rate,
    )

def _core_weight_for_profile(profile: Mapping[str, object], cfg: DualBranchGateConfig) -> float:
    if not bool(cfg.state_profile_core_proto_enabled):
        return 0.0
    return max(0.0, min(1.0, float(profile.get("core_weight", 0.0))))


def _structure_weight_for_profile(profile: Mapping[str, object], cfg: DualBranchGateConfig) -> float:
    if not bool(cfg.state_profile_structure_enabled):
        return 0.0
    return max(0.0, float(profile.get("structure_weight", 0.0)))


def _profile_id_from_obj(profile_or_id: Mapping[str, object] | str | None) -> str:
    if isinstance(profile_or_id, str):
        return profile_or_id
    if isinstance(profile_or_id, Mapping):
        return str(profile_or_id.get("profile_id", "stable_default_profile"))
    return "stable_default_profile"


def _apply_state_profile_alpha_package(alpha: float, profile_or_id: Mapping[str, object] | str | None, cfg: DualBranchGateConfig) -> tuple[float, str]:
    """Apply state-specific branch tendency as a complete profile package.

    Historical experiments showed that some states prefer preservation while
    others prefer adapted/guided rankings.  This clamp is the inference-time
    equivalent of those training-strength findings and remains label-free.
    """
    profile_id = _profile_id_from_obj(profile_or_id)
    out = float(alpha)
    reason = "none"
    if not bool(cfg.state_profile_alpha_bounds_enabled):
        return max(float(cfg.min_alpha), min(float(cfg.max_alpha), out)), "bounds_disabled"
    if profile_id == "strong_recovery_profile":
        before = out
        out = max(out, float(cfg.state_profile_strong_alpha_min))
        reason = f"strong_min_{float(cfg.state_profile_strong_alpha_min):.4f}" if out != before else "strong_no_change"
    elif profile_id == "moderate_recall_profile":
        before = out
        out = max(float(cfg.state_profile_moderate_alpha_min), min(float(cfg.state_profile_moderate_alpha_max), out))
        reason = f"moderate_range_{float(cfg.state_profile_moderate_alpha_min):.4f}_{float(cfg.state_profile_moderate_alpha_max):.4f}" if out != before else "moderate_no_change"
    elif profile_id in {
        "boundary_preserve_profile",
        "boundary_plateau_precision_profile",
        "boundary_margin_sensitive_profile",
    }:
        before = out
        out = min(out, float(cfg.state_profile_boundary_alpha_max))
        reason = f"boundary_max_{float(cfg.state_profile_boundary_alpha_max):.4f}" if out != before else "boundary_no_change"
    elif profile_id == "shift_adaptive_profile":
        before = out
        out = max(out, float(cfg.state_profile_shift_alpha_min))
        reason = f"shift_min_{float(cfg.state_profile_shift_alpha_min):.4f}" if out != before else "shift_no_change"
    else:
        reason = "stable_default_no_change"
    out = max(float(cfg.min_alpha), min(float(cfg.max_alpha), float(out)))
    return float(out), reason



def _state_profile_core_source_mode_for_profile(
    profile_or_id: Mapping[str, object] | str | None,
    cfg: DualBranchGateConfig,
) -> str:
    """Return the core source mode selected by the state package.

    This is intentionally state-based rather than target-name-based.  It fixes
    the V5.6.9 limitation where all states shared a single core source mode.
    """
    if not bool(cfg.state_profile_core_source_by_state_enabled):
        return str(cfg.state_profile_core_source_mode)
    profile_id = _profile_id_from_obj(profile_or_id)
    if profile_id == "strong_recovery_profile":
        return str(cfg.state_profile_core_source_mode_strong)
    if profile_id == "moderate_recall_profile":
        return str(cfg.state_profile_core_source_mode_moderate)
    if profile_id in {
        "boundary_preserve_profile",
        "boundary_plateau_precision_profile",
        "boundary_margin_sensitive_profile",
    }:
        return str(cfg.state_profile_core_source_mode_boundary)
    if profile_id == "shift_adaptive_profile":
        return str(cfg.state_profile_core_source_mode_shift)
    if profile_id == "stable_default_profile":
        return str(cfg.state_profile_core_source_mode_stable)
    return str(cfg.state_profile_core_source_mode)

def _state_profile_score_mode_for_profile(
    profile_or_id: Mapping[str, object] | str | None,
    cfg: DualBranchGateConfig,
) -> str:
    if not bool(cfg.state_profile_score_source_enabled):
        return "default_fusion"
    profile_id = _profile_id_from_obj(profile_or_id)
    if profile_id == "strong_recovery_profile":
        return str(cfg.state_profile_score_mode_strong)
    if profile_id == "moderate_recall_profile":
        return str(cfg.state_profile_score_mode_moderate)
    if profile_id == "boundary_plateau_precision_profile":
        return str(cfg.state_profile_score_mode_boundary)
    if profile_id in {"boundary_preserve_profile", "boundary_margin_sensitive_profile"}:
        # The parent/fallback state and its margin-sensitive child are safe
        # probes.  Only the plateau child may execute the precision action.
        if bool(cfg.hierarchical_boundary_state_enabled):
            return "default_fusion"
        return str(cfg.state_profile_score_mode_boundary)
    if profile_id == "shift_adaptive_profile":
        return str(cfg.state_profile_score_mode_shift)
    if profile_id == "stable_default_profile":
        return str(cfg.state_profile_score_mode_stable)
    return "default_fusion"


def _state_profile_score_residual_scale_for_profile(
    profile_or_id: Mapping[str, object] | str | None,
    cfg: DualBranchGateConfig,
) -> float:
    if not bool(cfg.state_profile_score_source_enabled):
        return 1.0
    profile_id = _profile_id_from_obj(profile_or_id)
    if profile_id == "strong_recovery_profile":
        return float(cfg.state_profile_score_residual_scale_strong)
    if profile_id == "moderate_recall_profile":
        return float(cfg.state_profile_score_residual_scale_moderate)
    if profile_id in {
        "boundary_preserve_profile",
        "boundary_plateau_precision_profile",
        "boundary_margin_sensitive_profile",
    }:
        return float(cfg.state_profile_score_residual_scale_boundary)
    if profile_id == "shift_adaptive_profile":
        return float(cfg.state_profile_score_residual_scale_shift)
    if profile_id == "stable_default_profile":
        return float(cfg.state_profile_score_residual_scale_stable)
    return 1.0


def _state_profile_score_action_allowed(
    profile_or_id: Mapping[str, object] | str | None,
    cfg: DualBranchGateConfig,
) -> bool:
    """Return whether the recognized profile may execute its score action.

    ``profile_specific`` remains isolated: it permits only a profile whose own
    configured mode is non-default, rather than broadcasting one action to all
    targets.  No target name, target label, or ground truth is consulted.
    """
    if not bool(cfg.state_profile_action_isolation_enabled):
        return True
    profile_id = _profile_id_from_obj(profile_or_id)
    scope = str(cfg.state_profile_action_isolation_scope)
    if scope == "moderate_only":
        return profile_id == "moderate_recall_profile"
    if scope == "profile_specific":
        return _state_profile_score_mode_for_profile(profile_or_id, cfg) != "default_fusion"
    raise V5ProtocolError(f"unsupported state-profile action isolation scope={scope!r}")


def _apply_state_profile_score_source(
    *,
    zp: torch.Tensor,
    za: torch.Tensor,
    alpha: torch.Tensor,
    cfg: DualBranchGateConfig,
    profile_or_id: Mapping[str, object] | str | None,
) -> torch.Tensor:
    """Apply the state-conditioned score-source package.

    This is the inference-time counterpart of historical hyperparameter
    findings.  It changes the node ranking itself rather than only changing the
    selected prefix length: moderate-recall states use weaker adapted residuals
    plus a small recall union term, boundary-sensitive states preserve more of
    the preserved branch and reward branch consensus, and shift/strong states
    lean more toward the adapted branch.
    """
    profile_id = _profile_id_from_obj(profile_or_id)
    mode = _state_profile_score_mode_for_profile(profile_or_id, cfg)
    scale = _state_profile_score_residual_scale_for_profile(profile_or_id, cfg)
    # Historical action-isolated presets allow only moderate-recall actions.
    # The opt-in profile-specific scope instead permits each recognized state
    # to execute only its own explicitly configured non-default action.  This
    # keeps recall and boundary actions isolated from unrelated profiles.
    if not _state_profile_score_action_allowed(profile_or_id, cfg):
        mode = "default_fusion"
        scale = 1.0
    eff_alpha = torch.clamp(alpha * float(scale), 0.0, 1.0)
    base = zp + float(cfg.residual_gain) * eff_alpha * (za - zp)
    if not bool(cfg.state_profile_score_source_enabled) or mode == "default_fusion":
        return base
    if mode == "adapt_preferred":
        # Strong/shift states keep the same residual form but use a larger
        # effective alpha and a small adapted-branch bonus so the ranking truly
        # moves toward the guided branch.
        bonus = 0.0
        if profile_id == "shift_adaptive_profile":
            bonus = float(cfg.state_profile_score_adapt_bonus_shift)
        return base + float(bonus) * za
    if mode == "weak_adapt_residual":
        # Moderate recall states should not over-trust the adapted residual, but
        # they can benefit from nodes that either branch ranks highly.
        return base + float(cfg.state_profile_score_recall_weight_moderate) * torch.maximum(zp, za)
    if mode == "preserve_consensus":
        # Boundary-sensitive states are conservative: preserve the branch closer
        # to epoch0/raw ranking and give a small bonus to nodes supported by both
        # branches.  This differs from merely clamping alpha because the
        # consensus term changes local boundary ordering.
        return base + float(cfg.state_profile_score_consensus_weight_boundary) * torch.minimum(zp, za)
    return base


def _core_reliability_tensor(
    *,
    q_pres: torch.Tensor,
    q_adapt: torch.Tensor,
    pres_core: torch.Tensor,
    adapt_core: torch.Tensor,
    core_nodes: Sequence[Sequence[int]],
    agreement_details: Sequence[Mapping[str, object]],
    cfg: DualBranchGateConfig,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Return per-query reliability for state-profile core prototypes.

    Reliability is label-free.  It rewards enough core nodes, candidate-level
    agreement between preserved/adapted branches, and consistency between the
    core prototype and the original query prototype.  It downweights cores that
    are tiny or depend heavily on rank-only fallback.
    """
    weights_sum = float(cfg.state_profile_core_reliability_agreement_weight) + float(cfg.state_profile_core_reliability_count_weight) + float(cfg.state_profile_core_reliability_query_sim_weight)
    if weights_sum <= 0.0:
        weights_sum = 1.0
    out: list[float] = []
    rows: list[dict[str, float]] = []
    min_nodes = max(1, int(cfg.state_profile_core_reliability_min_nodes))
    for qi, nodes in enumerate(core_nodes):
        detail = agreement_details[qi] if qi < len(agreement_details) else {}
        core_count = int(len(nodes))
        cand_agree = int(detail.get("candidate_agreement_count", 0))
        used_fallback = bool(detail.get("used_rank_fallback", False))
        count_score = min(1.0, float(core_count) / float(min_nodes))
        agree_score = min(1.0, float(cand_agree) / float(min_nodes))
        sim_pres = float(torch.dot(q_pres[qi], pres_core[qi]).item()) if core_count > 0 else 0.0
        sim_adapt = float(torch.dot(q_adapt[qi], adapt_core[qi]).item()) if core_count > 0 else 0.0
        sim_score = max(0.0, min(1.0, 0.5 * ((sim_pres + 1.0) * 0.5 + (sim_adapt + 1.0) * 0.5)))
        raw = (
            float(cfg.state_profile_core_reliability_count_weight) * count_score
            + float(cfg.state_profile_core_reliability_agreement_weight) * agree_score
            + float(cfg.state_profile_core_reliability_query_sim_weight) * sim_score
        ) / weights_sum
        if used_fallback:
            raw *= 0.85
        rel = max(float(cfg.state_profile_core_reliability_min), min(1.0, float(raw)))
        out.append(rel)
        rows.append({
            "state_profile_core_reliability": float(rel),
            "state_profile_core_reliability_count_score": float(count_score),
            "state_profile_core_reliability_agreement_score": float(agree_score),
            "state_profile_core_reliability_query_sim_score": float(sim_score),
            "state_profile_core_query_sim_preserved": float(sim_pres),
            "state_profile_core_query_sim_adapted": float(sim_adapt),
        })
    return torch.tensor(out, dtype=torch.float32), rows


def _state_profile_floor_package(profile_or_id: Mapping[str, object] | str | None, cfg: DualBranchGateConfig) -> dict[str, object]:
    """Return the floor/search package for a selected state profile.

    This makes the router choose a profile-level search package rather than
    only a global floor setting.  Ratios <= 0 mean that floor recovery is
    disabled for the corresponding profile.
    """
    profile_id = _profile_id_from_obj(profile_or_id)
    mapping = {
        "strong_recovery_profile": (float(cfg.state_profile_strong_floor_ratio), str(cfg.state_profile_strong_accept_mode), "strong_recovery_profile_floor"),
        "moderate_recall_profile": (float(cfg.state_profile_moderate_floor_ratio), str(cfg.state_profile_moderate_accept_mode), "moderate_recall_profile_floor"),
        "boundary_preserve_profile": (float(cfg.state_profile_boundary_floor_ratio), str(cfg.state_profile_boundary_accept_mode), "boundary_preserve_profile_floor"),
        "boundary_plateau_precision_profile": (float(cfg.state_profile_boundary_floor_ratio), str(cfg.state_profile_boundary_accept_mode), "boundary_plateau_precision_profile_floor"),
        "boundary_margin_sensitive_profile": (float(cfg.state_profile_boundary_floor_ratio), str(cfg.state_profile_boundary_accept_mode), "boundary_margin_sensitive_profile_floor"),
        "shift_adaptive_profile": (float(cfg.state_profile_shift_floor_ratio), str(cfg.state_profile_shift_accept_mode), "shift_adaptive_profile_floor"),
        "stable_default_profile": (float(cfg.state_profile_stable_floor_ratio), str(cfg.state_profile_stable_accept_mode), "stable_default_profile_floor"),
    }
    ratio, accept_mode, state = mapping.get(profile_id, (0.0, str(cfg.guarded_floor_accept_mode), "unknown_profile_floor"))
    return {
        "profile_id": profile_id,
        "enabled": bool(cfg.state_profile_floor_enabled and ratio > 0.0),
        "ratio": float(ratio),
        "accept_mode": str(accept_mode),
        "state": str(state),
        "reason": f"state_profile_{profile_id}_ratio_{float(ratio):.4f}",
    }


def _load_node_set_mean_embs(
    manifest_path: str | Path,
    node_sets: Sequence[Sequence[int]],
    *,
    max_nodes: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Load mean embeddings for arbitrary node sets from a shard manifest."""
    manifest_file, manifest = load_embedding_manifest(manifest_path)
    dim = int(manifest["embedding_dim"])
    node_to_queries: dict[int, list[int]] = {}
    clipped_sets: list[list[int]] = []
    for qi, nodes in enumerate(node_sets):
        seen: list[int] = []
        for node in nodes:
            nn = int(node)
            if nn in seen:
                continue
            seen.append(nn)
            if len(seen) >= int(max_nodes):
                break
        clipped_sets.append(seen)
        for node in seen:
            node_to_queries.setdefault(int(node), []).append(qi)
    sums = [torch.zeros(dim, dtype=torch.float32) for _ in node_sets]
    counts = [0 for _ in node_sets]
    for row in manifest.get("shards", []):
        start, end = int(row["start"]), int(row["end"])
        needed = [node for node in node_to_queries if start <= node < end]
        if not needed:
            continue
        emb = _load_shard_tensor(manifest_file.parent / str(row["path"]), start=start, end=end).float()
        emb = F.normalize(emb, p=2, dim=-1, eps=eps)
        for node in needed:
            vec = emb[int(node) - start].detach().cpu()
            for qi in node_to_queries[int(node)]:
                sums[qi] += vec
                counts[qi] += 1
    outs: list[torch.Tensor] = []
    for qi, total in enumerate(sums):
        if counts[qi] <= 0:
            outs.append(torch.zeros(dim, dtype=torch.float32))
        else:
            outs.append(total / float(counts[qi]))
    return F.normalize(torch.stack(outs, dim=0).float(), p=2, dim=-1, eps=eps)


def _edge_index_to_adjacency(edge_index: torch.Tensor | None, num_nodes: int) -> list[set[int]] | None:
    if edge_index is None:
        return None
    try:
        ei = edge_index.long().cpu()
        if ei.ndim != 2 or int(ei.shape[0]) != 2:
            return None
        adj = [set() for _ in range(int(num_nodes))]
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):
            uu, vv = int(u), int(v)
            if 0 <= uu < num_nodes and 0 <= vv < num_nodes and uu != vv:
                adj[uu].add(vv)
                adj[vv].add(uu)
        return adj
    except Exception:
        return None


def _ranked_structure_scores(
    *,
    ranked_nodes: Sequence[int],
    query_nodes: Sequence[int],
    adjacency: list[set[int]] | None,
    core_topk: int,
) -> list[float] | None:
    if adjacency is None:
        return None
    query_set = set(int(x) for x in query_nodes)
    core = set(query_set)
    for node in ranked_nodes[: max(0, int(core_topk))]:
        core.add(int(node))
    if not core:
        return None
    vals: list[float] = []
    for node in ranked_nodes:
        nn = int(node)
        if nn < 0 or nn >= len(adjacency):
            vals.append(0.0)
            continue
        neigh = adjacency[nn]
        if not neigh:
            vals.append(0.0)
            continue
        in_core = len(neigh & core)
        q_hits = len(neigh & query_set)
        vals.append(float(in_core + 0.5 * q_hits) / math.sqrt(float(len(neigh)) + 1.0))
    # Keep a strict one-to-one relation between ranked nodes and structure
    # scores.  This prevents structure-enabled profiles from silently changing
    # scale because of duplicated per-node scores.
    if len(vals) != len(ranked_nodes):
        raise V5ProtocolError("ranked structure score length mismatch")
    return vals


def _high_confidence_core_nodes(
    result: ChunkedScoreResult,
    qi: int,
    stats: BranchCandidateStats,
    *,
    max_nodes: int,
) -> tuple[int, ...]:
    """Return a label-free high-confidence core for a branch candidate.

    The core prototype should not be averaged over the entire searched
    community, because a searched candidate may contain boundary nodes.  We
    therefore take the highest-ranked nodes that are also in the branch
    candidate.  Query seed nodes are only used as a fallback to avoid an empty
    prototype.
    """
    limit = max(1, int(max_nodes))
    qset = set(int(x) for x in result.query_node_sets[int(qi)])
    cand = set(int(x) for x in stats.nodes)
    out: list[int] = []
    for node in result.top_nodes[int(qi)].tolist():
        nn = int(node)
        if nn in qset:
            continue
        if cand and nn not in cand:
            continue
        if nn not in out:
            out.append(nn)
        if len(out) >= limit:
            break
    if not out:
        for node in stats.nodes:
            nn = int(node)
            if nn in qset:
                continue
            if nn not in out:
                out.append(nn)
            if len(out) >= limit:
                break
    if not out:
        out = list(qset)[:limit]
    return tuple(int(x) for x in out[:limit])


def _agreement_core_details(
    preserved: ChunkedScoreResult,
    adapted: ChunkedScoreResult,
    qi: int,
    pres: BranchCandidateStats,
    ad: BranchCandidateStats,
    *,
    max_nodes: int,
    topk: int,
    strict_candidate_agreement: bool = False,
) -> dict[str, object]:
    """Build a label-free high-confidence agreement core with diagnostics.

    Candidate-agreement nodes are inside both branch search candidates.  Rank-
    agreement nodes are merely high-ranked by both branches.  V5.7.0 exposes
    both counts so later analysis can tell whether the core is truly supported
    by both searched communities or mostly by fallback ranking overlap.
    """
    limit = max(1, int(max_nodes))
    k = max(1, int(topk))
    qset = set(int(x) for x in preserved.query_node_sets[int(qi)])
    pres_rank = {int(node): rank for rank, node in enumerate(preserved.top_nodes[int(qi), :k].tolist())}
    adapt_rank = {int(node): rank for rank, node in enumerate(adapted.top_nodes[int(qi), :k].tolist())}
    pres_cand = set(int(x) for x in pres.nodes)
    adapt_cand = set(int(x) for x in ad.nodes)

    cand_common = (pres_cand & adapt_cand & set(pres_rank) & set(adapt_rank)) - qset
    cand_nodes = sorted(cand_common, key=lambda node: pres_rank.get(node, 10**9) + adapt_rank.get(node, 10**9))

    rank_nodes: list[int] = []
    if not bool(strict_candidate_agreement):
        rank_common = (set(pres_rank) & set(adapt_rank)) - qset - set(cand_nodes)
        rank_nodes = sorted(rank_common, key=lambda node: pres_rank.get(node, 10**9) + adapt_rank.get(node, 10**9))

    out: list[int] = []
    for node in cand_nodes:
        nn = int(node)
        if nn not in out:
            out.append(nn)
        if len(out) >= limit:
            break
    used_rank_fallback = False
    if len(out) < limit:
        for node in rank_nodes:
            nn = int(node)
            if nn not in out:
                out.append(nn)
                used_rank_fallback = True
            if len(out) >= limit:
                break

    return {
        "nodes": tuple(out[:limit]),
        "candidate_agreement_count": int(len(cand_nodes)),
        "rank_agreement_count": int(len(rank_nodes)),
        "used_rank_fallback": bool(used_rank_fallback),
        "strict_candidate_agreement": bool(strict_candidate_agreement),
    }


def _agreement_core_nodes(
    preserved: ChunkedScoreResult,
    adapted: ChunkedScoreResult,
    qi: int,
    pres: BranchCandidateStats,
    ad: BranchCandidateStats,
    *,
    max_nodes: int,
    topk: int,
) -> tuple[int, ...]:
    return tuple(
        _agreement_core_details(
            preserved,
            adapted,
            qi,
            pres,
            ad,
            max_nodes=max_nodes,
            topk=topk,
            strict_candidate_agreement=False,
        )["nodes"]
    )


def _merge_core_nodes(*groups: Sequence[int], limit: int) -> tuple[int, ...]:
    out: list[int] = []
    for group in groups:
        for node in group:
            nn = int(node)
            if nn not in out:
                out.append(nn)
            if len(out) >= int(limit):
                return tuple(out)
    return tuple(out)

def _choose_recalibrated_prefix(
    *,
    q_scores: list[float],
    ranked_nodes: list[int],
    ranked_scores: list[float],
    full_score_sum: float,
    num_nodes: int,
    alpha: float,
    preserved_size: int,
    adapted_size: int,
    entropy: float,
    cfg: DualBranchGateConfig,
    target_floor_enabled: bool = True,
    target_underfloor_rate: float = 0.0,
    effective_floor_ratio: float | None = None,
    effective_accept_mode: str | None = None,
    floor_route_state: str = "single",
    floor_route_reason: str = "single_guarded_floor",
    target_moderate_underfloor_rate: float = 0.0,
    target_strong_underfloor_rate: float = 0.0,
    target_probe_rel_quality_drop: float = 0.0,
    target_probe_rel_density_drop: float = 0.0,
    target_branch_jaccard_mean: float = 0.0,
    target_overexpansion_rate: float = 0.0,
    target_overexpansion_action: bool = False,
    target_overexpansion_reason: str = "disabled",
    target_moderate_max_branch_jaccard: float = 1.0,
    target_moderate_quality_drop_threshold: float = 1.0,
    target_moderate_density_drop_threshold: float = 1.0,
    floor_router_disagreement_pass: bool = True,
    floor_router_quality_drop_pass: bool = True,
    floor_router_density_drop_pass: bool = True,
    state_profile_id: str = "disabled",
    state_profile_reason: str = "state_profile_disabled",
    state_profile_alpha_bias: float = 0.0,
    state_profile_core_weight: float = 0.0,
    state_profile_structure_weight: float = 0.0,
    state_profile_search_mode: str = "default",
    ranked_structure_scores: Sequence[float] | None = None,
) -> PrefixSearchDecision:
    endpoint = min(len(ranked_scores), min(int(0.50 * int(num_nodes)), 10_000))
    max_branch_size = max(int(len(q_scores)), int(preserved_size), int(adapted_size))
    eff_ratio = float(cfg.guarded_floor_ratio if effective_floor_ratio is None else effective_floor_ratio)
    eff_accept_mode = str(cfg.guarded_floor_accept_mode if effective_accept_mode is None else effective_accept_mode)
    disabled = PrefixSearchDecision(
        best_prefix_k=-1,
        final_prefix_k=-1,
        best_size=int(len(q_scores)),
        final_size=int(len(q_scores)),
        best_density=float("nan"),
        final_density=float("nan"),
        best_quality=float("nan"),
        final_quality=float("nan"),
        floor_prefix_k=-1,
        floor_size=int(math.ceil(float(eff_ratio) * float(max_branch_size))),
        floor_density=float("nan"),
        floor_quality=float("nan"),
        density_drop=float("nan"),
        quality_drop=float("nan"),
        rel_density_drop=float("nan"),
        rel_quality_drop=float("nan"),
        floor_accept_mode=str(eff_accept_mode),
        floor_accept_reason="",
        floor_reject_reason="disabled_senior_or_empty_endpoint",
        query_floor_enabled=False,
        max_branch_size=int(max_branch_size),
        floor_ratio=float(eff_ratio),
        floor_triggered=False,
        floor_accepted=False,
        floor_reason="disabled_senior_or_empty_endpoint",
        target_underfloor_rate=float(target_underfloor_rate),
        target_floor_enabled=bool(target_floor_enabled),
        floor_route_state=str(floor_route_state),
        floor_route_reason=str(floor_route_reason),
        effective_floor_ratio=float(eff_ratio),
        effective_accept_mode=str(eff_accept_mode),
        target_moderate_underfloor_rate=float(target_moderate_underfloor_rate),
        target_strong_underfloor_rate=float(target_strong_underfloor_rate),
        target_probe_rel_quality_drop=float(target_probe_rel_quality_drop),
        target_probe_rel_density_drop=float(target_probe_rel_density_drop),
        target_branch_jaccard_mean=float(target_branch_jaccard_mean),
        target_moderate_max_branch_jaccard=float(target_moderate_max_branch_jaccard),
        target_moderate_quality_drop_threshold=float(target_moderate_quality_drop_threshold),
        target_moderate_density_drop_threshold=float(target_moderate_density_drop_threshold),
        floor_router_disagreement_pass=bool(floor_router_disagreement_pass),
        floor_router_quality_drop_pass=bool(floor_router_quality_drop_pass),
        floor_router_density_drop_pass=bool(floor_router_density_drop_pass),
        state_profile_id=str(state_profile_id),
        state_profile_reason=str(state_profile_reason),
        state_profile_alpha_bias=float(state_profile_alpha_bias),
        state_profile_core_weight=float(state_profile_core_weight),
        state_profile_structure_weight=float(state_profile_structure_weight),
        state_profile_search_mode=str(state_profile_search_mode),
        state_profile_query_proto_enabled=bool(float(state_profile_core_weight) > 0.0),
        state_profile_structure_enabled=bool(ranked_structure_scores is not None and float(state_profile_structure_weight) > 0.0),
        search_endpoint=int(endpoint),
    )
    if endpoint <= 0 or cfg.search_profile == "senior":
        return disabled
    q_sum = float(sum(q_scores))
    prefix = [0.0]
    acc = 0.0
    for s in ranked_scores[:endpoint]:
        acc += float(s)
        prefix.append(acc)
    struct_prefix = [0.0]
    if ranked_structure_scores is not None and float(state_profile_structure_weight) > 0.0:
        sacc = 0.0
        for val in list(ranked_structure_scores)[:endpoint]:
            sacc += float(val)
            struct_prefix.append(sacc)
        while len(struct_prefix) < endpoint + 1:
            struct_prefix.append(struct_prefix[-1])
    else:
        struct_prefix = [0.0] * (endpoint + 1)
    ref_size = max(float(cfg.min_size_reference), float(num_nodes) ** float(cfg.size_reference_power))
    consensus = max(1.0, 0.5 * float(preserved_size) + 0.5 * float(adapted_size))
    lo = float(cfg.balanced_size_low) * consensus
    hi = float(cfg.balanced_size_high) * consensus
    precision_over_coefficient = float(cfg.precision_recall_over_penalty)
    if bool(target_overexpansion_action):
        precision_over_coefficient += float(cfg.overexpansion_precision_extra_penalty)
    if str(state_profile_id) == "boundary_plateau_precision_profile":
        precision_over_coefficient += float(cfg.boundary_plateau_precision_extra_penalty)

    def precision_penalty_for_size(size: int) -> float:
        if cfg.search_profile != "precision_recall":
            return 0.0
        over_ref = max(0.0, math.log((float(size) + cfg.eps) / (ref_size + cfg.eps)))
        return float(alpha) * float(precision_over_coefficient) * float(over_ref)

    def eval_prefix(k: int) -> tuple[int, float, float]:
        kk = max(0, min(int(k), int(endpoint)))
        size = max(1, len(q_scores) + kk)
        density = subgraph_density_controlled(q_sum + prefix[kk], size, full_score_sum=full_score_sum, num_nodes=num_nodes)
        under_cons = max(0.0, math.log((lo + cfg.eps) / (float(size) + cfg.eps))) if size < lo else 0.0
        over_cons = max(0.0, math.log((float(size) + cfg.eps) / (hi + cfg.eps))) if size > hi else 0.0
        over_ref = max(0.0, math.log((float(size) + cfg.eps) / (ref_size + cfg.eps)))
        quality = float(density)
        if cfg.search_profile == "balanced_size":
            quality = density - 0.06 * under_cons - 0.06 * over_cons + (0.02 if lo <= size <= hi else 0.0)
        elif cfg.search_profile == "entropy_aware":
            recall_bonus = float(cfg.high_entropy_recall_bonus) * math.log1p(kk) if float(entropy) >= float(cfg.entropy_threshold) else 0.0
            quality = density + recall_bonus - 0.05 * over_ref
        elif cfg.search_profile == "precision_recall":
            # alpha close to adapted -> more precision; alpha close to preserved -> recall protection.
            quality = (
                density
                + (1.0 - float(alpha)) * 0.04 * math.log1p(kk)
                - precision_penalty_for_size(size)
                - 0.05 * under_cons
            )
        elif cfg.search_profile == "target_state_recovery":
            recall_bonus = float(cfg.high_entropy_recall_bonus) * math.log1p(kk) if float(entropy) >= float(cfg.entropy_threshold) else 0.0
            consensus_bonus = float(cfg.search_consensus_bonus) if lo <= size <= hi else 0.0
            quality = (
                density
                + (1.0 - float(alpha)) * recall_bonus
                + consensus_bonus
                - float(cfg.recall_under_penalty) * under_cons
                - float(cfg.precision_over_penalty) * float(alpha) * over_ref
                - 0.04 * over_cons
            )
        if bool(cfg.state_profile_search_enabled):
            mode = str(state_profile_search_mode)
            if mode == "moderate_recall":
                quality += float(cfg.state_profile_moderate_recall_bonus) * math.log1p(kk)
                quality -= 0.02 * over_ref
            elif mode == "boundary_preserve":
                quality -= float(cfg.state_profile_boundary_quality_penalty) * (under_cons + over_cons + 0.5 * over_ref)
            elif mode == "shift_adaptive":
                quality += float(cfg.state_profile_shift_adapt_bonus) * max(0.0, float(alpha))
            elif mode == "strong_recovery":
                quality += 0.01 * math.log1p(kk)
        if ranked_structure_scores is not None and float(state_profile_structure_weight) > 0.0:
            struct_mean = float(struct_prefix[kk]) / float(max(1, kk)) if kk > 0 else 0.0
            quality += float(state_profile_structure_weight) * struct_mean
        return int(size), float(density), float(quality)

    best_k = 0
    best_size, best_density, best_quality = eval_prefix(0)
    for k in range(1, endpoint + 1):
        size, density, quality = eval_prefix(k)
        if quality > best_quality:
            best_k = int(k)
            best_size, best_density, best_quality = int(size), float(density), float(quality)

    floor_target_size = max(len(q_scores), int(math.ceil(float(eff_ratio) * float(max_branch_size))))
    floor_k = max(0, min(int(endpoint), int(floor_target_size) - int(len(q_scores))))
    floor_size, floor_density, floor_quality = eval_prefix(floor_k)
    density_drop = float(best_density - floor_density)
    quality_drop = float(best_quality - floor_quality)
    rel_density_drop = float(density_drop) / max(abs(float(best_density)), float(cfg.eps))
    rel_quality_drop = float(quality_drop) / max(abs(float(best_quality)), float(cfg.eps))
    final_k = int(best_k)
    final_size = int(best_size)
    final_density = float(best_density)
    final_quality = float(best_quality)
    floor_triggered = False
    floor_accepted = False
    floor_reason = "not_triggered"
    floor_accept_reason = ""
    floor_reject_reason = ""
    query_floor_enabled = False

    if bool(cfg.guarded_floor_enabled) and bool(target_floor_enabled):
        floor_threshold_size = float(eff_ratio) * float(max_branch_size)
        query_floor_enabled = bool(float(best_size) + cfg.eps < floor_threshold_size)
        if query_floor_enabled:
            floor_triggered = True
            density_ok = float(density_drop) <= float(cfg.guarded_floor_density_tolerance)
            quality_ok = float(quality_drop) <= float(cfg.guarded_floor_quality_tolerance)
            rel_density_ok = float(rel_density_drop) <= float(cfg.guarded_floor_rel_density_tolerance)
            rel_quality_ok = float(rel_quality_drop) <= float(cfg.guarded_floor_rel_quality_tolerance)
            mode = str(eff_accept_mode)
            metric = str(cfg.guarded_floor_guard_metric)
            if mode == "force":
                accept = True
                floor_accept_reason = "force"
            elif mode == "relative_quality":
                accept = bool(rel_quality_ok)
                floor_accept_reason = "relative_quality_ok" if accept else ""
                floor_reject_reason = "relative_quality_drop_exceeds_tolerance" if not accept else ""
            elif mode == "relative_quality_density":
                accept = bool(rel_quality_ok and rel_density_ok)
                floor_accept_reason = "relative_quality_density_ok" if accept else ""
                if not accept:
                    reasons = []
                    if not rel_quality_ok:
                        reasons.append("relative_quality_drop_exceeds_tolerance")
                    if not rel_density_ok:
                        reasons.append("relative_density_drop_exceeds_tolerance")
                    floor_reject_reason = "+".join(reasons)
            else:  # strict_abs legacy guard
                if metric == "density":
                    accept = density_ok
                elif metric == "quality":
                    accept = quality_ok
                elif metric == "either":
                    accept = density_ok or quality_ok
                else:  # both
                    accept = density_ok and quality_ok
                floor_accept_reason = f"strict_abs_{metric}_ok" if accept else ""
                floor_reject_reason = f"strict_abs_{metric}_drop_exceeds_tolerance" if not accept else ""
            if accept and int(floor_k) > int(best_k):
                floor_accepted = True
                final_k = int(floor_k)
                final_size = int(floor_size)
                final_density = float(floor_density)
                final_quality = float(floor_quality)
                floor_reason = f"accepted_{mode}"
            else:
                if accept and int(floor_k) <= int(best_k):
                    floor_reject_reason = "floor_prefix_not_larger_than_best"
                floor_reason = f"rejected_{mode}"
        else:
            floor_reason = "best_size_above_floor"
            floor_reject_reason = "best_size_above_floor"
    elif bool(cfg.guarded_floor_enabled) and not bool(target_floor_enabled):
        floor_reason = "target_trigger_disabled"
        floor_reject_reason = "target_trigger_disabled"

    # CoCS-like precision recovery: inspect a shorter endpoint only after the
    # original decision is complete.  This is the exact opposite of guarded
    # floor recovery and is active only for a label-free target-level
    # over-expansion state.  No target name or target metric is consulted.
    final_size_before_ceiling = int(final_size)
    ceiling_prefix_k = int(final_k)
    ceiling_size = int(final_size)
    ceiling_density = float(final_density)
    ceiling_quality = float(final_quality)
    ceiling_rel_density_drop = 0.0
    ceiling_rel_quality_drop = 0.0
    ceiling_triggered = False
    ceiling_accepted = False
    ceiling_reason = "disabled"
    query_overexpanded = bool(
        float(final_size) > float(cfg.overexpansion_ratio_threshold) * float(max_branch_size) + cfg.eps
    )
    plateau_ceiling_active = bool(
        cfg.boundary_plateau_ceiling_enabled
        and str(state_profile_id) == "boundary_plateau_precision_profile"
    )
    if plateau_ceiling_active:
        ceiling_target_size = max(
            int(preserved_size),
            int(math.ceil(float(final_size) * float(cfg.boundary_plateau_ceiling_ratio))),
        )
        ceiling_prefix_k = max(
            0,
            min(int(endpoint), int(ceiling_target_size) - int(len(q_scores))),
        )
        if int(ceiling_prefix_k) < int(final_k):
            ceiling_triggered = True
            ceiling_size, ceiling_density, ceiling_quality = eval_prefix(ceiling_prefix_k)
            ceiling_rel_density_drop = float(final_density - ceiling_density) / max(
                abs(float(final_density)), float(cfg.eps)
            )
            ceiling_rel_quality_drop = float(final_quality - ceiling_quality) / max(
                abs(float(final_quality)), float(cfg.eps)
            )
            density_ok = bool(
                ceiling_rel_density_drop
                <= float(cfg.boundary_plateau_ceiling_rel_density_tolerance) + cfg.eps
            )
            quality_ok = bool(
                ceiling_rel_quality_drop
                <= float(cfg.boundary_plateau_ceiling_rel_quality_tolerance) + cfg.eps
            )
            if density_ok and quality_ok:
                ceiling_accepted = True
                final_k = int(ceiling_prefix_k)
                final_size = int(ceiling_size)
                final_density = float(ceiling_density)
                final_quality = float(ceiling_quality)
                ceiling_reason = "accepted_boundary_plateau_relative_quality_density"
            else:
                reasons = []
                if not quality_ok:
                    reasons.append("relative_quality_drop_exceeds_tolerance")
                if not density_ok:
                    reasons.append("relative_density_drop_exceeds_tolerance")
                ceiling_reason = "rejected_boundary_plateau_" + "+".join(reasons)
        else:
            ceiling_reason = "boundary_plateau_candidate_not_shorter_than_best"
    elif bool(cfg.guarded_ceiling_enabled) and bool(target_overexpansion_action):
        if query_overexpanded:
            ceiling_triggered = True
            ceiling_target_size = max(
                int(max_branch_size),
                int(math.ceil(float(final_size) * float(cfg.guarded_ceiling_ratio))),
            )
            ceiling_prefix_k = max(
                0,
                min(int(endpoint), int(ceiling_target_size) - int(len(q_scores))),
            )
            ceiling_size, ceiling_density, ceiling_quality = eval_prefix(ceiling_prefix_k)
            ceiling_rel_density_drop = float(final_density - ceiling_density) / max(
                abs(float(final_density)), float(cfg.eps)
            )
            ceiling_rel_quality_drop = float(final_quality - ceiling_quality) / max(
                abs(float(final_quality)), float(cfg.eps)
            )
            density_ok = bool(
                ceiling_rel_density_drop <= float(cfg.guarded_ceiling_rel_density_tolerance) + cfg.eps
            )
            quality_ok = bool(
                ceiling_rel_quality_drop <= float(cfg.guarded_ceiling_rel_quality_tolerance) + cfg.eps
            )
            if int(ceiling_prefix_k) < int(final_k) and density_ok and quality_ok:
                ceiling_accepted = True
                final_k = int(ceiling_prefix_k)
                final_size = int(ceiling_size)
                final_density = float(ceiling_density)
                final_quality = float(ceiling_quality)
                ceiling_reason = "accepted_relative_quality_density"
            elif int(ceiling_prefix_k) >= int(final_k):
                ceiling_reason = "rejected_candidate_not_shorter"
            else:
                reasons = []
                if not quality_ok:
                    reasons.append("relative_quality_drop_exceeds_tolerance")
                if not density_ok:
                    reasons.append("relative_density_drop_exceeds_tolerance")
                ceiling_reason = "rejected_" + "+".join(reasons)
        else:
            ceiling_reason = "query_not_overexpanded"
    elif bool(cfg.guarded_ceiling_enabled):
        ceiling_reason = "target_overexpansion_action_disabled"
    elif bool(cfg.boundary_plateau_ceiling_enabled):
        ceiling_reason = "boundary_plateau_state_not_active"

    prev_k = max(0, int(final_k) - 1)
    next_k = min(int(endpoint), int(final_k) + 1)
    _, previous_prefix_density, previous_prefix_quality = eval_prefix(prev_k)
    _, next_prefix_density, next_prefix_quality = eval_prefix(next_k)
    marginal_density_from_previous = float(final_density - previous_prefix_density)
    marginal_quality_from_previous = float(final_quality - previous_prefix_quality)
    marginal_density_to_next = float(next_prefix_density - final_density)
    marginal_quality_to_next = float(next_prefix_quality - final_quality)
    precision_size_penalty = precision_penalty_for_size(int(final_size))

    return PrefixSearchDecision(
        best_prefix_k=int(best_k),
        final_prefix_k=int(final_k),
        best_size=int(best_size),
        final_size=int(final_size),
        best_density=float(best_density),
        final_density=float(final_density),
        best_quality=float(best_quality),
        final_quality=float(final_quality),
        floor_prefix_k=int(floor_k),
        floor_size=int(floor_size),
        floor_density=float(floor_density),
        floor_quality=float(floor_quality),
        density_drop=float(density_drop),
        quality_drop=float(quality_drop),
        rel_density_drop=float(rel_density_drop),
        rel_quality_drop=float(rel_quality_drop),
        floor_accept_mode=str(eff_accept_mode),
        floor_accept_reason=str(floor_accept_reason),
        floor_reject_reason=str(floor_reject_reason),
        query_floor_enabled=bool(query_floor_enabled),
        max_branch_size=int(max_branch_size),
        floor_ratio=float(eff_ratio),
        floor_triggered=bool(floor_triggered),
        floor_accepted=bool(floor_accepted),
        floor_reason=str(floor_reason),
        target_underfloor_rate=float(target_underfloor_rate),
        target_floor_enabled=bool(target_floor_enabled),
        floor_route_state=str(floor_route_state),
        floor_route_reason=str(floor_route_reason),
        effective_floor_ratio=float(eff_ratio),
        effective_accept_mode=str(eff_accept_mode),
        target_moderate_underfloor_rate=float(target_moderate_underfloor_rate),
        target_strong_underfloor_rate=float(target_strong_underfloor_rate),
        target_probe_rel_quality_drop=float(target_probe_rel_quality_drop),
        target_probe_rel_density_drop=float(target_probe_rel_density_drop),
        target_branch_jaccard_mean=float(target_branch_jaccard_mean),
        target_moderate_max_branch_jaccard=float(target_moderate_max_branch_jaccard),
        target_moderate_quality_drop_threshold=float(target_moderate_quality_drop_threshold),
        target_moderate_density_drop_threshold=float(target_moderate_density_drop_threshold),
        floor_router_disagreement_pass=bool(floor_router_disagreement_pass),
        floor_router_quality_drop_pass=bool(floor_router_quality_drop_pass),
        floor_router_density_drop_pass=bool(floor_router_density_drop_pass),
        state_profile_id=str(state_profile_id),
        state_profile_reason=str(state_profile_reason),
        state_profile_alpha_bias=float(state_profile_alpha_bias),
        state_profile_core_weight=float(state_profile_core_weight),
        state_profile_structure_weight=float(state_profile_structure_weight),
        state_profile_search_mode=str(state_profile_search_mode),
        state_profile_query_proto_enabled=bool(float(state_profile_core_weight) > 0.0),
        state_profile_structure_enabled=bool(ranked_structure_scores is not None and float(state_profile_structure_weight) > 0.0),
        search_endpoint=int(endpoint),
        precision_over_coefficient=float(precision_over_coefficient),
        precision_size_penalty=float(precision_size_penalty),
        target_overexpansion_rate=float(target_overexpansion_rate),
        target_overexpansion_action=bool(target_overexpansion_action),
        target_overexpansion_reason=str(target_overexpansion_reason),
        ceiling_prefix_k=int(ceiling_prefix_k),
        ceiling_size=int(ceiling_size),
        ceiling_density=float(ceiling_density),
        ceiling_quality=float(ceiling_quality),
        ceiling_rel_density_drop=float(ceiling_rel_density_drop),
        ceiling_rel_quality_drop=float(ceiling_rel_quality_drop),
        ceiling_triggered=bool(ceiling_triggered),
        ceiling_accepted=bool(ceiling_accepted),
        ceiling_reason=str(ceiling_reason),
        final_size_before_ceiling=int(final_size_before_ceiling),
        final_size_after_ceiling=int(final_size),
        previous_prefix_density=float(previous_prefix_density),
        previous_prefix_quality=float(previous_prefix_quality),
        next_prefix_density=float(next_prefix_density),
        next_prefix_quality=float(next_prefix_quality),
        marginal_density_from_previous=float(marginal_density_from_previous),
        marginal_quality_from_previous=float(marginal_quality_from_previous),
        marginal_density_to_next=float(marginal_density_to_next),
        marginal_quality_to_next=float(marginal_quality_to_next),
    )


def _boundary_child_profile_from_normalized_marginal(
    parent_profile: Mapping[str, object],
    *,
    target_normalized_endpoint_marginal: float,
    cfg: DualBranchGateConfig,
) -> dict[str, object]:
    """Return the second-stage boundary child for a normalized endpoint signal."""
    profile = dict(parent_profile)
    parent_id = str(profile.get("profile_id", "disabled"))
    if not bool(cfg.hierarchical_boundary_state_enabled) or parent_id != "boundary_preserve_profile":
        return profile
    threshold = float(cfg.boundary_plateau_relative_quality_threshold)
    is_plateau = bool(float(target_normalized_endpoint_marginal) <= threshold + float(cfg.eps))
    profile["parent_profile_id"] = parent_id
    profile["profile_id"] = (
        "boundary_plateau_precision_profile"
        if is_plateau
        else "boundary_margin_sensitive_profile"
    )
    profile["reason"] = (
        f"hierarchical_boundary_target_normalized_endpoint_marginal_"
        f"{float(target_normalized_endpoint_marginal):.12g}_"
        f"{'le' if is_plateau else 'gt'}_{threshold:.12g}"
    )
    profile["search_mode"] = "boundary_preserve"
    return profile


def _refine_boundary_retrieval_state(
    *,
    safe_score_result: ChunkedScoreResult,
    gate_rows: Sequence[Mapping[str, object]],
    parent_profile: Mapping[str, object],
    cfg: DualBranchGateConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """Refine only the first-stage boundary fallback using endpoint saturation.

    The probe is computed from the safe/default B1 ranking before any boundary
    precision action is applied.  It is normalized by the target's own search
    quality scale, so the decision is independent of graph names and target
    ground truth.  Other first-stage states are returned unchanged.
    """
    profile = dict(parent_profile)
    parent_id = str(profile.get("profile_id", "disabled"))
    disabled = {
        "enabled": bool(cfg.hierarchical_boundary_state_enabled),
        "eligible": False,
        "parent_profile_id": parent_id,
        "refined_profile_id": parent_id,
        "target_normalized_endpoint_marginal": float("nan"),
        "threshold": float(cfg.boundary_plateau_relative_quality_threshold),
        "threshold_margin": float("nan"),
        "query_plateau_rate": 0.0,
        "query_rows": [],
        "labels_used": False,
        "ground_truth_used": False,
    }
    if not bool(cfg.hierarchical_boundary_state_enabled) or parent_id != "boundary_preserve_profile":
        return profile, disabled

    rows_by_query = {int(row.get("query_index", -1)): row for row in gate_rows}
    floor_pkg = _state_profile_floor_package(parent_id, cfg)
    decisions: list[PrefixSearchDecision] = []
    query_rows: list[dict[str, object]] = []
    for qi, qset in enumerate(safe_score_result.query_node_sets):
        gate = rows_by_query.get(int(qi), {})
        ranked_nodes = [int(x) for x in safe_score_result.top_nodes[qi].tolist()]
        ranked_scores = [float(x) for x in safe_score_result.top_scores[qi].tolist()]
        q_scores = [float(safe_score_result.query_seed_scores[qi][int(seed)]) for seed in qset]
        decision = _choose_recalibrated_prefix(
            q_scores=q_scores,
            ranked_nodes=ranked_nodes,
            ranked_scores=ranked_scores,
            full_score_sum=float(safe_score_result.full_score_sums[qi].item()),
            num_nodes=int(safe_score_result.num_nodes),
            alpha=float(gate.get("alpha", cfg.fixed_alpha)),
            preserved_size=int(gate.get("preserved_size", len(qset))),
            adapted_size=int(gate.get("adapted_size", len(qset))),
            entropy=float(max(float(gate.get("preserved_entropy", 0.0)), float(gate.get("adapted_entropy", 0.0)))),
            cfg=cfg,
            target_floor_enabled=bool(floor_pkg.get("enabled", False)),
            effective_floor_ratio=float(floor_pkg.get("ratio", cfg.guarded_floor_ratio)),
            effective_accept_mode=str(floor_pkg.get("accept_mode", cfg.guarded_floor_accept_mode)),
            floor_route_state=str(floor_pkg.get("state", "boundary_preserve_profile_floor")),
            floor_route_reason=str(floor_pkg.get("reason", "safe_boundary_probe")),
            state_profile_id=parent_id,
            state_profile_reason=str(profile.get("reason", "")),
            state_profile_alpha_bias=float(profile.get("alpha_bias", 0.0)),
            state_profile_core_weight=float(profile.get("core_weight", 0.0)),
            state_profile_structure_weight=0.0,
            state_profile_search_mode=str(profile.get("search_mode", "boundary_preserve")),
        )
        decisions.append(decision)
        normalized = (
            abs(float(decision.marginal_quality_from_previous))
            + abs(float(decision.marginal_quality_to_next))
        ) / max(abs(float(decision.final_quality)), float(cfg.eps))
        query_rows.append({
            "query_index": int(qi),
            "safe_prefix_k": int(decision.final_prefix_k),
            "safe_final_size": int(decision.final_size),
            "safe_final_density": float(decision.final_density),
            "safe_final_quality": float(decision.final_quality),
            "safe_marginal_quality_from_previous": float(decision.marginal_quality_from_previous),
            "safe_marginal_quality_to_next": float(decision.marginal_quality_to_next),
            "safe_normalized_endpoint_marginal": float(normalized),
        })

    mean_prev = _mean_stat(
        [{"value": d.marginal_quality_from_previous} for d in decisions], "value", 0.0
    )
    mean_next = _mean_stat(
        [{"value": d.marginal_quality_to_next} for d in decisions], "value", 0.0
    )
    mean_quality = _mean_stat(
        [{"value": d.final_quality} for d in decisions], "value", 0.0
    )
    target_normalized = (abs(mean_prev) + abs(mean_next)) / max(abs(mean_quality), float(cfg.eps))
    threshold = float(cfg.boundary_plateau_relative_quality_threshold)
    profile = _boundary_child_profile_from_normalized_marginal(
        profile,
        target_normalized_endpoint_marginal=target_normalized,
        cfg=cfg,
    )
    refined_id = str(profile.get("profile_id", parent_id))
    query_plateau_rate = float(
        sum(float(row["safe_normalized_endpoint_marginal"]) <= threshold + float(cfg.eps) for row in query_rows)
    ) / float(max(1, len(query_rows)))
    diagnostics = {
        "enabled": True,
        "eligible": True,
        "parent_profile_id": parent_id,
        "refined_profile_id": refined_id,
        "target_normalized_endpoint_marginal": float(target_normalized),
        "target_mean_marginal_quality_from_previous": float(mean_prev),
        "target_mean_marginal_quality_to_next": float(mean_next),
        "target_mean_final_quality": float(mean_quality),
        "threshold": threshold,
        "threshold_margin": float(threshold - target_normalized),
        "query_plateau_rate": float(query_plateau_rate),
        "query_rows": query_rows,
        "labels_used": False,
        "ground_truth_used": False,
    }
    return profile, diagnostics


def chunked_dual_branch_query_scores(
    *,
    preserved_manifest_path: str | Path,
    adapted_manifest_path: str | Path,
    query_node_sets: SeniorQueryBatch,
    chunk_size: int = 4096,
    top_k_pool: int | None = None,
    runtime_jsonl: str | Path | None = None,
    config: DualBranchGateConfig | None = None,
) -> tuple[ChunkedScoreResult, dict[str, object]]:
    cfg = (config or DualBranchGateConfig(enabled=True)).validate()
    pres_manifest_file, pres_manifest = load_embedding_manifest(preserved_manifest_path)
    adapt_manifest_file, adapt_manifest = load_embedding_manifest(adapted_manifest_path)
    if int(pres_manifest["num_nodes"]) != int(adapt_manifest["num_nodes"]):
        raise V5ProtocolError("dual branch manifests have num_nodes mismatch")
    if int(pres_manifest["embedding_dim"]) != int(adapt_manifest["embedding_dim"]):
        raise V5ProtocolError("dual branch manifests have embedding_dim mismatch")
    n = int(pres_manifest["num_nodes"])
    h = int(pres_manifest["embedding_dim"])
    qsets = _normalize_query_sets(query_node_sets, num_nodes=n)
    q_count = len(qsets)
    max_q = max(len(q) for q in qsets)
    k_pool = default_top_k_pool(n, max_q) if top_k_pool is None else min(int(top_k_pool), n)

    q_pres = _load_query_embs(preserved_manifest_path, qsets)
    q_adapt = _load_query_embs(adapted_manifest_path, qsets)
    q_pres_original = q_pres.clone()
    q_adapt_original = q_adapt.clone()
    mean_pres, std_pres = _branch_stats(manifest_path=preserved_manifest_path, query_embs=q_pres, chunk_size=chunk_size)
    mean_adapt, std_adapt = _branch_stats(manifest_path=adapted_manifest_path, query_embs=q_adapt, chunk_size=chunk_size)

    pres_result = _score_result_from_branch(
        manifest_path=preserved_manifest_path,
        query_node_sets=query_node_sets,
        chunk_size=chunk_size,
        top_k_pool=k_pool,
        runtime_jsonl=runtime_jsonl,
        score_calibration="zscore",
    )
    adapt_result = _score_result_from_branch(
        manifest_path=adapted_manifest_path,
        query_node_sets=query_node_sets,
        chunk_size=chunk_size,
        top_k_pool=k_pool,
        runtime_jsonl=runtime_jsonl,
        score_calibration="zscore",
    )
    alphas: list[float] = []
    gate_rows: list[dict[str, object]] = []
    preserved_core_node_sets: list[tuple[int, ...]] = []
    adapted_core_node_sets: list[tuple[int, ...]] = []
    agreement_core_node_sets: list[tuple[int, ...]] = []
    agreement_core_detail_sets: list[dict[str, object]] = []
    for qi in range(q_count):
        ps = _candidate_stats(pres_result, qi)
        ad = _candidate_stats(adapt_result, qi)
        preserved_core_node_sets.append(
            _high_confidence_core_nodes(pres_result, qi, ps, max_nodes=int(cfg.state_profile_core_max_nodes))
        )
        adapted_core_node_sets.append(
            _high_confidence_core_nodes(adapt_result, qi, ad, max_nodes=int(cfg.state_profile_core_max_nodes))
        )
        agreement_detail = _agreement_core_details(
            pres_result,
            adapt_result,
            qi,
            ps,
            ad,
            max_nodes=int(cfg.state_profile_core_max_nodes),
            topk=int(cfg.state_profile_core_agreement_topk),
            strict_candidate_agreement=bool(cfg.state_profile_core_strict_agreement),
        )
        agreement_core_detail_sets.append(agreement_detail)
        agreement_core_node_sets.append(tuple(int(x) for x in agreement_detail.get("nodes", ())))
        if cfg.alpha_mode == "fixed":
            alpha = float(cfg.fixed_alpha)
            comps = {"raw_alpha": alpha}
        else:
            alpha, comps = _adaptive_alpha(preserved=ps, adapted=ad, num_nodes=n, cfg=cfg)
        alphas.append(alpha)
        gate_rows.append({
            "query_index": int(qi),
            "alpha": float(alpha),
            "preserved_size": int(ps.size),
            "adapted_size": int(ad.size),
            "preserved_density": float(ps.density),
            "adapted_density": float(ad.density),
            "preserved_entropy": float(ps.entropy),
            "adapted_entropy": float(ad.entropy),
            "preserved_boundary_gap": float(ps.boundary_gap),
            "adapted_boundary_gap": float(ad.boundary_gap),
            "preserved_adapted_jaccard": float(_jaccard(ps.nodes, ad.nodes)),
            **{k: float(v) for k, v in comps.items()},
        })
    state_profile = _select_state_profile_from_branch_rows(gate_rows, cfg)

    # Diagnostic only: inference still uses the target-population profile.
    query_counterfactual_profiles = [
        _select_state_profile_from_branch_rows([row], cfg) for row in gate_rows
    ]
    query_profile_counts: dict[str, int] = {}
    for query_profile in query_counterfactual_profiles:
        profile_id = str(query_profile.get("profile_id", "disabled"))
        query_profile_counts[profile_id] = int(query_profile_counts.get(profile_id, 0)) + 1
    target_profile_id = str(state_profile.get("profile_id", "disabled"))
    query_profile_agreement_rate = float(
        sum(
            1
            for query_profile in query_counterfactual_profiles
            if str(query_profile.get("profile_id", "disabled")) == target_profile_id
        )
    ) / float(max(1, len(query_counterfactual_profiles)))
    query_profile_entropy = 0.0
    for count in query_profile_counts.values():
        probability = float(count) / float(max(1, len(query_counterfactual_profiles)))
        if probability > 0.0:
            query_profile_entropy -= probability * math.log(probability + 1e-12)
    query_profile_entropy_normalized = (
        query_profile_entropy / math.log(max(2, len(query_profile_counts)))
        if len(query_profile_counts) > 1
        else 0.0
    )
    route_threshold_margins = {
        "strong_size_margin": float(state_profile.get("target_profile_size_ratio", 0.0)) - float(cfg.state_profile_strong_size_ratio_min),
        "strong_jaccard_margin": float(cfg.state_profile_strong_branch_jaccard_max) - float(state_profile.get("target_profile_branch_jaccard", 0.0)),
        "moderate_size_margin": float(state_profile.get("target_profile_size_ratio", 0.0)) - float(cfg.state_profile_moderate_size_ratio_min),
        "moderate_jaccard_margin": float(cfg.state_profile_moderate_branch_jaccard_max) - float(state_profile.get("target_profile_branch_jaccard", 0.0)),
        "moderate_over_rate_margin": float(state_profile.get("target_profile_adapted_over_rate", 0.0)) - float(cfg.state_profile_moderate_over_rate_min),
        "moderate_density_margin": float(state_profile.get("target_profile_density_delta", 0.0)) - float(cfg.state_profile_moderate_density_delta_min),
        "moderate_entropy_margin": float(state_profile.get("target_profile_entropy", 0.0)) - float(cfg.state_profile_moderate_entropy_min),
        "stable_density_margin": float(cfg.state_profile_stable_density_delta_abs_max) - abs(float(state_profile.get("target_profile_density_delta", 0.0))),
        "shift_density_margin": float(state_profile.get("target_profile_density_delta", 0.0)) - float(cfg.state_profile_shift_density_delta_min),
    }
    state_alpha_bias = float(state_profile.get("alpha_bias", 0.0)) if bool(cfg.state_profile_alpha_enabled) else 0.0
    state_core_weight = _core_weight_for_profile(state_profile, cfg)
    if bool(cfg.state_profile_alpha_enabled) or bool(cfg.state_profile_alpha_bounds_enabled):
        new_alphas = []
        for row, alpha in zip(gate_rows, alphas):
            pre_state = float(alpha)
            after_bias = max(float(cfg.min_alpha), min(float(cfg.max_alpha), pre_state + state_alpha_bias))
            post_state, bound_reason = _apply_state_profile_alpha_package(after_bias, state_profile, cfg)
            row["state_profile_alpha_pre"] = float(pre_state)
            row["state_profile_alpha_bias"] = float(state_alpha_bias)
            row["state_profile_alpha_after_bias"] = float(after_bias)
            row["state_profile_alpha_bound_reason"] = str(bound_reason)
            row["state_profile_alpha_post"] = float(post_state)
            row["alpha"] = float(post_state)
            new_alphas.append(post_state)
        alphas = new_alphas
    else:
        for row, alpha in zip(gate_rows, alphas):
            row["state_profile_alpha_pre"] = float(alpha)
            row["state_profile_alpha_bias"] = float(0.0)
            row["state_profile_alpha_after_bias"] = float(alpha)
            row["state_profile_alpha_bound_reason"] = "disabled"
            row["state_profile_alpha_post"] = float(alpha)
    core_source_mode = _state_profile_core_source_mode_for_profile(state_profile, cfg)
    core_nodes: list[tuple[int, ...]] = [tuple() for _ in range(q_count)]
    if state_core_weight > 0.0:
        # Use high-confidence branch/agreement cores as a label-free community
        # prototype.  The refined default can prefer nodes jointly supported by
        # both branches, then fall back to the state-preferred branch.
        core_profile = str(state_profile.get("profile_id", "stable_default_profile"))
        for i in range(q_count):
            agreement = agreement_core_node_sets[i]
            pres_core_i = preserved_core_node_sets[i]
            adapt_core_i = adapted_core_node_sets[i]
            if core_profile == "boundary_preserve_profile":
                profile_core = pres_core_i
            elif core_profile in {"moderate_recall_profile", "strong_recovery_profile", "shift_adaptive_profile"}:
                profile_core = adapt_core_i
            else:
                profile_core = _merge_core_nodes(pres_core_i, adapt_core_i, limit=int(cfg.state_profile_core_max_nodes))
            min_agree = int(cfg.state_profile_core_min_agreement_nodes)
            enough_agreement = len(agreement) >= max(0, min_agree)
            if core_source_mode == "agreement_top":
                chosen = agreement if enough_agreement else profile_core
            elif core_source_mode == "agreement_then_profile":
                if enough_agreement:
                    chosen = _merge_core_nodes(agreement, profile_core, limit=int(cfg.state_profile_core_max_nodes))
                else:
                    chosen = profile_core
            else:
                chosen = profile_core
            core_nodes[i] = tuple(int(x) for x in chosen[: int(cfg.state_profile_core_max_nodes)])
        pres_core = _load_node_set_mean_embs(preserved_manifest_path, core_nodes, max_nodes=int(cfg.state_profile_core_max_nodes))
        adapt_core = _load_node_set_mean_embs(adapted_manifest_path, core_nodes, max_nodes=int(cfg.state_profile_core_max_nodes))
        if bool(cfg.state_profile_core_reliability_enabled):
            core_rel_t, core_rel_rows = _core_reliability_tensor(
                q_pres=q_pres_original,
                q_adapt=q_adapt_original,
                pres_core=pres_core,
                adapt_core=adapt_core,
                core_nodes=core_nodes,
                agreement_details=agreement_core_detail_sets,
                cfg=cfg,
            )
        else:
            core_rel_t = torch.ones(q_count, dtype=torch.float32)
            core_rel_rows = [{
                "state_profile_core_reliability": 1.0,
                "state_profile_core_reliability_count_score": 1.0,
                "state_profile_core_reliability_agreement_score": 1.0,
                "state_profile_core_reliability_query_sim_score": 1.0,
                "state_profile_core_query_sim_preserved": 0.0,
                "state_profile_core_query_sim_adapted": 0.0,
            } for _ in range(q_count)]
        eff_core_w = (float(state_core_weight) * core_rel_t).view(-1, 1)
        q_pres = F.normalize((1.0 - eff_core_w) * q_pres + eff_core_w * pres_core, p=2, dim=-1)
        q_adapt = F.normalize((1.0 - eff_core_w) * q_adapt + eff_core_w * adapt_core, p=2, dim=-1)
        mean_pres, std_pres = _branch_stats(manifest_path=preserved_manifest_path, query_embs=q_pres, chunk_size=chunk_size)
        mean_adapt, std_adapt = _branch_stats(manifest_path=adapted_manifest_path, query_embs=q_adapt, chunk_size=chunk_size)
    else:
        core_rel_t = torch.ones(q_count, dtype=torch.float32)
        core_rel_rows = [{
            "state_profile_core_reliability": 0.0,
            "state_profile_core_reliability_count_score": 0.0,
            "state_profile_core_reliability_agreement_score": 0.0,
            "state_profile_core_reliability_query_sim_score": 0.0,
            "state_profile_core_query_sim_preserved": 0.0,
            "state_profile_core_query_sim_adapted": 0.0,
        } for _ in range(q_count)]
    for row in gate_rows:
        row["state_profile_router_enabled"] = bool(cfg.state_profile_router_enabled)
        row["state_profile_id"] = str(state_profile.get("profile_id", "disabled"))
        row["state_profile_reason"] = str(state_profile.get("reason", ""))
        qidx = int(row.get("query_index", 0))
        query_profile = query_counterfactual_profiles[qidx] if qidx < len(query_counterfactual_profiles) else {}
        row["state_profile_query_counterfactual_id"] = str(query_profile.get("profile_id", "disabled"))
        row["state_profile_query_counterfactual_reason"] = str(query_profile.get("reason", ""))
        row["state_profile_query_agrees_with_target"] = bool(
            str(query_profile.get("profile_id", "disabled")) == str(state_profile.get("profile_id", "disabled"))
        )
        row["state_profile_query_population_agreement_rate"] = float(query_profile_agreement_rate)
        row["state_profile_query_population_entropy"] = float(query_profile_entropy)
        row["state_profile_query_population_entropy_normalized"] = float(query_profile_entropy_normalized)
        row["state_profile_query_population_profile_count"] = int(len(query_profile_counts))
        row["state_profile_query_population_counts_json"] = json.dumps(query_profile_counts, sort_keys=True)
        for margin_key, margin_value in route_threshold_margins.items():
            row[f"state_profile_route_margin_{margin_key}"] = float(margin_value)
        row["state_profile_search_mode"] = str(state_profile.get("search_mode", "default"))
        row["state_profile_core_weight"] = float(state_core_weight)
        qidx = int(row.get("query_index", 0))
        rel_row = core_rel_rows[qidx] if qidx < len(core_rel_rows) else {}
        row["state_profile_core_reliability_enabled"] = bool(cfg.state_profile_core_reliability_enabled)
        for _k, _v in rel_row.items():
            row[_k] = float(_v)
        row["state_profile_core_effective_weight"] = float(state_core_weight) * float(row.get("state_profile_core_reliability", 1.0))
        detail = agreement_core_detail_sets[qidx] if qidx < len(agreement_core_detail_sets) else {}
        profile_count = 0
        if qidx < len(preserved_core_node_sets):
            if str(state_profile.get("profile_id", "")) == "boundary_preserve_profile":
                profile_count = len(preserved_core_node_sets[qidx])
            elif str(state_profile.get("profile_id", "")) in {"moderate_recall_profile", "strong_recovery_profile", "shift_adaptive_profile"}:
                profile_count = len(adapted_core_node_sets[qidx])
            else:
                profile_count = len(_merge_core_nodes(preserved_core_node_sets[qidx], adapted_core_node_sets[qidx], limit=int(cfg.state_profile_core_max_nodes)))
        row["state_profile_core_source_mode"] = str(cfg.state_profile_core_source_mode)
        row["state_profile_core_effective_source_mode"] = str(core_source_mode)
        row["state_profile_core_source_by_state_enabled"] = bool(cfg.state_profile_core_source_by_state_enabled)
        row["state_profile_core_node_count"] = int(len(core_nodes[qidx])) if state_core_weight > 0.0 else 0
        # Node ids are retained only in the local diagnostic JSONL.  They are
        # never consumed by routing and allow a later GT-authorized offline
        # analysis to measure core precision/coverage without rerunning search.
        row["state_profile_core_node_ids"] = (
            [int(x) for x in core_nodes[qidx]] if state_core_weight > 0.0 else []
        )
        row["state_profile_core_profile_node_count"] = int(profile_count) if state_core_weight > 0.0 else 0
        row["state_profile_core_agreement_node_count"] = int(len(agreement_core_node_sets[qidx]))
        row["state_profile_core_candidate_agreement_node_count"] = int(detail.get("candidate_agreement_count", 0))
        row["state_profile_core_rank_agreement_node_count"] = int(detail.get("rank_agreement_count", 0))
        row["state_profile_core_rank_fallback_used"] = bool(detail.get("used_rank_fallback", False))
        row["state_profile_core_strict_agreement"] = bool(cfg.state_profile_core_strict_agreement)
        row["state_profile_core_min_agreement_nodes"] = int(cfg.state_profile_core_min_agreement_nodes)
        row["state_profile_state_selection_stage"] = "pre_core"
        row["state_profile_structure_weight"] = float(_structure_weight_for_profile(state_profile, cfg))
        row["state_profile_target_size_ratio"] = float(state_profile.get("target_profile_size_ratio", 0.0))
        row["state_profile_target_branch_jaccard"] = float(state_profile.get("target_profile_branch_jaccard", 0.0))
        row["state_profile_target_density_delta"] = float(state_profile.get("target_profile_density_delta", 0.0))
        row["state_profile_target_disagreement"] = float(state_profile.get("target_profile_disagreement", 0.0))
        row["state_profile_target_entropy"] = float(state_profile.get("target_profile_entropy", 0.0))
        row["state_profile_route_version"] = str(state_profile.get("target_profile_route_version", "legacy"))
        row["state_profile_target_adapted_over_rate"] = float(state_profile.get("target_profile_adapted_over_rate", 0.0))
        row["state_profile_target_adapted_strong_rate"] = float(state_profile.get("target_profile_adapted_strong_rate", 0.0))
        row["state_profile_target_stable_close_rate"] = float(state_profile.get("target_profile_stable_close_rate", 0.0))
        row["state_profile_target_moderate_eligible_rate"] = float(state_profile.get("target_profile_moderate_eligible_rate", 0.0))
        row["state_profile_target_shift_eligible_rate"] = float(state_profile.get("target_profile_shift_eligible_rate", 0.0))
        row["state_profile_target_density_delta_abs"] = float(state_profile.get("target_profile_density_delta_abs", 0.0))
        _score_mode = str(_state_profile_score_mode_for_profile(state_profile, cfg))
        _score_scale = float(_state_profile_score_residual_scale_for_profile(state_profile, cfg))
        _profile_id = str(state_profile.get("profile_id", "stable_default_profile"))
        _action_allowed = _state_profile_score_action_allowed(state_profile, cfg)
        _action_applies = bool(cfg.state_profile_score_source_enabled) and bool(_action_allowed)
        if not _action_allowed:
            _score_mode = "default_fusion"
            _score_scale = 1.0
        row["state_profile_score_source_enabled"] = bool(cfg.state_profile_score_source_enabled)
        row["state_profile_action_isolation_enabled"] = bool(cfg.state_profile_action_isolation_enabled)
        row["state_profile_action_isolation_scope"] = str(cfg.state_profile_action_isolation_scope)
        row["state_profile_score_action_applied"] = bool(_action_applies)
        if not bool(cfg.state_profile_action_isolation_enabled):
            _action_target = "all_profiles"
        elif str(cfg.state_profile_action_isolation_scope) == "moderate_only":
            _action_target = "moderate_recall_profile"
        else:
            _action_target = "configured_profile_action"
        row["state_profile_score_action_target_profile"] = _action_target
        row["state_profile_score_mode"] = str(_score_mode)
        row["state_profile_score_residual_scale"] = float(_score_scale)
        row["state_profile_score_recall_weight_moderate"] = float(cfg.state_profile_score_recall_weight_moderate)
        row["state_profile_score_consensus_weight_boundary"] = float(cfg.state_profile_score_consensus_weight_boundary)
        row["state_profile_score_adapt_bonus_shift"] = float(cfg.state_profile_score_adapt_bonus_shift)

    graph_bias, graph_state = _compute_graph_state_bias(gate_rows, cfg)
    graph_state = dict(graph_state)
    graph_state["state_profile"] = dict(state_profile)
    graph_state["query_counterfactual_profile_counts"] = dict(query_profile_counts)
    graph_state["query_counterfactual_profile_agreement_rate"] = float(query_profile_agreement_rate)
    graph_state["query_counterfactual_profile_entropy"] = float(query_profile_entropy)
    graph_state["query_counterfactual_profile_entropy_normalized"] = float(query_profile_entropy_normalized)
    graph_state["route_threshold_margins"] = dict(route_threshold_margins)
    if bool(cfg.graph_bias_enabled) and graph_bias != 0.0:
        new_alphas = []
        for row, alpha in zip(gate_rows, alphas):
            pre = float(alpha)
            post = max(float(cfg.min_alpha), min(float(cfg.max_alpha), pre + float(graph_bias)))
            row["alpha_pre_graph_bias"] = float(pre)
            row["graph_bias"] = float(graph_bias)
            row["alpha_post_graph_bias"] = float(post)
            row["alpha"] = float(post)
            new_alphas.append(post)
        alphas = new_alphas
    else:
        for row, alpha in zip(gate_rows, alphas):
            row["alpha_pre_graph_bias"] = float(alpha)
            row["graph_bias"] = 0.0
            row["alpha_post_graph_bias"] = float(alpha)
            row["alpha"] = float(alpha)

    for row in gate_rows:
        row["state_profile_alpha_final"] = float(row.get("alpha", cfg.fixed_alpha))
        row["state_profile_alpha_final_reason"] = "post_graph_bias" if bool(cfg.graph_bias_enabled) and graph_bias != 0.0 else "post_state_profile"

    alpha_t = torch.tensor(alphas, dtype=torch.float32)

    pres_rows = tuple(dict(row) for row in pres_manifest.get("shards", []))
    adapt_rows = tuple(dict(row) for row in adapt_manifest.get("shards", []))
    if len(pres_rows) != len(adapt_rows):
        raise V5ProtocolError("dual branch manifests have shard count mismatch")

    def _combine_score_result(profile_for_score: Mapping[str, object]) -> ChunkedScoreResult:
        """Combine branch scores without materializing a full Q x N matrix."""
        top_scores = torch.empty((q_count, 0), dtype=torch.float32)
        top_nodes = torch.empty((q_count, 0), dtype=torch.long)
        full_sums = torch.zeros(q_count, dtype=torch.float64)
        query_seed_scores: list[dict[int, float]] = [dict() for _ in range(q_count)]
        chunk_count = 0
        max_live_c = 0
        for shard_index, (prow, arow) in enumerate(zip(pres_rows, adapt_rows)):
            ps, pe = int(prow["start"]), int(prow["end"])
            as_, ae = int(arow["start"]), int(arow["end"])
            if ps != as_ or pe != ae:
                raise V5ProtocolError("dual branch shard ranges mismatch")
            pshard = _load_shard_tensor(
                pres_manifest_file.parent / str(prow["path"]), start=ps, end=pe
            ).float()
            ashard = _load_shard_tensor(
                adapt_manifest_file.parent / str(arow["path"]), start=as_, end=ae
            ).float()
            for local_start in range(0, int(pshard.shape[0]), int(chunk_size)):
                local_end = min(int(pshard.shape[0]), local_start + int(chunk_size))
                start = ps + local_start
                end = ps + local_end
                profile_event(runtime_jsonl, "dual_branch_score_chunk_start", shard_index=shard_index, start=start, end=end)
                pc = F.normalize(pshard[local_start:local_end], p=2, dim=-1)
                ac = F.normalize(ashard[local_start:local_end], p=2, dim=-1)
                sp = q_pres @ pc.t()
                sa = q_adapt @ ac.t()
                zp = ((sp.double() - mean_pres.view(-1, 1)) / std_pres.view(-1, 1)).float()
                za = ((sa.double() - mean_adapt.view(-1, 1)) / std_adapt.view(-1, 1)).float()
                final_scores = _apply_state_profile_score_source(
                    zp=zp,
                    za=za,
                    alpha=alpha_t.view(-1, 1),
                    cfg=cfg,
                    profile_or_id=profile_for_score,
                )
                chunk_count += 1
                max_live_c = max(max_live_c, int(final_scores.shape[1]))
                full_sums += final_scores.double().sum(dim=1)
                node_ids = torch.arange(start, end, dtype=torch.long).unsqueeze(0).expand(q_count, -1)
                merged_scores = torch.cat([top_scores, final_scores], dim=1)
                merged_nodes = torch.cat([top_nodes, node_ids], dim=1)
                keep = min(k_pool, int(merged_scores.shape[1]))
                top_scores, pos = torch.topk(merged_scores, k=keep, dim=1, largest=True, sorted=True)
                top_nodes = torch.gather(merged_nodes, dim=1, index=pos)
                for qi, qset in enumerate(qsets):
                    for seed in qset:
                        if start <= seed < end:
                            query_seed_scores[qi][int(seed)] = float(
                                final_scores[qi, int(seed) - start].detach().cpu().item()
                            )
                profile_event(runtime_jsonl, "dual_branch_score_chunk_end", shard_index=shard_index, start=start, end=end)
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

    # Safe pass: hierarchical boundary routing deliberately keeps the parent
    # boundary fallback on default fusion.  Non-boundary profiles retain their
    # established V574 actions, so they require only this single scoring pass.
    parent_state_profile = dict(state_profile)
    safe_result = _combine_score_result(parent_state_profile)
    state_profile, boundary_refinement = _refine_boundary_retrieval_state(
        safe_score_result=safe_result,
        gate_rows=gate_rows,
        parent_profile=parent_state_profile,
        cfg=cfg,
    )
    if str(state_profile.get("profile_id", "")) == "boundary_plateau_precision_profile":
        result = _combine_score_result(state_profile)
    else:
        result = safe_result

    query_boundary_rows = {
        int(row.get("query_index", -1)): row
        for row in boundary_refinement.get("query_rows", [])
        if isinstance(row, Mapping)
    }
    for row in gate_rows:
        qi = int(row.get("query_index", -1))
        row["state_profile_parent_id"] = str(parent_state_profile.get("profile_id", "disabled"))
        row["state_profile_id"] = str(state_profile.get("profile_id", "disabled"))
        row["state_profile_reason"] = str(state_profile.get("reason", ""))
        row["state_profile_search_mode"] = str(state_profile.get("search_mode", "default"))
        row["state_profile_state_selection_stage"] = (
            "post_safe_endpoint_refinement"
            if bool(boundary_refinement.get("eligible", False))
            else "pre_core"
        )
        row["hierarchical_boundary_state_enabled"] = bool(cfg.hierarchical_boundary_state_enabled)
        row["hierarchical_boundary_state_eligible"] = bool(boundary_refinement.get("eligible", False))
        row["boundary_target_normalized_endpoint_marginal"] = float(
            boundary_refinement.get("target_normalized_endpoint_marginal", float("nan"))
        )
        row["boundary_plateau_relative_quality_threshold"] = float(
            cfg.boundary_plateau_relative_quality_threshold
        )
        row["boundary_plateau_threshold_margin"] = float(
            boundary_refinement.get("threshold_margin", float("nan"))
        )
        row["boundary_query_plateau_rate"] = float(boundary_refinement.get("query_plateau_rate", 0.0))
        for key, value in query_boundary_rows.get(qi, {}).items():
            if key != "query_index":
                row[f"boundary_{key}"] = value
        score_mode = _state_profile_score_mode_for_profile(state_profile, cfg)
        score_scale = _state_profile_score_residual_scale_for_profile(state_profile, cfg)
        action_allowed = _state_profile_score_action_allowed(state_profile, cfg)
        row["state_profile_score_action_applied"] = bool(
            cfg.state_profile_score_source_enabled and action_allowed and score_mode != "default_fusion"
        )
        row["state_profile_score_mode"] = str(score_mode if action_allowed else "default_fusion")
        row["state_profile_score_residual_scale"] = float(score_scale if action_allowed else 1.0)

    graph_state["state_profile_parent"] = dict(parent_state_profile)
    graph_state["state_profile"] = dict(state_profile)
    graph_state["hierarchical_boundary_refinement"] = {
        key: value for key, value in boundary_refinement.items() if key != "query_rows"
    }
    report = {
        "dual_branch_score": True,
        "labels_used": False,
        "ground_truth_used": False,
        "metrics_computed": False,
        "config": cfg.to_dict(),
        "alpha_mean": float(sum(alphas) / max(1, len(alphas))),
        "alpha_min": float(min(alphas) if alphas else 0.0),
        "alpha_max": float(max(alphas) if alphas else 0.0),
        "graph_state": graph_state,
        "alpha_rows": gate_rows,
        "preserved_branch_summary": pres_result.to_dict(),
        "adapted_branch_summary": adapt_result.to_dict(),
    }
    return result, report



def _select_floor_route(
    *,
    cfg: DualBranchGateConfig,
    target_underfloor_rate: float,
    target_probe_rel_quality_drop: float,
    target_probe_rel_density_drop: float,
    target_branch_jaccard_mean: float,
) -> dict[str, object]:
    """Select a label-free target-level floor-recovery route.

    The route is intentionally based only on unsupervised diagnostics computed
    from the target query workload and branch predictions.  It does not inspect
    graph names, labels, or evaluation metrics.
    """
    if not bool(cfg.guarded_floor_enabled):
        return {
            "enabled": False,
            "ratio": float(cfg.guarded_floor_ratio),
            "accept_mode": str(cfg.guarded_floor_accept_mode),
            "state": "floor_disabled",
            "reason": "guarded_floor_disabled",
            "moderate_max_branch_jaccard": float(cfg.guarded_floor_router_moderate_max_branch_jaccard),
            "moderate_quality_drop_threshold": float(cfg.guarded_floor_router_quality_drop_threshold),
            "moderate_density_drop_threshold": float(cfg.guarded_floor_router_density_drop_threshold),
            "disagreement_pass": True,
            "quality_drop_pass": True,
            "density_drop_pass": True,
        }
    if not bool(cfg.guarded_floor_router_enabled):
        enabled = bool(
            not bool(cfg.guarded_floor_target_trigger)
            or float(target_underfloor_rate) >= float(cfg.guarded_floor_target_under_rate_threshold)
        )
        return {
            "enabled": enabled,
            "ratio": float(cfg.guarded_floor_ratio),
            "accept_mode": str(cfg.guarded_floor_accept_mode),
            "state": "single_floor_enabled" if enabled else "single_floor_disabled",
            "reason": "legacy_target_trigger" if bool(cfg.guarded_floor_target_trigger) else "legacy_query_level_floor",
            "moderate_max_branch_jaccard": float(cfg.guarded_floor_router_moderate_max_branch_jaccard),
            "moderate_quality_drop_threshold": float(cfg.guarded_floor_router_quality_drop_threshold),
            "moderate_density_drop_threshold": float(cfg.guarded_floor_router_density_drop_threshold),
            "disagreement_pass": True,
            "quality_drop_pass": True,
            "density_drop_pass": True,
        }

    under = float(target_underfloor_rate)
    qdrop = float(target_probe_rel_quality_drop)
    strong_thr = float(cfg.guarded_floor_router_strong_under_rate_threshold)
    mid_thr = float(cfg.guarded_floor_router_moderate_under_rate_threshold)
    qdrop_thr = float(cfg.guarded_floor_router_quality_drop_threshold)
    ddrop_thr = float(cfg.guarded_floor_router_density_drop_threshold)
    max_jac = float(cfg.guarded_floor_router_moderate_max_branch_jaccard)
    branch_jac = float(target_branch_jaccard_mean)
    ddrop = float(target_probe_rel_density_drop)
    disagreement_pass = bool(branch_jac <= max_jac + float(cfg.eps))
    quality_pass = bool(qdrop <= qdrop_thr + float(cfg.eps))
    density_pass = bool(ddrop <= ddrop_thr + float(cfg.eps))
    if under >= strong_thr:
        return {
            "enabled": True,
            "ratio": float(cfg.guarded_floor_router_strong_ratio),
            "accept_mode": str(cfg.guarded_floor_router_strong_accept_mode),
            "state": "strong_underexpanded",
            "reason": f"underfloor_rate_{under:.4f}_ge_strong_{strong_thr:.4f}",
            "moderate_max_branch_jaccard": max_jac,
            "moderate_quality_drop_threshold": qdrop_thr,
            "moderate_density_drop_threshold": ddrop_thr,
            "disagreement_pass": True,
            "quality_drop_pass": quality_pass,
            "density_drop_pass": density_pass,
        }
    if under >= mid_thr:
        if quality_pass and density_pass and disagreement_pass:
            return {
                "enabled": True,
                "ratio": float(cfg.guarded_floor_router_moderate_ratio),
                "accept_mode": str(cfg.guarded_floor_router_moderate_accept_mode),
                "state": "moderate_underexpanded",
                "reason": (
                    f"underfloor_rate_{under:.4f}_ge_moderate_{mid_thr:.4f}_"
                    f"quality_drop_{qdrop:.4f}_le_{qdrop_thr:.4f}_"
                    f"density_drop_{ddrop:.4f}_le_{ddrop_thr:.4f}_"
                    f"branch_jaccard_{branch_jac:.4f}_le_{max_jac:.4f}"
                ),
                "moderate_max_branch_jaccard": max_jac,
                "moderate_quality_drop_threshold": qdrop_thr,
                "moderate_density_drop_threshold": ddrop_thr,
                "disagreement_pass": True,
                "quality_drop_pass": True,
                "density_drop_pass": True,
            }
        reasons = []
        if not quality_pass:
            reasons.append(f"quality_drop_{qdrop:.4f}_gt_{qdrop_thr:.4f}")
        if not density_pass:
            reasons.append(f"density_drop_{ddrop:.4f}_gt_{ddrop_thr:.4f}")
        if not disagreement_pass:
            reasons.append(f"branch_jaccard_{branch_jac:.4f}_gt_{max_jac:.4f}")
        return {
            "enabled": False,
            "ratio": float(cfg.guarded_floor_router_moderate_ratio),
            "accept_mode": str(cfg.guarded_floor_router_moderate_accept_mode),
            "state": "boundary_sensitive",
            "reason": "moderate_underfloor_but_" + "_and_".join(reasons or ["guard_failed"]),
            "moderate_max_branch_jaccard": max_jac,
            "moderate_quality_drop_threshold": qdrop_thr,
            "moderate_density_drop_threshold": ddrop_thr,
            "disagreement_pass": disagreement_pass,
            "quality_drop_pass": quality_pass,
            "density_drop_pass": density_pass,
        }
    if (not quality_pass) or (not density_pass):
        reasons = []
        if not quality_pass:
            reasons.append(f"quality_drop_{qdrop:.4f}_gt_{qdrop_thr:.4f}")
        if not density_pass:
            reasons.append(f"density_drop_{ddrop:.4f}_gt_{ddrop_thr:.4f}")
        return {
            "enabled": False,
            "ratio": float(cfg.guarded_floor_router_moderate_ratio),
            "accept_mode": str(cfg.guarded_floor_router_moderate_accept_mode),
            "state": "boundary_sensitive",
            "reason": "low_underfloor_" + "_and_".join(reasons),
            "moderate_max_branch_jaccard": max_jac,
            "moderate_quality_drop_threshold": qdrop_thr,
            "moderate_density_drop_threshold": ddrop_thr,
            "disagreement_pass": disagreement_pass,
            "quality_drop_pass": quality_pass,
            "density_drop_pass": density_pass,
        }
    return {
        "enabled": False,
        "ratio": float(cfg.guarded_floor_ratio),
        "accept_mode": str(cfg.guarded_floor_accept_mode),
        "state": "stable_default",
        "reason": f"underfloor_rate_{under:.4f}_lt_moderate_{mid_thr:.4f}",
        "moderate_max_branch_jaccard": max_jac,
        "moderate_quality_drop_threshold": qdrop_thr,
        "moderate_density_drop_threshold": ddrop_thr,
        "disagreement_pass": disagreement_pass,
        "quality_drop_pass": quality_pass,
        "density_drop_pass": density_pass,
    }

def build_dual_branch_prediction_records(
    *,
    graph_name: str,
    score_result: ChunkedScoreResult,
    dual_report: Mapping[str, object],
    global_node_ids: Sequence[int] | None = None,
    config: DualBranchGateConfig | None = None,
    edge_index: torch.Tensor | None = None,
) -> tuple["CommunityPredictionRecord", ...]:
    from .community_output import CommunityPredictionRecord

    cfg = (config or DualBranchGateConfig(enabled=True)).validate()
    gate_rows = {int(row["query_index"]): row for row in dual_report.get("alpha_rows", []) if isinstance(row, dict)}
    adjacency = _edge_index_to_adjacency(edge_index, int(score_result.num_nodes)) if bool(cfg.state_profile_structure_enabled) else None

    # First pass: compute the original prefix decision for every query without
    # accepting guarded floor recovery.  This allows a target-level label-free
    # trigger based on final-vs-branch undersize rate, avoiding hard-coded target
    # names while preventing unnecessary floor expansion on stable targets.
    base_decisions: dict[int, PrefixSearchDecision] = {}
    underfloor_hits = 0
    underfloor_total = 0
    probe_rel_quality_drops: list[float] = []
    probe_rel_density_drops: list[float] = []
    branch_jaccards: list[float] = []
    for qi, qset in enumerate(score_result.query_node_sets):
        ranked_nodes = [int(x) for x in score_result.top_nodes[qi].tolist()]
        ranked_scores = [float(x) for x in score_result.top_scores[qi].tolist()]
        q_scores = [float(score_result.query_seed_scores[qi][int(seed)]) for seed in qset]
        gate = gate_rows.get(int(qi), {})
        if cfg.search_profile == "senior":
            continue
        dec0 = _choose_recalibrated_prefix(
            q_scores=q_scores,
            ranked_nodes=ranked_nodes,
            ranked_scores=ranked_scores,
            full_score_sum=float(score_result.full_score_sums[qi].item()),
            num_nodes=int(score_result.num_nodes),
            alpha=float(gate.get("alpha", cfg.fixed_alpha)),
            preserved_size=int(gate.get("preserved_size", len(qset))),
            adapted_size=int(gate.get("adapted_size", len(qset))),
            entropy=float(max(float(gate.get("preserved_entropy", 0.0)), float(gate.get("adapted_entropy", 0.0)))),
            cfg=cfg,
            target_floor_enabled=False,
            target_underfloor_rate=0.0,
        )
        base_decisions[int(qi)] = dec0
        if int(dec0.max_branch_size) > 0:
            underfloor_total += 1
            if float(dec0.best_size) + cfg.eps < float(cfg.guarded_floor_probe_ratio) * float(dec0.max_branch_size):
                underfloor_hits += 1
            if math.isfinite(float(dec0.rel_quality_drop)):
                probe_rel_quality_drops.append(float(dec0.rel_quality_drop))
            if math.isfinite(float(dec0.rel_density_drop)):
                probe_rel_density_drops.append(float(dec0.rel_density_drop))
            if "preserved_adapted_jaccard" in gate:
                try:
                    branch_jaccards.append(float(gate.get("preserved_adapted_jaccard", 0.0)))
                except Exception:
                    pass
    target_underfloor_rate = float(underfloor_hits) / float(max(1, underfloor_total))
    target_overexpansion_rate = 0.0
    target_overexpansion_action = False
    target_overexpansion_reason = "overexpansion_router_disabled"
    target_probe_rel_quality_drop = float(sum(probe_rel_quality_drops) / max(1, len(probe_rel_quality_drops)))
    target_probe_rel_density_drop = float(sum(probe_rel_density_drops) / max(1, len(probe_rel_density_drops)))
    target_branch_jaccard_mean = float(sum(branch_jaccards) / max(1, len(branch_jaccards)))
    route = _select_floor_route(
        cfg=cfg,
        target_underfloor_rate=target_underfloor_rate,
        target_probe_rel_quality_drop=target_probe_rel_quality_drop,
        target_probe_rel_density_drop=target_probe_rel_density_drop,
        target_branch_jaccard_mean=target_branch_jaccard_mean,
    )
    target_floor_enabled = bool(route.get("enabled", False))
    effective_floor_ratio = float(route.get("ratio", cfg.guarded_floor_ratio))
    effective_accept_mode = str(route.get("accept_mode", cfg.guarded_floor_accept_mode))
    floor_route_state = str(route.get("state", "single"))
    floor_route_reason = str(route.get("reason", "single_guarded_floor"))
    target_moderate_max_branch_jaccard = float(route.get("moderate_max_branch_jaccard", cfg.guarded_floor_router_moderate_max_branch_jaccard))
    target_moderate_quality_drop_threshold = float(route.get("moderate_quality_drop_threshold", cfg.guarded_floor_router_quality_drop_threshold))
    target_moderate_density_drop_threshold = float(route.get("moderate_density_drop_threshold", cfg.guarded_floor_router_density_drop_threshold))
    floor_router_disagreement_pass = bool(route.get("disagreement_pass", target_branch_jaccard_mean <= target_moderate_max_branch_jaccard + cfg.eps))
    floor_router_quality_drop_pass = bool(route.get("quality_drop_pass", target_probe_rel_quality_drop <= target_moderate_quality_drop_threshold + cfg.eps))
    floor_router_density_drop_pass = bool(route.get("density_drop_pass", target_probe_rel_density_drop <= target_moderate_density_drop_threshold + cfg.eps))

    # Complete state-profile package override.  The older floor router uses a
    # single set of floor thresholds; the V5.6.8 package router lets the
    # unsupervised state profile choose whether floor recovery should be active
    # and which ratio/accept mode to use.  This is parameter routing, not a
    # candidate-result bank.
    target_state_profile_id = "disabled"
    if gate_rows:
        try:
            target_state_profile_id = str(next(iter(gate_rows.values())).get("state_profile_id", "disabled"))
        except Exception:
            target_state_profile_id = "disabled"
    state_floor_pkg = _state_profile_floor_package(target_state_profile_id, cfg)
    if bool(cfg.state_profile_floor_enabled):
        target_floor_enabled = bool(state_floor_pkg.get("enabled", False))
        effective_floor_ratio = float(state_floor_pkg.get("ratio", 0.0)) if target_floor_enabled else float(cfg.guarded_floor_ratio)
        effective_accept_mode = str(state_floor_pkg.get("accept_mode", cfg.guarded_floor_accept_mode))
        floor_route_state = str(state_floor_pkg.get("state", floor_route_state))
        floor_route_reason = str(state_floor_pkg.get("reason", floor_route_reason))

    # Probe the real pre-repair B1 endpoint, including the active state-search
    # profile and historical floor package.  This is intentionally separate
    # from the legacy undersize probe above so enabling the new router cannot
    # change old B1 behavior when it is disabled.
    if bool(cfg.overexpansion_router_enabled):
        overexpansion_hits = 0
        overexpansion_total = 0
        for qi, qset in enumerate(score_result.query_node_sets):
            ranked_nodes = [int(x) for x in score_result.top_nodes[qi].tolist()]
            ranked_scores = [float(x) for x in score_result.top_scores[qi].tolist()]
            q_scores = [float(score_result.query_seed_scores[qi][int(seed)]) for seed in qset]
            gate = gate_rows.get(int(qi), {})
            state_profile_structure_weight = float(gate.get("state_profile_structure_weight", 0.0))
            struct_scores = None
            if bool(cfg.state_profile_structure_enabled) and state_profile_structure_weight > 0.0:
                struct_scores = _ranked_structure_scores(
                    ranked_nodes=ranked_nodes,
                    query_nodes=qset,
                    adjacency=adjacency,
                    core_topk=int(cfg.state_profile_structure_core_topk),
                )
            probe = _choose_recalibrated_prefix(
                q_scores=q_scores,
                ranked_nodes=ranked_nodes,
                ranked_scores=ranked_scores,
                full_score_sum=float(score_result.full_score_sums[qi].item()),
                num_nodes=int(score_result.num_nodes),
                alpha=float(gate.get("alpha", cfg.fixed_alpha)),
                preserved_size=int(gate.get("preserved_size", len(qset))),
                adapted_size=int(gate.get("adapted_size", len(qset))),
                entropy=float(max(float(gate.get("preserved_entropy", 0.0)), float(gate.get("adapted_entropy", 0.0)))),
                cfg=cfg,
                target_floor_enabled=target_floor_enabled,
                target_underfloor_rate=target_underfloor_rate,
                effective_floor_ratio=effective_floor_ratio,
                effective_accept_mode=effective_accept_mode,
                floor_route_state=floor_route_state,
                floor_route_reason=floor_route_reason,
                target_moderate_underfloor_rate=target_underfloor_rate,
                target_strong_underfloor_rate=target_underfloor_rate,
                target_probe_rel_quality_drop=target_probe_rel_quality_drop,
                target_probe_rel_density_drop=target_probe_rel_density_drop,
                target_branch_jaccard_mean=target_branch_jaccard_mean,
                target_moderate_max_branch_jaccard=target_moderate_max_branch_jaccard,
                target_moderate_quality_drop_threshold=target_moderate_quality_drop_threshold,
                target_moderate_density_drop_threshold=target_moderate_density_drop_threshold,
                floor_router_disagreement_pass=floor_router_disagreement_pass,
                floor_router_quality_drop_pass=floor_router_quality_drop_pass,
                floor_router_density_drop_pass=floor_router_density_drop_pass,
                state_profile_id=str(gate.get("state_profile_id", "disabled")),
                state_profile_reason=str(gate.get("state_profile_reason", "")),
                state_profile_alpha_bias=float(gate.get("state_profile_alpha_bias", 0.0)),
                state_profile_core_weight=float(gate.get("state_profile_core_weight", 0.0)),
                state_profile_structure_weight=state_profile_structure_weight,
                state_profile_search_mode=str(gate.get("state_profile_search_mode", "default")),
                ranked_structure_scores=struct_scores,
            )
            if int(probe.max_branch_size) > 0:
                overexpansion_total += 1
                if float(probe.final_size) > float(cfg.overexpansion_ratio_threshold) * float(probe.max_branch_size) + cfg.eps:
                    overexpansion_hits += 1
        target_overexpansion_rate = float(overexpansion_hits) / float(max(1, overexpansion_total))
        target_overexpansion_action = bool(
            target_overexpansion_rate + cfg.eps >= float(cfg.overexpansion_rate_threshold)
        )
        relation = "ge" if target_overexpansion_action else "lt"
        target_overexpansion_reason = (
            f"target_overexpansion_rate_{target_overexpansion_rate:.4f}_{relation}_"
            f"{float(cfg.overexpansion_rate_threshold):.4f}_at_ratio_"
            f"{float(cfg.overexpansion_ratio_threshold):.4f}"
        )
    # Precision recovery and recall floor are mutually exclusive actions for a
    # target.  Otherwise an accepted shorter endpoint could be immediately
    # expanded by the historical undersize package.
    if bool(target_overexpansion_action):
        target_floor_enabled = False
        floor_route_reason = "disabled_by_target_overexpansion_action"

    records: list[CommunityPredictionRecord] = []
    for qi, qset in enumerate(score_result.query_node_sets):
        ranked_nodes = [int(x) for x in score_result.top_nodes[qi].tolist()]
        ranked_scores = [float(x) for x in score_result.top_scores[qi].tolist()]
        q_scores = [float(score_result.query_seed_scores[qi][int(seed)]) for seed in qset]
        gate = gate_rows.get(int(qi), {})
        decision: PrefixSearchDecision | None = None
        state_profile_id = str(gate.get("state_profile_id", "disabled"))
        state_profile_reason = str(gate.get("state_profile_reason", ""))
        state_profile_alpha_bias = float(gate.get("state_profile_alpha_bias", 0.0))
        state_profile_core_weight = float(gate.get("state_profile_core_weight", 0.0))
        state_profile_structure_weight = float(gate.get("state_profile_structure_weight", 0.0))
        state_profile_search_mode = str(gate.get("state_profile_search_mode", "default"))
        struct_scores = None
        if bool(cfg.state_profile_structure_enabled) and state_profile_structure_weight > 0.0:
            struct_scores = _ranked_structure_scores(
                ranked_nodes=ranked_nodes,
                query_nodes=qset,
                adjacency=adjacency,
                core_topk=int(cfg.state_profile_structure_core_topk),
            )
        if cfg.search_profile == "senior":
            pred = senior_global_search_from_ranked_scores(
                query_nodes=qset,
                query_scores=q_scores,
                ranked_nodes=ranked_nodes,
                ranked_scores=ranked_scores,
                num_nodes=score_result.num_nodes,
                full_score_sum=float(score_result.full_score_sums[qi].item()),
            )
            search_reason = "senior_global_search"
        else:
            decision = _choose_recalibrated_prefix(
                q_scores=q_scores,
                ranked_nodes=ranked_nodes,
                ranked_scores=ranked_scores,
                full_score_sum=float(score_result.full_score_sums[qi].item()),
                num_nodes=int(score_result.num_nodes),
                alpha=float(gate.get("alpha", cfg.fixed_alpha)),
                preserved_size=int(gate.get("preserved_size", len(qset))),
                adapted_size=int(gate.get("adapted_size", len(qset))),
                entropy=float(max(float(gate.get("preserved_entropy", 0.0)), float(gate.get("adapted_entropy", 0.0)))),
                cfg=cfg,
                target_floor_enabled=target_floor_enabled,
                target_underfloor_rate=target_underfloor_rate,
                effective_floor_ratio=effective_floor_ratio,
                effective_accept_mode=effective_accept_mode,
                floor_route_state=floor_route_state,
                floor_route_reason=floor_route_reason,
                target_moderate_underfloor_rate=target_underfloor_rate,
                target_strong_underfloor_rate=target_underfloor_rate,
                target_probe_rel_quality_drop=target_probe_rel_quality_drop,
                target_probe_rel_density_drop=target_probe_rel_density_drop,
                target_branch_jaccard_mean=target_branch_jaccard_mean,
                target_overexpansion_rate=target_overexpansion_rate,
                target_overexpansion_action=target_overexpansion_action,
                target_overexpansion_reason=target_overexpansion_reason,
                target_moderate_max_branch_jaccard=target_moderate_max_branch_jaccard,
                target_moderate_quality_drop_threshold=target_moderate_quality_drop_threshold,
                target_moderate_density_drop_threshold=target_moderate_density_drop_threshold,
                floor_router_disagreement_pass=floor_router_disagreement_pass,
                floor_router_quality_drop_pass=floor_router_quality_drop_pass,
                floor_router_density_drop_pass=floor_router_density_drop_pass,
                state_profile_id=state_profile_id,
                state_profile_reason=state_profile_reason,
                state_profile_alpha_bias=state_profile_alpha_bias,
                state_profile_core_weight=state_profile_core_weight,
                state_profile_structure_weight=state_profile_structure_weight,
                state_profile_search_mode=state_profile_search_mode,
                ranked_structure_scores=struct_scores,
            )
            if int(decision.final_prefix_k) < 0:
                pred = senior_global_search_from_ranked_scores(
                    query_nodes=qset,
                    query_scores=q_scores,
                    ranked_nodes=ranked_nodes,
                    ranked_scores=ranked_scores,
                    num_nodes=score_result.num_nodes,
                    full_score_sum=float(score_result.full_score_sums[qi].item()),
                )
                search_reason = "fallback_senior_global_search"
            else:
                pred = tuple(dict.fromkeys(list(qset) + ranked_nodes[: int(decision.final_prefix_k)]))
                if bool(decision.ceiling_accepted):
                    suffix = "ceiling"
                elif bool(decision.floor_accepted):
                    suffix = "floor"
                else:
                    suffix = "base"
                search_reason = f"dual_branch_{cfg.search_profile}_{suffix}_prefix_{int(decision.final_prefix_k)}"
        query_global = tuple(int(global_node_ids[int(i)]) for i in qset) if global_node_ids is not None else None
        pred_global = tuple(int(global_node_ids[int(i)]) for i in pred) if global_node_ids is not None else None
        if decision is not None:
            floor_payload = {
                "dual_branch_best_prefix_k": int(decision.best_prefix_k),
                "dual_branch_final_prefix_k": int(decision.final_prefix_k),
                "dual_branch_best_size": int(decision.best_size),
                "dual_branch_final_size_before_floor": int(decision.best_size),
                "dual_branch_final_size_after_floor": int(decision.final_size),
                "dual_branch_final_size": int(decision.final_size),
                "dual_branch_best_density": float(decision.best_density),
                "dual_branch_final_density": float(decision.final_density),
                "dual_branch_best_quality": float(decision.best_quality),
                "dual_branch_final_quality": float(decision.final_quality),
                "dual_branch_floor_prefix_k": int(decision.floor_prefix_k),
                "dual_branch_floor_size": int(decision.floor_size),
                "dual_branch_floor_density": float(decision.floor_density),
                "dual_branch_floor_quality": float(decision.floor_quality),
                "dual_branch_density_drop": float(decision.density_drop),
                "dual_branch_quality_drop": float(decision.quality_drop),
                "dual_branch_rel_density_drop": float(decision.rel_density_drop),
                "dual_branch_rel_quality_drop": float(decision.rel_quality_drop),
                "dual_branch_floor_accept_mode": str(decision.floor_accept_mode),
                "dual_branch_floor_accept_reason": str(decision.floor_accept_reason),
                "dual_branch_floor_reject_reason": str(decision.floor_reject_reason),
                "dual_branch_query_floor_enabled": bool(decision.query_floor_enabled),
                "dual_branch_max_branch_size": int(decision.max_branch_size),
                "dual_branch_floor_ratio": float(decision.floor_ratio),
                "dual_branch_floor_triggered": bool(decision.floor_triggered),
                "dual_branch_floor_accepted": bool(decision.floor_accepted),
                "dual_branch_floor_reason": str(decision.floor_reason),
                "dual_branch_target_underfloor_rate": float(decision.target_underfloor_rate),
                "dual_branch_target_floor_enabled": bool(decision.target_floor_enabled),
                "dual_branch_floor_route_state": str(decision.floor_route_state),
                "dual_branch_floor_route_reason": str(decision.floor_route_reason),
                "dual_branch_effective_floor_ratio": float(decision.effective_floor_ratio),
                "dual_branch_effective_accept_mode": str(decision.effective_accept_mode),
                "dual_branch_target_moderate_underfloor_rate": float(decision.target_moderate_underfloor_rate),
                "dual_branch_target_strong_underfloor_rate": float(decision.target_strong_underfloor_rate),
                "dual_branch_target_probe_rel_quality_drop": float(decision.target_probe_rel_quality_drop),
                "dual_branch_target_probe_rel_density_drop": float(decision.target_probe_rel_density_drop),
                "dual_branch_target_branch_jaccard_mean": float(decision.target_branch_jaccard_mean),
                "dual_branch_target_moderate_max_branch_jaccard": float(decision.target_moderate_max_branch_jaccard),
                "dual_branch_target_moderate_quality_drop_threshold": float(decision.target_moderate_quality_drop_threshold),
                "dual_branch_target_moderate_density_drop_threshold": float(decision.target_moderate_density_drop_threshold),
                "dual_branch_floor_router_disagreement_pass": bool(decision.floor_router_disagreement_pass),
                "dual_branch_floor_router_quality_drop_pass": bool(decision.floor_router_quality_drop_pass),
                "dual_branch_floor_router_density_drop_pass": bool(decision.floor_router_density_drop_pass),
                "dual_branch_state_profile_id": str(decision.state_profile_id),
                "dual_branch_state_profile_parent_id": str(gate.get("state_profile_parent_id", decision.state_profile_id)),
                "dual_branch_state_profile_reason": str(decision.state_profile_reason),
                "dual_branch_hierarchical_boundary_state_enabled": bool(gate.get("hierarchical_boundary_state_enabled", False)),
                "dual_branch_hierarchical_boundary_state_eligible": bool(gate.get("hierarchical_boundary_state_eligible", False)),
                "dual_branch_boundary_target_normalized_endpoint_marginal": float(gate.get("boundary_target_normalized_endpoint_marginal", float("nan"))),
                "dual_branch_boundary_plateau_relative_quality_threshold": float(gate.get("boundary_plateau_relative_quality_threshold", cfg.boundary_plateau_relative_quality_threshold)),
                "dual_branch_boundary_plateau_threshold_margin": float(gate.get("boundary_plateau_threshold_margin", float("nan"))),
                "dual_branch_boundary_query_plateau_rate": float(gate.get("boundary_query_plateau_rate", 0.0)),
                "dual_branch_boundary_safe_normalized_endpoint_marginal": float(gate.get("boundary_safe_normalized_endpoint_marginal", float("nan"))),
                "dual_branch_state_profile_alpha_bias": float(decision.state_profile_alpha_bias),
                "dual_branch_state_profile_alpha_final": float(gate.get("state_profile_alpha_final", gate.get("alpha", cfg.fixed_alpha))),
                "dual_branch_state_profile_alpha_bound_reason": str(gate.get("state_profile_alpha_bound_reason", "")),
                "dual_branch_state_profile_alpha_final_reason": str(gate.get("state_profile_alpha_final_reason", "")),
                "dual_branch_state_profile_core_weight": float(decision.state_profile_core_weight),
                "dual_branch_state_profile_core_source_mode": str(gate.get("state_profile_core_source_mode", "")),
                "dual_branch_state_profile_core_node_count": int(gate.get("state_profile_core_node_count", 0)),
                "dual_branch_state_profile_core_agreement_node_count": int(gate.get("state_profile_core_agreement_node_count", 0)),
                "dual_branch_state_profile_structure_weight": float(decision.state_profile_structure_weight),
                "dual_branch_state_profile_search_mode": str(decision.state_profile_search_mode),
                "dual_branch_state_profile_query_proto_enabled": bool(decision.state_profile_query_proto_enabled),
                "dual_branch_state_profile_structure_enabled": bool(decision.state_profile_structure_enabled),
                "dual_branch_state_profile_floor_package_enabled": bool(state_floor_pkg.get("enabled", False)),
                "dual_branch_state_profile_floor_package_reason": str(state_floor_pkg.get("reason", "")),
                "dual_branch_search_endpoint": int(decision.search_endpoint),
                "dual_branch_precision_over_coefficient": float(decision.precision_over_coefficient),
                "dual_branch_precision_size_penalty": float(decision.precision_size_penalty),
                "dual_branch_target_overexpansion_rate": float(decision.target_overexpansion_rate),
                "dual_branch_target_overexpansion_action": bool(decision.target_overexpansion_action),
                "dual_branch_target_overexpansion_reason": str(decision.target_overexpansion_reason),
                "dual_branch_ceiling_prefix_k": int(decision.ceiling_prefix_k),
                "dual_branch_ceiling_size": int(decision.ceiling_size),
                "dual_branch_ceiling_density": float(decision.ceiling_density),
                "dual_branch_ceiling_quality": float(decision.ceiling_quality),
                "dual_branch_ceiling_rel_density_drop": float(decision.ceiling_rel_density_drop),
                "dual_branch_ceiling_rel_quality_drop": float(decision.ceiling_rel_quality_drop),
                "dual_branch_ceiling_triggered": bool(decision.ceiling_triggered),
                "dual_branch_ceiling_accepted": bool(decision.ceiling_accepted),
                "dual_branch_ceiling_reason": str(decision.ceiling_reason),
                "dual_branch_final_size_before_ceiling": int(decision.final_size_before_ceiling),
                "dual_branch_final_size_after_ceiling": int(decision.final_size_after_ceiling),
                "dual_branch_previous_prefix_density": float(decision.previous_prefix_density),
                "dual_branch_previous_prefix_quality": float(decision.previous_prefix_quality),
                "dual_branch_next_prefix_density": float(decision.next_prefix_density),
                "dual_branch_next_prefix_quality": float(decision.next_prefix_quality),
                "dual_branch_marginal_density_from_previous": float(decision.marginal_density_from_previous),
                "dual_branch_marginal_quality_from_previous": float(decision.marginal_quality_from_previous),
                "dual_branch_marginal_density_to_next": float(decision.marginal_density_to_next),
                "dual_branch_marginal_quality_to_next": float(decision.marginal_quality_to_next),
            }
            # Mutate the report row before diagnostics are written so the JSONL
            # contains the same floor fields as prediction metadata.
            gate.update({k.replace("dual_branch_", ""): v for k, v in floor_payload.items()})
        else:
            floor_payload = {
                "dual_branch_target_underfloor_rate": float(target_underfloor_rate),
                "dual_branch_target_floor_enabled": bool(target_floor_enabled),
                "dual_branch_floor_route_state": str(floor_route_state),
                "dual_branch_floor_route_reason": str(floor_route_reason),
                "dual_branch_effective_floor_ratio": float(effective_floor_ratio),
                "dual_branch_effective_accept_mode": str(effective_accept_mode),
                "dual_branch_target_moderate_underfloor_rate": float(target_underfloor_rate),
                "dual_branch_target_strong_underfloor_rate": float(target_underfloor_rate),
                "dual_branch_target_probe_rel_quality_drop": float(target_probe_rel_quality_drop),
                "dual_branch_target_probe_rel_density_drop": float(target_probe_rel_density_drop),
                "dual_branch_target_branch_jaccard_mean": float(target_branch_jaccard_mean),
                "dual_branch_target_moderate_max_branch_jaccard": float(target_moderate_max_branch_jaccard),
                "dual_branch_target_moderate_quality_drop_threshold": float(target_moderate_quality_drop_threshold),
                "dual_branch_target_moderate_density_drop_threshold": float(target_moderate_density_drop_threshold),
                "dual_branch_floor_router_disagreement_pass": bool(floor_router_disagreement_pass),
                "dual_branch_floor_router_quality_drop_pass": bool(floor_router_quality_drop_pass),
                "dual_branch_floor_router_density_drop_pass": bool(floor_router_density_drop_pass),
                "dual_branch_state_profile_id": str(floor_route_state),
                "dual_branch_state_profile_reason": str(floor_route_reason),
                "dual_branch_state_profile_alpha_final": float(gate.get("state_profile_alpha_final", gate.get("alpha", cfg.fixed_alpha))) if isinstance(gate, dict) else float(cfg.fixed_alpha),
                "dual_branch_state_profile_floor_package_enabled": bool(state_floor_pkg.get("enabled", False)),
                "dual_branch_state_profile_floor_package_reason": str(state_floor_pkg.get("reason", "")),
            }
        metadata = {
            "dual_branch_score": True,
            "dual_branch_alpha": float(gate.get("alpha", cfg.fixed_alpha)),
            "dual_branch_alpha_mode": str(cfg.alpha_mode),
            "dual_branch_search_profile": str(cfg.search_profile),
            "dual_branch_preserved_size": int(gate.get("preserved_size", 0)),
            "dual_branch_adapted_size": int(gate.get("adapted_size", 0)),
            "dual_branch_preserved_adapted_jaccard": float(gate.get("preserved_adapted_jaccard", 0.0)),
            "dual_branch_search_reason": search_reason,
            **floor_payload,
            "labels_used": False,
            "ground_truth_used": False,
            "metrics_computed": False,
        }
        records.append(
            CommunityPredictionRecord(
                graph_name=graph_name,
                query_index=int(qi),
                query_nodes_local=tuple(int(x) for x in qset),
                predicted_nodes_local=tuple(int(x) for x in pred),
                query_nodes_global=query_global,
                predicted_nodes_global=pred_global,
                metadata=metadata,
            )
        )
    if isinstance(dual_report, dict) and gate_rows:
        diagnostic_rows = list(gate_rows.values())

        def _diag_mean(key: str, default: float = 0.0) -> float:
            vals: list[float] = []
            for row in diagnostic_rows:
                try:
                    vals.append(float(row.get(key, default)))
                except (TypeError, ValueError):
                    pass
            return float(sum(vals) / max(1, len(vals)))

        def _diag_rate(key: str) -> float:
            return float(sum(bool(row.get(key, False)) for row in diagnostic_rows)) / float(
                max(1, len(diagnostic_rows))
            )

        profile_counts: dict[str, int] = {}
        score_action_counts_by_profile: dict[str, int] = {}
        score_mode_counts: dict[str, int] = {}
        for row in diagnostic_rows:
            profile = str(row.get("state_profile_id", "missing"))
            profile_counts[profile] = int(profile_counts.get(profile, 0)) + 1
            if bool(row.get("state_profile_score_action_applied", False)):
                score_action_counts_by_profile[profile] = int(score_action_counts_by_profile.get(profile, 0)) + 1
            score_mode = str(row.get("state_profile_score_mode", "missing"))
            score_mode_counts[score_mode] = int(score_mode_counts.get(score_mode, 0)) + 1
        dual_report["endpoint_repair_summary"] = {
            "query_count": int(len(diagnostic_rows)),
            "state_profile_counts": profile_counts,
            "state_profile_parent_id": str(diagnostic_rows[0].get("state_profile_parent_id", "missing")),
            "hierarchical_boundary_state_enabled": bool(
                diagnostic_rows[0].get("hierarchical_boundary_state_enabled", False)
            ),
            "hierarchical_boundary_state_eligible": bool(
                diagnostic_rows[0].get("hierarchical_boundary_state_eligible", False)
            ),
            "boundary_target_normalized_endpoint_marginal": _diag_mean(
                "boundary_target_normalized_endpoint_marginal", float("nan")
            ),
            "boundary_plateau_relative_quality_threshold": _diag_mean(
                "boundary_plateau_relative_quality_threshold", float(cfg.boundary_plateau_relative_quality_threshold)
            ),
            "boundary_plateau_threshold_margin": _diag_mean(
                "boundary_plateau_threshold_margin", float("nan")
            ),
            "boundary_query_plateau_rate": _diag_mean("boundary_query_plateau_rate", 0.0),
            "boundary_plateau_precision_action_rate": float(
                sum(str(row.get("state_profile_id", "")) == "boundary_plateau_precision_profile" for row in diagnostic_rows)
            ) / float(max(1, len(diagnostic_rows))),
            "state_profile_action_isolation_scope": str(
                diagnostic_rows[0].get("state_profile_action_isolation_scope", "moderate_only")
            ),
            "state_profile_score_action_rate": _diag_rate("state_profile_score_action_applied"),
            "state_profile_score_action_counts_by_profile": score_action_counts_by_profile,
            "state_profile_score_mode_counts": score_mode_counts,
            "target_overexpansion_rate": _diag_mean("target_overexpansion_rate"),
            "target_overexpansion_action_rate": _diag_rate("target_overexpansion_action"),
            "target_overexpansion_reason": str(diagnostic_rows[0].get("target_overexpansion_reason", "")),
            "mean_preserved_size": _diag_mean("preserved_size"),
            "mean_adapted_size": _diag_mean("adapted_size"),
            "mean_best_size": _diag_mean("best_size"),
            "mean_final_size": _diag_mean("final_size"),
            "mean_max_branch_size": _diag_mean("max_branch_size"),
            "mean_final_size_before_ceiling": _diag_mean("final_size_before_ceiling"),
            "mean_final_size_after_ceiling": _diag_mean("final_size_after_ceiling"),
            "mean_best_density": _diag_mean("best_density"),
            "mean_final_density": _diag_mean("final_density"),
            "mean_best_quality": _diag_mean("best_quality"),
            "mean_final_quality": _diag_mean("final_quality"),
            "mean_precision_over_coefficient": _diag_mean("precision_over_coefficient"),
            "mean_precision_size_penalty": _diag_mean("precision_size_penalty"),
            "mean_marginal_density_from_previous": _diag_mean("marginal_density_from_previous"),
            "mean_marginal_quality_from_previous": _diag_mean("marginal_quality_from_previous"),
            "mean_marginal_density_to_next": _diag_mean("marginal_density_to_next"),
            "mean_marginal_quality_to_next": _diag_mean("marginal_quality_to_next"),
            "ceiling_trigger_rate": _diag_rate("ceiling_triggered"),
            "ceiling_accept_rate": _diag_rate("ceiling_accepted"),
            "mean_ceiling_rel_density_drop": _diag_mean("ceiling_rel_density_drop"),
            "mean_ceiling_rel_quality_drop": _diag_mean("ceiling_rel_quality_drop"),
            "labels_used": False,
            "ground_truth_used": False,
        }
    return tuple(records)


def write_dual_branch_diagnostics(path: str | Path, report: Mapping[str, object]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in report.get("alpha_rows", []):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = out.with_suffix(".summary.json")
    payload = {k: v for k, v in report.items() if k != "alpha_rows"}
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
