"""Single source of truth for the original five-target OFA-CS thresholds."""
from __future__ import annotations

OFFICIAL_FIVE_TARGET_THRESHOLDS = {
    "wisconsin": {"official_f1": 0.3790, "official_jaccard": 0.2414},
    "cora": {"official_f1": 0.5734, "official_jaccard": 0.4092},
    "citeseer": {"official_f1": 0.4932, "official_jaccard": 0.2997},
    "cs": {"official_f1": 0.4415, "official_jaccard": 0.2813},
    "physics": {"official_f1": 0.5086, "official_jaccard": 0.3677},
}

# Best value in every target/metric cell of the senior paper's Table 1.
# Keep this separate from the OFA-CS-row thresholds above: the paper-row
# reproduction question and the per-cell SOTA question are different reports.
TABLE1_SOTA_FIVE_TARGET_THRESHOLDS = {
    "wisconsin": {"official_f1": 0.3790, "official_jaccard": 0.2414},
    "cora": {"official_f1": 0.5746, "official_jaccard": 0.4092},
    "citeseer": {"official_f1": 0.4932, "official_jaccard": 0.2997},
    "cs": {"official_f1": 0.4415, "official_jaccard": 0.2998},
    "physics": {"official_f1": 0.5331, "official_jaccard": 0.3677},
}


def canonical_target_name(name: str) -> str:
    value = str(name).strip().lower()
    return "cs" if value == "cocs" else value


def threshold_for_target(name: str) -> dict[str, float] | None:
    return OFFICIAL_FIVE_TARGET_THRESHOLDS.get(canonical_target_name(name))


def table1_sota_threshold_for_target(name: str) -> dict[str, float] | None:
    return TABLE1_SOTA_FIVE_TARGET_THRESHOLDS.get(canonical_target_name(name))
