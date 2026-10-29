"""Unified representation boundary for the final V575 inference path.

The journal-facing method treats the preserved and adapted embedding manifests
as complementary outputs of one target-calibrated representation stage.  This
module only names that boundary.  It performs no tensor operation, manifest
rewrite, random sampling, or target-side fitting, so the validated numerical
path remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetCalibratedRepresentation:
    """Pair of frozen embedding views consumed by community retrieval.

    ``reference_manifest`` is the fixed epoch-0/raw representation.
    ``adapted_manifest`` is produced by the source-trained denoiser with
    parameter-free target evidence fitted from unlabeled target embeddings at
    fixed inference.  The historical implementation calls the first view
    ``preserved``; the compatibility property below keeps that low-level name
    out of the high-level inference loop without changing the scorer.
    """

    reference_manifest: Path
    adapted_manifest: Path

    @classmethod
    def from_manifests(
        cls,
        *,
        reference_manifest: str | Path,
        adapted_manifest: str | Path,
    ) -> "TargetCalibratedRepresentation":
        return cls(
            reference_manifest=Path(reference_manifest),
            adapted_manifest=Path(adapted_manifest),
        )

    @property
    def preserved_manifest(self) -> Path:
        """Historical low-level alias used by dual-branch scoring."""

        return self.reference_manifest

    def as_legacy_manifest_kwargs(self) -> dict[str, Path]:
        """Return the exact argument names expected by the validated scorer."""

        return {
            "preserved_manifest_path": self.reference_manifest,
            "adapted_manifest_path": self.adapted_manifest,
        }
