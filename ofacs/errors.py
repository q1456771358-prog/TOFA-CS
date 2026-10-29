"""V5 protocol and leakage errors.

These exceptions are deliberately small and dependency-free so every V5 module can
reject unsafe paths before any expensive tensor work starts.
"""

from __future__ import annotations


class V5ProtocolError(RuntimeError):
    """Raised when an input violates the V5 OFA-CS protocol."""


class V5DataLeakageError(V5ProtocolError):
    """Raised when unseen labels/ground truth/metrics could enter training or inference."""
