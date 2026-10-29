"""V5 SSGDiff-OFA-CS utilities.

V5 starts from the senior OFA-CS backbone.  This package contains only the
non-multifrequency engineering shell, source-only diagnostics, query-boundary
loss, and DiffGDA-inspired guided diffusion utilities.  V4 multifrequency modules
are intentionally not imported here.
"""

from .errors import V5DataLeakageError, V5ProtocolError

__all__ = ["V5DataLeakageError", "V5ProtocolError"]
