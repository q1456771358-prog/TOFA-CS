"""Disk cache for senior-style V5 global hop tokens.

The cache stores immutable graph tokens only.  It never stores labels, ground
truth, metrics, checkpoint selection state, or query answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from .errors import V5DataLeakageError, V5ProtocolError


@dataclass(frozen=True)
class TokenCacheIdentity:
    graph_name: str
    graph_pt_path: str
    graph_pt_mtime_ns: int
    graph_pt_size_bytes: int
    num_nodes: int
    input_dim: int
    adjacency_nnz: int
    hops: int
    raw_feature_dim: int | None = None
    unified_input_dim: int | None = None
    feature_projection_mode: str = "identity"
    feature_projection_hash_seed: int = 1729
    cache_format_version: str = "v5-global-token-cache-v2-unified-input"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]


@dataclass(frozen=True)
class TokenCacheReport:
    enabled: bool
    hit: bool
    saved: bool
    cache_path: str | None
    identity_hash: str | None
    tensor_shape: tuple[int, ...] | None
    tensor_bytes: int
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_token_cache_identity(
    *,
    graph_name: str,
    graph_pt_path: str | Path,
    num_nodes: int,
    input_dim: int,
    adjacency_nnz: int,
    hops: int,
    raw_feature_dim: int | None = None,
    unified_input_dim: int | None = None,
    feature_projection_mode: str = "identity",
    feature_projection_hash_seed: int = 1729,
) -> TokenCacheIdentity:
    path = Path(graph_pt_path)
    stat = path.stat()
    return TokenCacheIdentity(
        graph_name=str(graph_name),
        graph_pt_path=str(path.resolve()),
        graph_pt_mtime_ns=int(stat.st_mtime_ns),
        graph_pt_size_bytes=int(stat.st_size),
        num_nodes=int(num_nodes),
        input_dim=int(input_dim),
        adjacency_nnz=int(adjacency_nnz),
        hops=int(hops),
        raw_feature_dim=int(raw_feature_dim) if raw_feature_dim is not None else int(input_dim),
        unified_input_dim=int(unified_input_dim) if unified_input_dim is not None else int(input_dim),
        feature_projection_mode=str(feature_projection_mode),
        feature_projection_hash_seed=int(feature_projection_hash_seed),
    )


def token_cache_path(cache_dir: str | Path, identity: TokenCacheIdentity) -> Path:
    return Path(cache_dir) / f"{identity.graph_name}_{identity.digest()}.pt"


def load_token_cache(path: str | Path, *, expected_identity: TokenCacheIdentity) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, Mapping):
        raise V5ProtocolError("token cache file must contain a mapping")
    if bool(obj.get("labels_used", False)) or bool(obj.get("ground_truth_used", False)) or bool(obj.get("metrics_used", False)):
        raise V5DataLeakageError("token cache metadata contains forbidden supervision flags")
    if obj.get("identity") != expected_identity.to_dict():
        raise V5ProtocolError("token cache identity mismatch")
    tokens = obj.get("global_token_seqs")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise V5ProtocolError("token cache missing [N,K,D] global_token_seqs")
    if int(tokens.shape[-1]) != int(expected_identity.input_dim):
        raise V5ProtocolError(
            f"token cache tensor dim={int(tokens.shape[-1])} does not match expected input_dim={int(expected_identity.input_dim)}"
        )
    return tokens.float().contiguous()


def save_token_cache(path: str | Path, *, identity: TokenCacheIdentity, global_token_seqs: torch.Tensor) -> Path:
    if global_token_seqs.ndim != 3:
        raise V5ProtocolError("global_token_seqs must be [N,K,D]")
    if int(global_token_seqs.shape[-1]) != int(identity.input_dim):
        raise V5ProtocolError(
            f"token cache input_dim mismatch: tensor dim={int(global_token_seqs.shape[-1])}, identity input_dim={int(identity.input_dim)}"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": identity.to_dict(),
            "identity_hash": identity.digest(),
            "global_token_seqs": global_token_seqs.detach().cpu().float().contiguous(),
            "labels_used": False,
            "ground_truth_used": False,
            "metrics_used": False,
        },
        out,
    )
    return out


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())
