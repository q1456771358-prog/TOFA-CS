"""Senior-style real data loading for V5.

This adapter is intentionally narrower than V4's dataset shell.  It only loads
the tensors required by the senior/V5 mainline and keeps ground-truth community
files outside the training and fixed-inference path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .errors import V5DataLeakageError, V5ProtocolError
from .runtime_profile import profile_event
from .senior_compat import SeniorQueryBatch, load_senior_query_file
from .token_cache import (
    TokenCacheReport,
    build_token_cache_identity,
    load_token_cache,
    save_token_cache,
    tensor_bytes,
    token_cache_path,
)


_FORBIDDEN_PATH_TOKENS = (
    ".gt",
    "groundtruth",
    "ground_truth",
    "answer",
    "answers",
    "metric",
    "metrics",
    "jaccard",
    "nmi",
    "ari",
    "f1",
)


@dataclass(frozen=True)
class SeniorDatasetFiles:
    dataset_name: str
    dataset_root: str
    graph_pt_path: str
    preprocessed_pt_path: str | None
    query_path: str | None
    gt_path_present_but_not_read: str | None
    graph_pt_exists: bool
    preprocessed_pt_exists: bool
    query_exists: bool
    gt_exists: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SeniorGraphData:
    graph_name: str
    features: torch.Tensor
    labels: torch.Tensor | None
    adjacency: torch.Tensor
    edge_index: torch.Tensor
    global_token_seqs: torch.Tensor
    local_token_seqs: torch.Tensor | None
    pos_samples: torch.Tensor | None
    neg_samples: torch.Tensor | None
    query_batch: SeniorQueryBatch | None
    global_node_ids: torch.Tensor
    files: SeniorDatasetFiles
    report: dict[str, object]

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.global_token_seqs.shape[-1])


@dataclass(frozen=True)
class SeniorGraphMetadata:
    graph_name: str
    split: str
    num_nodes: int
    raw_feature_dim: int
    feature_shape: tuple[int, int]
    adjacency_nnz: int
    graph_pt_path: str
    preprocessed_pt_path: str | None
    query_file_loaded: bool
    query_set_count: int
    gt_path_present_but_not_read: str | None
    labels_present_in_file: bool
    labels_loaded: bool = False
    gt_file_read: bool = False
    metrics_read: bool = False
    unified_input_dim: int | None = None
    feature_projection_mode: str = "identity"
    feature_projection_hash_seed: int = 1729

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise V5ProtocolError(f"dataset_name must be a simple name, got {dataset_name!r}")
    lowered = name.lower()
    if any(tok in lowered for tok in _FORBIDDEN_PATH_TOKENS):
        raise V5DataLeakageError(f"dataset_name contains forbidden feedback token: {dataset_name!r}")
    return name


def _reject_feedback_path(path: str | Path | None, *, role: str) -> None:
    if path is None:
        return
    lowered = str(path).lower()
    if any(tok in lowered for tok in _FORBIDDEN_PATH_TOKENS):
        raise V5DataLeakageError(f"{role} appears to reference feedback/metric artifact: {path}")


def discover_senior_dataset_files(
    dataset_root: str | Path,
    dataset_name: str,
    *,
    dims: int = 64,
    hops: int = 4,
    local_topk: int = 10,
    pos_count: int = 10,
    neg_count: int = 10,
) -> SeniorDatasetFiles:
    name = validate_dataset_name(dataset_name)
    root = Path(dataset_root)
    dataset_dir = root / name
    preprocessed_candidates = [
        root / f"{name}_d{dims}_khop{hops}_Nl{local_topk}_pos{pos_count}_neg{neg_count}.pt",
        dataset_dir / f"{name}_d{dims}_khop{hops}_Nl{local_topk}_pos{pos_count}_neg{neg_count}.pt",
        root / "constructed" / "my" / f"{name}_{dims}_{hops}_5_10.pt",
    ]
    graph_candidates = []
    if name in {"photo", "cs"}:
        graph_candidates.extend([root / f"{name}_dgl.pt", dataset_dir / f"{name}_dgl.pt"])
    graph_candidates.extend(
        [
            root / f"{name}_pyg.pt",
            dataset_dir / f"{name}_pyg.pt",
            root / f"{name}_dgl.pt",
            dataset_dir / f"{name}_dgl.pt",
            root / f"{name}.pt",
            dataset_dir / f"{name}.pt",
        ]
    )

    preprocessed = next((p for p in preprocessed_candidates if p.exists()), None)
    graph = next((p for p in graph_candidates if p.exists()), graph_candidates[0])
    query = dataset_dir / f"{name}.query"
    gt = dataset_dir / f"{name}.gt"
    return SeniorDatasetFiles(
        dataset_name=name,
        dataset_root=str(root),
        graph_pt_path=str(graph),
        preprocessed_pt_path=None if preprocessed is None else str(preprocessed),
        query_path=str(query) if query.exists() else None,
        gt_path_present_but_not_read=str(gt) if gt.exists() else None,
        graph_pt_exists=bool(graph.exists()),
        preprocessed_pt_exists=bool(preprocessed is not None),
        query_exists=bool(query.exists()),
        gt_exists=bool(gt.exists()),
    )


def _load_torch(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    _reject_feedback_path(path, role="torch_load_path")
    return torch.load(path, map_location=map_location, weights_only=False)


def _coerce_features(features: Any) -> torch.Tensor:
    out = torch.as_tensor(features).float().contiguous()
    if out.ndim != 2 or out.shape[0] <= 0 or out.shape[1] <= 0:
        raise V5ProtocolError(f"features must be [N,D], got {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise V5ProtocolError("features contain NaN/Inf")
    return out


def _feature_projection_report(
    *,
    raw_feature_dim: int,
    unified_input_dim: int | None,
    mode: str,
    hash_seed: int,
    applied: bool,
) -> dict[str, object]:
    final_dim = int(raw_feature_dim if unified_input_dim is None else unified_input_dim)
    return {
        "enabled": bool(unified_input_dim is not None),
        "applied": bool(applied),
        "mode": "identity" if not applied else str(mode),
        "raw_feature_dim": int(raw_feature_dim),
        "unified_input_dim": int(final_dim),
        "hash_seed": int(hash_seed),
        "projection_config_version": "v5-signed-feature-hash-v1",
        "pads_or_truncates": False,
        "learned_target_input_layer": False,
    }


def project_features_to_unified_dim(
    features: torch.Tensor,
    *,
    unified_input_dim: int | None,
    mode: str = "signed_feature_hash",
    hash_seed: int = 1729,
    chunk_feature_cols: int = 1024,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Project raw graph features into the shared V5 input space.

    This is intentionally deterministic and graph-agnostic.  It does not pad,
    truncate, learn a target-specific input layer, or use labels/metrics.  When
    ``unified_input_dim`` is provided and differs from the raw feature count,
    every raw feature column contributes to a signed hashing bucket in the
    unified space, so source training and target fixed export instantiate the
    same ``PretrainModelV5(input_dim=unified_input_dim, ...)``.
    """

    x = _coerce_features(features)
    raw_dim = int(x.shape[1])
    if unified_input_dim is None:
        return x, _feature_projection_report(
            raw_feature_dim=raw_dim,
            unified_input_dim=None,
            mode="identity",
            hash_seed=hash_seed,
            applied=False,
        )
    out_dim = int(unified_input_dim)
    if out_dim <= 0:
        raise V5ProtocolError("unified_input_dim must be positive")
    if str(mode) in {"identity", "none"}:
        if raw_dim != out_dim:
            raise V5ProtocolError(
                f"identity feature projection requires raw_feature_dim == unified_input_dim, got {raw_dim} vs {out_dim}"
            )
        return x, _feature_projection_report(
            raw_feature_dim=raw_dim,
            unified_input_dim=out_dim,
            mode="identity",
            hash_seed=hash_seed,
            applied=False,
        )
    if str(mode) != "signed_feature_hash":
        raise V5ProtocolError(f"unsupported feature_projection_mode={mode!r}")
    # Deterministic signed feature hashing.  All raw columns are consumed; none
    # are silently dropped or padded.  Column chunking avoids a second giant
    # [N, raw_dim] allocation beyond the tensor already loaded from disk.
    col_ids = torch.arange(raw_dim, dtype=torch.long)
    bucket = ((col_ids * 1103515245 + 12345 + int(hash_seed)) & 0x7FFFFFFF).remainder(out_dim).long()
    sign_bits = ((col_ids * 214013 + 2531011 + int(hash_seed)) & 0x7FFFFFFF).remainder(2)
    signs = torch.where(sign_bits == 0, torch.ones_like(col_ids, dtype=x.dtype), -torch.ones_like(col_ids, dtype=x.dtype))
    out = torch.zeros((int(x.shape[0]), out_dim), dtype=x.dtype)
    step = max(1, int(chunk_feature_cols))
    for start in range(0, raw_dim, step):
        end = min(raw_dim, start + step)
        vals = x[:, start:end] * signs[start:end].unsqueeze(0)
        out.index_add_(1, bucket[start:end], vals)
    return out.contiguous(), _feature_projection_report(
        raw_feature_dim=raw_dim,
        unified_input_dim=out_dim,
        mode="signed_feature_hash",
        hash_seed=hash_seed,
        applied=True,
    )


def adjacency_to_sparse_and_edge_index(adjacency: Any, *, num_nodes: int, allow_dense: bool = False) -> tuple[torch.Tensor, torch.Tensor, str]:
    if int(num_nodes) <= 0:
        raise V5ProtocolError("num_nodes must be positive")
    if isinstance(adjacency, torch.Tensor) and adjacency.is_sparse:
        adj = adjacency.coalesce().float()
        if tuple(adj.shape) != (int(num_nodes), int(num_nodes)):
            raise V5ProtocolError(f"sparse adjacency must be [N,N], got {tuple(adj.shape)}")
        return adj, adj.indices().long().contiguous(), "sparse_coo_square"

    tensor = torch.as_tensor(adjacency)
    square = tensor.ndim == 2 and tuple(tensor.shape) == (int(num_nodes), int(num_nodes))
    if tensor.ndim == 2 and int(tensor.shape[0]) == 2 and not square:
        edge_index = tensor.long().contiguous()
        return _edge_index_to_sparse(edge_index, num_nodes=num_nodes), edge_index, "edge_index_2_by_e"
    if tensor.ndim == 2 and int(tensor.shape[1]) == 2 and not square:
        edge_index = tensor.t().long().contiguous()
        return _edge_index_to_sparse(edge_index, num_nodes=num_nodes), edge_index, "edge_index_e_by_2"
    if square:
        if not allow_dense:
            raise V5ProtocolError("dense [N,N] adjacency is rejected by default")
        edge_index = torch.nonzero(tensor, as_tuple=False).t().long().contiguous()
        return _edge_index_to_sparse(edge_index, num_nodes=num_nodes), edge_index, "dense_square_allowed"
    raise V5ProtocolError(f"unsupported adjacency shape: {tuple(tensor.shape)}")


def _edge_index_to_sparse(edge_index: torch.Tensor, *, num_nodes: int) -> torch.Tensor:
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise V5ProtocolError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")
    if edge_index.numel() > 0:
        if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= int(num_nodes):
            raise V5ProtocolError("edge_index contains node id outside [0,N)")
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    return torch.sparse_coo_tensor(edge_index.long(), values, (int(num_nodes), int(num_nodes)), device=edge_index.device).coalesce()


def build_global_token_seqs(
    adjacency: torch.Tensor,
    features: torch.Tensor,
    *,
    hops: int,
    runtime_jsonl: str | Path | None = None,
    graph_name: str = "graph",
) -> torch.Tensor:
    """Senior-compatible hop trajectory: self features plus repeated sparse propagation."""

    if not adjacency.is_sparse:
        raise V5ProtocolError("build_global_token_seqs expects sparse adjacency")
    if int(hops) < 0:
        raise V5ProtocolError("hops must be non-negative")
    x = features.float()
    tokens = [x]
    adj = adjacency.coalesce().float()
    cur = x
    profile_event(
        runtime_jsonl,
        "global_token_build_start",
        graph_name=graph_name,
        num_nodes=int(features.shape[0]),
        input_dim=int(features.shape[1]),
        hops=int(hops),
        adjacency_nnz=int(adj._nnz()),
    )
    for hop in range(1, int(hops) + 1):
        profile_event(runtime_jsonl, "global_token_build_hop_start", graph_name=graph_name, hop=hop, hops=int(hops))
        cur = torch.sparse.mm(adj, cur)
        tokens.append(cur)
        profile_event(
            runtime_jsonl,
            "global_token_build_hop_end",
            graph_name=graph_name,
            hop=hop,
            hops=int(hops),
            token_shape=(int(cur.shape[0]), int(cur.shape[1])),
        )
    out = torch.stack(tokens, dim=1).contiguous()
    profile_event(runtime_jsonl, "global_token_build_end", graph_name=graph_name, global_token_shape=tuple(int(x) for x in out.shape))
    return out


def _unpack_raw_graph(obj: Any) -> tuple[Any, Any, Any | None, bool]:
    if isinstance(obj, Mapping):
        adjacency = obj.get("adj", obj.get("adjacency", obj.get("edge_index")))
        features = obj.get("feat", obj.get("features", obj.get("x")))
        labels = obj.get("label", obj.get("labels", obj.get("y")))
        if adjacency is None or features is None:
            raise V5ProtocolError("mapping graph file must contain adjacency/edge_index and features/x")
        return adjacency, features, labels, labels is not None
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if len(obj) < 2:
            raise V5ProtocolError("sequence graph file must contain adjacency and features")
        labels_present = len(obj) >= 3 and obj[2] is not None
        return obj[0], obj[1], obj[2] if labels_present else None, labels_present
    raise V5ProtocolError(f"unsupported graph object type: {type(obj)!r}")


def inspect_senior_graph_metadata(
    dataset_root: str | Path,
    dataset_name: str,
    *,
    split: str,
    dims: int = 64,
    hops: int = 4,
    local_topk: int = 10,
    pos_count: int = 10,
    neg_count: int = 10,
    load_queries: bool = True,
    map_location: str | torch.device = "cpu",
    unified_input_dim: int | None = None,
    feature_projection_mode: str = "signed_feature_hash",
    feature_projection_hash_seed: int = 1729,
) -> SeniorGraphMetadata:
    """Inspect graph shape metadata without building hop tokens or reading .gt."""

    name = validate_dataset_name(dataset_name)
    files = discover_senior_dataset_files(
        dataset_root,
        name,
        dims=dims,
        hops=hops,
        local_topk=local_topk,
        pos_count=pos_count,
        neg_count=neg_count,
    )
    path = files.graph_pt_path
    if path is None or not Path(path).exists():
        raise V5ProtocolError(f"no graph tensor found for metadata inspect: {name}")
    obj = _load_torch(path, map_location=map_location)
    adjacency_raw, features_raw, labels_raw, labels_present = _unpack_raw_graph(obj)
    features = _coerce_features(features_raw)
    feature_shape = (int(features.shape[0]), int(features.shape[1]))
    num_nodes = int(features.shape[0])
    raw_feature_dim = int(features.shape[1])
    adjacency, _, _ = adjacency_to_sparse_and_edge_index(adjacency_raw, num_nodes=num_nodes, allow_dense=True)
    query_count = 0
    query_loaded = False
    if load_queries and files.query_path is not None:
        qbatch = load_senior_query_file(files.query_path, num_nodes=num_nodes, graph_name=name)
        query_count = int(qbatch.q_count)
        query_loaded = True
    return SeniorGraphMetadata(
        graph_name=name,
        split=str(split),
        num_nodes=num_nodes,
        raw_feature_dim=raw_feature_dim,
        feature_shape=feature_shape,
        adjacency_nnz=int(adjacency.coalesce()._nnz()),
        graph_pt_path=files.graph_pt_path,
        preprocessed_pt_path=files.preprocessed_pt_path,
        query_file_loaded=query_loaded,
        query_set_count=query_count,
        gt_path_present_but_not_read=files.gt_path_present_but_not_read,
        labels_present_in_file=bool(labels_present),
        labels_loaded=False,
        gt_file_read=False,
        metrics_read=False,
        unified_input_dim=int(unified_input_dim) if unified_input_dim is not None else raw_feature_dim,
        feature_projection_mode="identity" if unified_input_dim is None or int(unified_input_dim) == raw_feature_dim else str(feature_projection_mode),
        feature_projection_hash_seed=int(feature_projection_hash_seed),
    )


def _unpack_preprocessed(obj: Any) -> tuple[Any, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool]:
    if not isinstance(obj, Mapping):
        raise V5ProtocolError("preprocessed senior tensor must be a mapping")
    adjacency = obj.get("adj", obj.get("adjacency", obj.get("edge_index")))
    features = obj.get("feat", obj.get("features", obj.get("x")))
    labels = obj.get("label", obj.get("labels", obj.get("y")))
    global_tokens = obj.get("global_token_seqs")
    local_tokens = obj.get("local_token_seqs")
    pos_samples = obj.get("pos_samples", obj.get("positive_samples", obj.get("pos_sample", obj.get("pos"))))
    neg_samples = obj.get("neg_samples", obj.get("negative_samples", obj.get("neg_sample", obj.get("neg"))))
    if adjacency is None or features is None or global_tokens is None:
        raise V5ProtocolError("preprocessed graph must contain adj, feat, and global_token_seqs")
    return (
        adjacency,
        _coerce_features(features),
        labels,
        local_tokens,
        torch.as_tensor(global_tokens).float().contiguous(),
        None if pos_samples is None else torch.as_tensor(pos_samples).long().contiguous(),
        None if neg_samples is None else torch.as_tensor(neg_samples).long().contiguous(),
        labels is not None,
    )


def load_senior_graph_data(
    dataset_root: str | Path,
    dataset_name: str,
    *,
    split: str,
    purpose: str,
    dims: int = 64,
    hops: int = 4,
    local_topk: int = 10,
    pos_count: int = 10,
    neg_count: int = 10,
    require_labels: bool = False,
    load_queries: bool = True,
    allow_dense_adjacency: bool = False,
    prefer_preprocessed: bool = True,
    senior_cache_mode: str = "auto",
    map_location: str | torch.device = "cpu",
    token_cache_dir: str | Path | None = None,
    reuse_token_cache: bool = False,
    save_token_cache_enabled: bool = True,
    runtime_jsonl: str | Path | None = None,
    unified_input_dim: int | None = None,
    feature_projection_mode: str = "signed_feature_hash",
    feature_projection_hash_seed: int = 1729,
) -> SeniorGraphData:
    split_l = str(split).lower()
    purpose_l = str(purpose).lower()
    if "unseen" in split_l and require_labels:
        raise V5DataLeakageError("unseen graph loading cannot require labels")
    files = discover_senior_dataset_files(
        dataset_root,
        dataset_name,
        dims=dims,
        hops=hops,
        local_topk=local_topk,
        pos_count=pos_count,
        neg_count=neg_count,
    )
    senior_cache_mode_l = str(senior_cache_mode or "auto").lower()
    if senior_cache_mode_l not in {"auto", "required", "disabled"}:
        raise V5ProtocolError(f"unsupported senior_cache_mode={senior_cache_mode!r}")
    if senior_cache_mode_l == "disabled":
        prefer_preprocessed = False
    if senior_cache_mode_l == "required":
        prefer_preprocessed = True
        strict_name = f"{validate_dataset_name(dataset_name)}_d{dims}_khop{hops}_Nl{local_topk}_pos{pos_count}_neg{neg_count}.pt"
        if files.preprocessed_pt_path is None:
            raise V5ProtocolError(
                f"senior_cache_mode=required but senior preprocessed cache is missing for {dataset_name}: "
                f"expected {strict_name}"
            )
        if Path(files.preprocessed_pt_path).name != strict_name:
            raise V5ProtocolError(
                "senior_cache_mode=required refuses legacy constructed/my cache; "
                f"expected exact strict cache file {strict_name}, got {files.preprocessed_pt_path}"
            )
        if str(feature_projection_mode) != "identity":
            raise V5ProtocolError(
                "senior_cache_mode=required preserves senior cached token space and therefore requires "
                "--feature-projection-mode identity"
            )
    if not files.graph_pt_exists and not files.preprocessed_pt_exists:
        raise V5ProtocolError(f"no senior graph tensor found for {dataset_name} under {dataset_root}")

    loaded_preprocessed = False
    local_tokens = None
    pos_samples = None
    neg_samples = None
    preprocessed_error: str | None = None
    raw_feature_dim: int | None = None
    feature_projection_report: dict[str, object] | None = None
    token_cache_report = TokenCacheReport(
        enabled=bool(token_cache_dir),
        hit=False,
        saved=False,
        cache_path=None,
        identity_hash=None,
        tensor_shape=None,
        tensor_bytes=0,
    )
    if prefer_preprocessed and files.preprocessed_pt_path is not None:
        try:
            adjacency_raw, features, labels_raw, local_tokens, global_tokens, pos_samples, neg_samples, labels_present = _unpack_preprocessed(
                _load_torch(files.preprocessed_pt_path, map_location=map_location)
            )
            raw_feature_dim = int(features.shape[1])
            if unified_input_dim is not None and str(feature_projection_mode) == "signed_feature_hash":
                # Legacy preprocessed tensors do not carry the V5 unified feature
                # hashing identity.  Rebuild from the raw graph so source and
                # target use exactly the same graph-agnostic projection function.
                if files.graph_pt_exists:
                    preprocessed_error = (
                        "legacy preprocessed tokens skipped because unified signed_feature_hash projection is required; "
                        "rebuilding hop tokens from raw graph"
                    )
                    loaded_preprocessed = False
                else:
                    raise V5ProtocolError(
                        "unified signed_feature_hash projection requires raw graph features; no raw graph tensor is available"
                    )
            elif unified_input_dim is not None and int(global_tokens.shape[-1]) != int(unified_input_dim):
                # A raw-dim preprocessed token cache cannot be reused for cross-graph
                # fixed export.  Fall back to raw graph loading so the shared feature
                # projection can be applied before hop-token construction.
                if files.graph_pt_exists:
                    preprocessed_error = (
                        f"preprocessed global_token dim={int(global_tokens.shape[-1])} does not match "
                        f"unified_input_dim={int(unified_input_dim)}; rebuilding from raw graph with shared projection"
                    )
                    loaded_preprocessed = False
                else:
                    raise V5ProtocolError(
                        f"preprocessed graph dim={int(global_tokens.shape[-1])} does not match unified_input_dim={int(unified_input_dim)} "
                        "and no raw graph tensor is available for projected rebuild"
                    )
            else:
                if unified_input_dim is not None and int(features.shape[1]) != int(unified_input_dim):
                    features, feature_projection_report = project_features_to_unified_dim(
                        features,
                        unified_input_dim=int(unified_input_dim),
                        mode=feature_projection_mode,
                        hash_seed=int(feature_projection_hash_seed),
                    )
                else:
                    feature_projection_report = _feature_projection_report(
                        raw_feature_dim=int(features.shape[1]),
                        unified_input_dim=int(unified_input_dim) if unified_input_dim is not None else None,
                        mode="identity",
                        hash_seed=int(feature_projection_hash_seed),
                        applied=False,
                    )
                loaded_preprocessed = True
        except V5ProtocolError as exc:
            preprocessed_error = str(exc)
            loaded_preprocessed = False
    if senior_cache_mode_l == "required" and not loaded_preprocessed:
        raise V5ProtocolError(
            f"senior_cache_mode=required failed to load senior preprocessed cache for {dataset_name}: "
            f"{preprocessed_error or files.preprocessed_pt_path}"
        )
    if not loaded_preprocessed:
        if not files.graph_pt_exists:
            raise V5ProtocolError(f"raw graph tensor not found: {files.graph_pt_path}")
        adjacency_raw, features_raw, labels_raw, labels_present = _unpack_raw_graph(_load_torch(files.graph_pt_path, map_location=map_location))
        raw_features = _coerce_features(features_raw)
        raw_feature_dim = int(raw_features.shape[1])
        features, feature_projection_report = project_features_to_unified_dim(
            raw_features,
            unified_input_dim=int(unified_input_dim) if unified_input_dim is not None else None,
            mode=feature_projection_mode,
            hash_seed=int(feature_projection_hash_seed),
        )
        adjacency_tmp, _, _ = adjacency_to_sparse_and_edge_index(adjacency_raw, num_nodes=int(features.shape[0]), allow_dense=allow_dense_adjacency)
        identity = None
        cache_path = None
        if token_cache_dir is not None:
            identity = build_token_cache_identity(
                graph_name=validate_dataset_name(dataset_name),
                graph_pt_path=files.graph_pt_path,
                num_nodes=int(features.shape[0]),
                input_dim=int(features.shape[1]),
                adjacency_nnz=int(adjacency_tmp.coalesce()._nnz()),
                hops=int(hops),
                raw_feature_dim=int(raw_feature_dim if raw_feature_dim is not None else features.shape[1]),
                unified_input_dim=int(features.shape[1]),
                feature_projection_mode=str((feature_projection_report or {}).get("mode", "identity")),
                feature_projection_hash_seed=int(feature_projection_hash_seed),
            )
            cache_path = token_cache_path(token_cache_dir, identity)
        if cache_path is not None and reuse_token_cache and cache_path.exists() and identity is not None:
            global_tokens = load_token_cache(cache_path, expected_identity=identity)
            token_cache_report = TokenCacheReport(
                enabled=True,
                hit=True,
                saved=False,
                cache_path=str(cache_path),
                identity_hash=identity.digest(),
                tensor_shape=tuple(int(x) for x in global_tokens.shape),
                tensor_bytes=tensor_bytes(global_tokens),
            )
        else:
            global_tokens = build_global_token_seqs(
                adjacency_tmp,
                features,
                hops=hops,
                runtime_jsonl=runtime_jsonl,
                graph_name=validate_dataset_name(dataset_name),
            )
            if cache_path is not None and save_token_cache_enabled and identity is not None:
                save_token_cache(cache_path, identity=identity, global_token_seqs=global_tokens)
                token_cache_report = TokenCacheReport(
                    enabled=True,
                    hit=False,
                    saved=True,
                    cache_path=str(cache_path),
                    identity_hash=identity.digest(),
                    tensor_shape=tuple(int(x) for x in global_tokens.shape),
                    tensor_bytes=tensor_bytes(global_tokens),
                )

    adjacency, edge_index, adjacency_kind = adjacency_to_sparse_and_edge_index(
        adjacency_raw,
        num_nodes=int(features.shape[0]),
        allow_dense=allow_dense_adjacency,
    )
    if feature_projection_report is None:
        feature_projection_report = _feature_projection_report(
            raw_feature_dim=int(raw_feature_dim if raw_feature_dim is not None else features.shape[1]),
            unified_input_dim=int(unified_input_dim) if unified_input_dim is not None else None,
            mode="identity",
            hash_seed=int(feature_projection_hash_seed),
            applied=False,
        )
    if global_tokens.ndim != 3 or global_tokens.shape[0] != features.shape[0]:
        raise V5ProtocolError(f"global_token_seqs must be [N,K,D], got {tuple(global_tokens.shape)}")
    if int(global_tokens.shape[-1]) != int(features.shape[1]):
        raise V5ProtocolError(
            f"global_token input dim={int(global_tokens.shape[-1])} does not match feature dim={int(features.shape[1])}; "
            "refusing mixed raw/projected token space"
        )
    if unified_input_dim is not None and int(global_tokens.shape[-1]) != int(unified_input_dim):
        raise V5ProtocolError(
            f"global_token dim={int(global_tokens.shape[-1])} does not match unified_input_dim={int(unified_input_dim)}"
        )
    labels = None
    labels_loaded = False
    if require_labels:
        if not labels_present:
            raise V5ProtocolError("source loading requires labels but graph file has none")
        labels = torch.as_tensor(labels_raw).long().flatten().cpu()
        if labels.numel() != features.shape[0]:
            raise V5ProtocolError("labels/features num_nodes mismatch")
        labels_loaded = True

    query_batch = None
    if load_queries and files.query_path is not None:
        query_batch = load_senior_query_file(files.query_path, num_nodes=int(features.shape[0]), graph_name=validate_dataset_name(dataset_name))

    report = {
        "graph_name": validate_dataset_name(dataset_name),
        "split": split,
        "purpose": purpose,
        "senior_cache_mode": senior_cache_mode_l,
        "loaded_preprocessed": loaded_preprocessed,
        "preprocessed_fallback_error": preprocessed_error,
        "graph_pt_path": files.graph_pt_path,
        "preprocessed_pt_path": files.preprocessed_pt_path,
        "query_path": files.query_path,
        "gt_path_present_but_not_read": files.gt_path_present_but_not_read,
        "labels_present_in_file": bool(labels_present),
        "labels_loaded": bool(labels_loaded),
        "query_file_loaded": bool(query_batch is not None),
        "gt_file_read": False,
        "metrics_read": False,
        "adjacency_input_kind": adjacency_kind,
        "adjacency_nnz": int(adjacency.coalesce()._nnz()),
        "raw_feature_dim": int(raw_feature_dim if raw_feature_dim is not None else features.shape[1]),
        "feature_shape": (int(features.shape[0]), int(features.shape[1])),
        "unified_input_dim": int(global_tokens.shape[-1]),
        "feature_projection": feature_projection_report,
        "global_token_shape": tuple(int(x) for x in global_tokens.shape),
        "local_token_shape": None if local_tokens is None else tuple(int(x) for x in torch.as_tensor(local_tokens).shape),
        "pos_samples_shape": None if pos_samples is None else tuple(int(x) for x in torch.as_tensor(pos_samples).shape),
        "neg_samples_shape": None if neg_samples is None else tuple(int(x) for x in torch.as_tensor(neg_samples).shape),
        "query_set_count": 0 if query_batch is None else int(query_batch.q_count),
        "query_duplicate_seed_count": 0 if query_batch is None else int(query_batch.duplicate_seed_count),
        "token_cache": token_cache_report.to_dict(),
    }
    return SeniorGraphData(
        graph_name=validate_dataset_name(dataset_name),
        features=features,
        labels=labels,
        adjacency=adjacency,
        edge_index=edge_index,
        global_token_seqs=global_tokens,
        local_token_seqs=None if local_tokens is None else torch.as_tensor(local_tokens).float().contiguous(),
        pos_samples=None if pos_samples is None else torch.as_tensor(pos_samples).long().contiguous(),
        neg_samples=None if neg_samples is None else torch.as_tensor(neg_samples).long().contiguous(),
        query_batch=query_batch,
        global_node_ids=torch.arange(features.shape[0], dtype=torch.long),
        files=files,
        report=report,
    )
