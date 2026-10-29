"""Shape-derived resource preflight for V5 SSGDiff.

These estimates are engineering guards only.  They do not run training,
inference, search, metrics, or checkpoint selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .errors import V5ProtocolError


FLOAT32_BYTES = 4
INT64_BYTES = 8


@dataclass(frozen=True)
class V5ResourcePreflightConfig:
    input_dim: int
    hidden_dim: int
    hops: int = 4
    query_chunk_size: int = 16
    max_query_chunk_nodes: int = 8192
    node_chunk_size: int = 4096
    score_query_chunk_size: int = 32
    max_in_memory_tensor_bytes: int = 2_000_000_000
    dtype_bytes: int = FLOAT32_BYTES

    def validate(self) -> "V5ResourcePreflightConfig":
        for name in (
            "input_dim",
            "hidden_dim",
            "hops",
            "query_chunk_size",
            "max_query_chunk_nodes",
            "node_chunk_size",
            "score_query_chunk_size",
            "max_in_memory_tensor_bytes",
            "dtype_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise V5ProtocolError(f"{name} must be positive")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5GraphResourceEstimate:
    graph_name: str
    num_nodes: int
    raw_feature_dim: int
    adjacency_nnz: int
    global_token_bytes: int
    raw_embedding_bytes: int
    refined_embedding_bytes: int
    query_chunk_token_bytes: int
    query_chunk_embedding_bytes: int
    score_chunk_elements: int
    score_chunk_bytes: int
    dense_adjacency_bytes: int
    dense_adjacency_forbidden: bool
    full_graph_autograd_forbidden: bool
    requires_token_cache_strategy: bool
    requires_embedding_cache_strategy: bool
    requires_score_chunk_strategy: bool
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_computed: bool = False
    unseen_metrics_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V5ResourcePreflightReport:
    config: V5ResourcePreflightConfig
    graph_estimates: tuple[V5GraphResourceEstimate, ...]
    graph_count: int
    requires_memory_strategy_graphs: tuple[str, ...]
    max_global_token_bytes: int
    max_refined_embedding_bytes: int
    max_score_chunk_bytes: int
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_computed: bool = False
    unseen_metrics_used: bool = False
    formal_experiment: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["graph_estimates"] = [e.to_dict() for e in self.graph_estimates]
        return payload


@dataclass(frozen=True)
class V5ResourcePreflightGateReport:
    enabled: bool
    require_pass: bool
    passed: bool
    failing_graphs: tuple[str, ...]
    graph_count: int
    max_global_token_bytes: int
    max_refined_embedding_bytes: int
    max_score_chunk_bytes: int
    token_cache_strategy_enabled: bool = False
    embedding_cache_strategy_enabled: bool = False
    score_chunk_strategy_enabled: bool = False
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_computed: bool = False
    unseen_metrics_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_graph_resource(
    *,
    graph_name: str,
    num_nodes: int,
    raw_feature_dim: int,
    adjacency_nnz: int,
    config: V5ResourcePreflightConfig,
) -> V5GraphResourceEstimate:
    cfg = config.validate()
    n = int(num_nodes)
    raw_dim = int(raw_feature_dim)
    nnz = int(adjacency_nnz)
    if n <= 0 or raw_dim <= 0 or nnz < 0:
        raise V5ProtocolError("num_nodes/raw_feature_dim must be positive and adjacency_nnz non-negative")
    k = int(cfg.hops) + 1
    embedding_dim = 2 * int(cfg.hidden_dim)
    global_token_bytes = n * k * int(cfg.input_dim) * int(cfg.dtype_bytes)
    raw_embedding_bytes = n * embedding_dim * int(cfg.dtype_bytes)
    refined_embedding_bytes = n * embedding_dim * int(cfg.dtype_bytes)
    query_chunk_token_bytes = int(cfg.max_query_chunk_nodes) * k * int(cfg.input_dim) * int(cfg.dtype_bytes)
    query_chunk_embedding_bytes = int(cfg.max_query_chunk_nodes) * embedding_dim * int(cfg.dtype_bytes)
    score_elements = int(cfg.score_query_chunk_size) * int(cfg.node_chunk_size)
    score_chunk_bytes = score_elements * int(cfg.dtype_bytes)
    dense_adj_bytes = n * n * int(cfg.dtype_bytes)
    limit = int(cfg.max_in_memory_tensor_bytes)
    return V5GraphResourceEstimate(
        graph_name=str(graph_name),
        num_nodes=n,
        raw_feature_dim=raw_dim,
        adjacency_nnz=nnz,
        global_token_bytes=global_token_bytes,
        raw_embedding_bytes=raw_embedding_bytes,
        refined_embedding_bytes=refined_embedding_bytes,
        query_chunk_token_bytes=query_chunk_token_bytes,
        query_chunk_embedding_bytes=query_chunk_embedding_bytes,
        score_chunk_elements=score_elements,
        score_chunk_bytes=score_chunk_bytes,
        dense_adjacency_bytes=dense_adj_bytes,
        dense_adjacency_forbidden=True,
        full_graph_autograd_forbidden=True,
        requires_token_cache_strategy=global_token_bytes > limit,
        requires_embedding_cache_strategy=refined_embedding_bytes > limit,
        requires_score_chunk_strategy=score_chunk_bytes > limit,
    )


def build_resource_preflight_report(
    graph_specs: Sequence[Mapping[str, object]],
    *,
    config: V5ResourcePreflightConfig,
) -> V5ResourcePreflightReport:
    cfg = config.validate()
    estimates = tuple(
        estimate_graph_resource(
            graph_name=str(spec["graph_name"]),
            num_nodes=int(spec["num_nodes"]),
            raw_feature_dim=int(spec["raw_feature_dim"]),
            adjacency_nnz=int(spec["adjacency_nnz"]),
            config=cfg,
        )
        for spec in graph_specs
    )
    requiring = tuple(
        e.graph_name
        for e in estimates
        if e.requires_token_cache_strategy or e.requires_embedding_cache_strategy or e.requires_score_chunk_strategy
    )
    return V5ResourcePreflightReport(
        config=cfg,
        graph_estimates=estimates,
        graph_count=len(estimates),
        requires_memory_strategy_graphs=requiring,
        max_global_token_bytes=max((e.global_token_bytes for e in estimates), default=0),
        max_refined_embedding_bytes=max((e.refined_embedding_bytes for e in estimates), default=0),
        max_score_chunk_bytes=max((e.score_chunk_bytes for e in estimates), default=0),
    )


def graph_specs_from_manifest(path: str | Path, *, graph_names: Sequence[str] | None = None) -> tuple[dict[str, object], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graphs = payload.get("graphs")
    if not isinstance(graphs, Mapping):
        raise V5ProtocolError("graph manifest must contain a graphs mapping")
    selected = tuple(graph_names or graphs.keys())
    specs: list[dict[str, object]] = []
    for graph_name in selected:
        if graph_name not in graphs:
            raise V5ProtocolError(f"graph {graph_name!r} is missing from manifest")
        entry = graphs[graph_name]
        if not isinstance(entry, Mapping):
            raise V5ProtocolError(f"graph manifest entry must be a mapping: {graph_name}")
        raw_dim = entry.get("raw_feature_dim")
        if raw_dim is None and "feature_shape" in entry:
            raw_dim = entry["feature_shape"][1]
        nnz = entry.get("adjacency_nnz", entry.get("directed_edge_entries"))
        if raw_dim is None or nnz is None:
            raise V5ProtocolError(f"manifest entry missing raw_feature_dim/adjacency_nnz: {graph_name}")
        specs.append(
            {
                "graph_name": str(graph_name),
                "num_nodes": int(entry["num_nodes"]),
                "raw_feature_dim": int(raw_dim),
                "adjacency_nnz": int(nnz),
            }
        )
    return tuple(specs)


def load_resource_preflight_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise V5ProtocolError("resource preflight report must be a JSON object")
    return dict(payload)


def build_resource_preflight_gate_report(
    payload: Mapping[str, object],
    *,
    require_pass: bool,
    token_cache_strategy_enabled: bool = False,
    embedding_cache_strategy_enabled: bool = False,
    score_chunk_strategy_enabled: bool = False,
) -> V5ResourcePreflightGateReport:
    """Build a gate report from preflight estimates.

    ``requires_memory_strategy_graphs`` is not itself a failure in formal V5:
    large OFA-CS graphs are expected to require token caching, embedding shards,
    and chunked scoring.  The gate fails only when a required strategy is not
    configured by the caller.  Existing direct calls that do not provide strategy
    flags remain strict, which preserves the old safety behavior for ad-hoc runs.
    """

    failing: list[str] = []
    estimates = payload.get("graph_estimates", ()) or ()
    if isinstance(estimates, list):
        for entry in estimates:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("graph_name", "unknown"))
            reasons: list[str] = []
            if bool(entry.get("requires_token_cache_strategy", False)) and not bool(token_cache_strategy_enabled):
                reasons.append("token_cache")
            if bool(entry.get("requires_embedding_cache_strategy", False)) and not bool(embedding_cache_strategy_enabled):
                reasons.append("embedding_shards")
            if bool(entry.get("requires_score_chunk_strategy", False)) and not bool(score_chunk_strategy_enabled):
                reasons.append("score_chunking")
            if reasons:
                failing.append(f"{name}:{'+'.join(reasons)}")
    else:
        failing = [str(name) for name in payload.get("requires_memory_strategy_graphs", ()) or ()]
    return V5ResourcePreflightGateReport(
        enabled=True,
        require_pass=bool(require_pass),
        passed=len(failing) == 0,
        failing_graphs=tuple(failing),
        graph_count=int(payload.get("graph_count", 0)),
        max_global_token_bytes=int(payload.get("max_global_token_bytes", 0)),
        max_refined_embedding_bytes=int(payload.get("max_refined_embedding_bytes", 0)),
        max_score_chunk_bytes=int(payload.get("max_score_chunk_bytes", 0)),
        token_cache_strategy_enabled=bool(token_cache_strategy_enabled),
        embedding_cache_strategy_enabled=bool(embedding_cache_strategy_enabled),
        score_chunk_strategy_enabled=bool(score_chunk_strategy_enabled),
    )


def assert_resource_preflight_gate_passed(report: V5ResourcePreflightGateReport) -> None:
    if report.require_pass and not report.passed:
        raise V5ProtocolError(
            "resource preflight requires memory/cache strategy before running: "
            f"{list(report.failing_graphs)}"
        )


__all__ = [
    "V5GraphResourceEstimate",
    "V5ResourcePreflightConfig",
    "V5ResourcePreflightGateReport",
    "V5ResourcePreflightReport",
    "assert_resource_preflight_gate_passed",
    "build_resource_preflight_gate_report",
    "build_resource_preflight_report",
    "estimate_graph_resource",
    "graph_specs_from_manifest",
    "load_resource_preflight_payload",
]
