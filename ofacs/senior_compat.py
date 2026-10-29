"""Senior OFA-CS query/search protocol adapters for V5.

This module keeps the senior semantics explicit:
- query files contain graph-local query seed sets only;
- query embedding is the mean of all seed-node embeddings;
- GlobalSearch always starts from all query seeds;
- .gt files and metrics are rejected by protocol-facing loaders.

The implementation is adapted from the V4 senior compatibility shell, but all
multifrequency scoring hooks have been removed.  Scores are supplied by the
senior/V5 cosine scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .errors import V5DataLeakageError, V5ProtocolError

_VALID_ID_SPACES = {"local", "global"}
_FORBIDDEN_GT_SUFFIXES = {".gt", ".groundtruth", ".ground_truth", ".answer", ".answers"}
_FORBIDDEN_METRIC_SUFFIXES = {".metric", ".metrics", ".f1", ".jaccard", ".nmi", ".ari"}


@dataclass(frozen=True)
class SeniorQueryBatch:
    """Query-only view of the senior ``*.query`` protocol.

    Attributes:
        graph_name: Dataset/graph name.
        num_nodes: Number of graph-local nodes.
        query_node_sets: tuple of query seed sets, each using local node IDs.
        source_path: Optional path of the source query file.
    """

    graph_name: str
    num_nodes: int
    query_node_sets: tuple[tuple[int, ...], ...]
    id_space: str = "local"
    source_path: str | None = None
    loaded_ground_truth: bool = False
    metrics_loaded: bool = False
    duplicate_seed_count: int = 0

    def __post_init__(self) -> None:
        if self.id_space not in _VALID_ID_SPACES:
            raise V5ProtocolError(f"id_space must be one of {_VALID_ID_SPACES}, got {self.id_space!r}")
        if int(self.num_nodes) <= 0:
            raise V5ProtocolError("num_nodes must be positive")
        if self.loaded_ground_truth or self.metrics_loaded:
            raise V5DataLeakageError("SeniorQueryBatch cannot contain ground truth or metrics")
        if not self.query_node_sets:
            raise V5ProtocolError("at least one senior query set is required")

        normalized: list[tuple[int, ...]] = []
        dup_count = 0
        for qset in self.query_node_sets:
            raw = tuple(int(x) for x in qset)
            if not raw:
                raise V5ProtocolError("empty query seed set is not allowed")
            dedup = tuple(dict.fromkeys(raw))
            dup_count += len(raw) - len(dedup)
            if min(dedup) < 0 or max(dedup) >= int(self.num_nodes):
                raise V5ProtocolError(f"query seed outside [0,{self.num_nodes}): {dedup}")
            normalized.append(dedup)
        object.__setattr__(self, "query_node_sets", tuple(normalized))
        object.__setattr__(self, "duplicate_seed_count", int(dup_count))

    @property
    def q_count(self) -> int:
        return len(self.query_node_sets)

    @property
    def max_query_size(self) -> int:
        return max(len(q) for q in self.query_node_sets)

    @property
    def representative_query_nodes(self) -> torch.Tensor:
        """First seed per query; only for compatibility with single-id callers."""

        return torch.tensor([q[0] for q in self.query_node_sets], dtype=torch.long)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _reject_ground_truth_or_metric_path(path: Path) -> None:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in _FORBIDDEN_GT_SUFFIXES or name.endswith(".gt") or ".gt" in name:
        raise V5DataLeakageError(f"ground-truth path is forbidden here: {path}")
    if suffix in _FORBIDDEN_METRIC_SUFFIXES or any(x in name for x in ("f1", "jaccard", "metric", "metrics")):
        raise V5DataLeakageError(f"metric path is forbidden here: {path}")


def parse_query_lines(lines: Iterable[str]) -> tuple[tuple[int, ...], ...]:
    qsets: list[tuple[int, ...]] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            nodes = tuple(int(x) for x in stripped.replace(",", " ").split())
        except ValueError as exc:
            raise V5ProtocolError(f"invalid query line {line_no}: {line!r}") from exc
        if not nodes:
            raise V5ProtocolError(f"empty query line {line_no}")
        qsets.append(nodes)
    if not qsets:
        raise V5ProtocolError("query file contains no query sets")
    return tuple(qsets)


def load_senior_query_file(
    path: str | Path,
    *,
    num_nodes: int,
    graph_name: str | None = None,
    id_space: str = "local",
    global_node_ids: Sequence[int] | torch.Tensor | None = None,
) -> SeniorQueryBatch:
    """Load a senior ``*.query`` file without reading ``*.gt`` or metrics."""

    query_path = Path(path)
    _reject_ground_truth_or_metric_path(query_path)
    if id_space not in _VALID_ID_SPACES:
        raise V5ProtocolError(f"id_space must be one of {_VALID_ID_SPACES}, got {id_space!r}")
    raw_sets = parse_query_lines(query_path.read_text(encoding="utf-8").splitlines())
    if id_space == "global":
        if global_node_ids is None:
            raise V5ProtocolError("global id_space requires global_node_ids")
        raw_sets = global_sets_to_local(raw_sets, global_node_ids)
    return SeniorQueryBatch(
        graph_name=graph_name or query_path.stem,
        num_nodes=int(num_nodes),
        query_node_sets=raw_sets,
        id_space="local",
        source_path=str(query_path),
        loaded_ground_truth=False,
        metrics_loaded=False,
    )


def senior_query_batch_from_node_sets(
    query_node_sets: Sequence[Sequence[int] | torch.Tensor],
    *,
    num_nodes: int,
    graph_name: str = "unknown",
) -> SeniorQueryBatch:
    qsets: list[tuple[int, ...]] = []
    for qset in query_node_sets:
        if isinstance(qset, torch.Tensor):
            qsets.append(tuple(int(x) for x in qset.detach().cpu().flatten().tolist()))
        else:
            qsets.append(tuple(int(x) for x in qset))
    return SeniorQueryBatch(graph_name=graph_name, num_nodes=int(num_nodes), query_node_sets=tuple(qsets))


def global_sets_to_local(
    query_sets: Sequence[Sequence[int]],
    global_node_ids: Sequence[int] | torch.Tensor,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(global_node_ids, torch.Tensor):
        gids = [int(x) for x in global_node_ids.detach().cpu().flatten().tolist()]
    else:
        gids = [int(x) for x in global_node_ids]
    mapping = {gid: i for i, gid in enumerate(gids)}
    out: list[tuple[int, ...]] = []
    for qset in query_sets:
        local: list[int] = []
        for gid in qset:
            if int(gid) not in mapping:
                raise V5ProtocolError(f"global query node {gid} not found in global_node_ids")
            local.append(mapping[int(gid)])
        out.append(tuple(local))
    return tuple(out)


def senior_global_search_prefix_limit(num_nodes: int) -> int:
    """Senior GlobalSearch endpoint cap: ``min(0.5 * N, 10000)``."""

    if int(num_nodes) <= 0:
        raise V5ProtocolError("num_nodes must be positive")
    return min(int(0.50 * int(num_nodes)), 10_000)


def subgraph_density_controlled(candidate_score_sum: float, candidate_size: int, *, full_score_sum: float, num_nodes: int) -> float:
    """Senior density/gain score for a candidate prefix.

    This is the same form as the uploaded senior code:
    ``(sum(candidate) - sum(graph) * |candidate| / |graph|) / sqrt(|candidate|)``.
    """

    if candidate_size <= 0:
        return float("-inf")
    return (float(candidate_score_sum) - float(full_score_sum) * (float(candidate_size) / float(num_nodes))) / math.sqrt(float(candidate_size))


def senior_global_search_from_ranked_scores(
    *,
    query_nodes: Sequence[int],
    query_scores: Sequence[float],
    ranked_nodes: Sequence[int],
    ranked_scores: Sequence[float],
    num_nodes: int,
    full_score_sum: float,
) -> tuple[int, ...]:
    """Apply senior GlobalSearch to an already ranked score view.

    ``ranked_nodes`` and ``ranked_scores`` should contain the largest scores from
    full-graph cosine scoring.  The full score sum must be computed over all N
    nodes, even when ranking is chunked.
    """

    n = int(num_nodes)
    if n <= 0:
        raise V5ProtocolError("num_nodes must be positive")
    query = tuple(dict.fromkeys(int(x) for x in query_nodes))
    q_scores = tuple(float(x) for x in query_scores)
    ranked = tuple(int(x) for x in ranked_nodes)
    scores = tuple(float(x) for x in ranked_scores)
    if not query:
        raise V5ProtocolError("GlobalSearch requires at least one query seed")
    if len(query) != len(q_scores):
        raise V5ProtocolError("query_nodes and query_scores length mismatch")
    if len(ranked) != len(scores):
        raise V5ProtocolError("ranked_nodes and ranked_scores length mismatch")
    if any(x < 0 or x >= n for x in query) or any(x < 0 or x >= n for x in ranked):
        raise V5ProtocolError("local node id outside graph range")
    if not all(math.isfinite(x) for x in (*q_scores, *scores, float(full_score_sum))):
        raise V5ProtocolError("GlobalSearch received non-finite scores")

    endpoint = min(senior_global_search_prefix_limit(n), len(ranked))
    if endpoint <= 0:
        return query

    # Prefix sums for ranked candidates.  Query seeds may also appear in ranked;
    # they are allowed for senior compatibility, but final output is deduplicated.
    score_prefix = np.zeros(endpoint + 1, dtype=np.float64)
    if endpoint > 0:
        score_prefix[1:] = np.cumsum(np.asarray(scores[:endpoint], dtype=np.float64))
    q_score_sum = float(sum(q_scores))

    start = 0
    end = int(endpoint)
    while True:
        mid = int((start + end) / 2)
        half_sum = q_score_sum + float(score_prefix[mid])
        end_sum = q_score_sum + float(score_prefix[end])
        half_size = len(query) + mid
        end_size = len(query) + end
        half_density = subgraph_density_controlled(half_sum, half_size, full_score_sum=full_score_sum, num_nodes=n)
        end_density = subgraph_density_controlled(end_sum, end_size, full_score_sum=full_score_sum, num_nodes=n)
        if end_density >= half_density:
            start = mid
        else:
            end = mid
        if start == end or start + 1 == end:
            break

    selected = list(query) + list(ranked[:start])
    return tuple(dict.fromkeys(int(x) for x in selected))


def senior_global_search_from_full_scores(query_nodes: Sequence[int], scores: torch.Tensor | Sequence[float]) -> tuple[int, ...]:
    """Reference implementation using a full CPU score vector for small graphs/tests."""

    if isinstance(scores, torch.Tensor):
        score_tensor = scores.detach().cpu().float().flatten()
        score_list = score_tensor.tolist()
    else:
        score_list = [float(x) for x in scores]
        score_tensor = torch.tensor(score_list, dtype=torch.float32)
    ranked_nodes = torch.argsort(score_tensor, descending=True).tolist()
    ranked_scores = [score_list[i] for i in ranked_nodes]
    q = tuple(int(x) for x in query_nodes)
    return senior_global_search_from_ranked_scores(
        query_nodes=q,
        query_scores=[score_list[i] for i in q],
        ranked_nodes=ranked_nodes,
        ranked_scores=ranked_scores,
        num_nodes=len(score_list),
        full_score_sum=float(score_tensor.sum().item()),
    )
