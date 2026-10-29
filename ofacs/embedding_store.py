"""Disk-backed embedding shards for large V5 source diagnostics.

The formal path must not keep large ``[N,H]`` embedding tensors in memory when a
chunked artifact is sufficient.  This module writes no labels, no ground truth,
and no metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from .chunked_scoring import ChunkedScoreResult, _normalize_query_sets, default_top_k_pool
from .errors import V5ProtocolError
from .runtime_profile import profile_event
from .senior_compat import SeniorQueryBatch


EMBEDDING_SHARD_MANIFEST = "embedding_shards_manifest.json"


@dataclass(frozen=True)
class EmbeddingShardManifest:
    version: str
    graph_name: str
    num_nodes: int
    embedding_dim: int
    dtype: str
    shard_size: int
    shard_count: int
    shards: tuple[dict[str, object], ...]
    labels_used: bool = False
    ground_truth_used: bool = False
    metrics_computed: bool = False

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["shards"] = list(self.shards)
        return out


def save_embedding_shards(
    *,
    embeddings: torch.Tensor,
    output_dir: str | Path,
    graph_name: str,
    shard_size: int = 8192,
    metadata: Mapping[str, object] | None = None,
    runtime_jsonl: str | Path | None = None,
) -> Path:
    if embeddings.ndim != 2:
        raise V5ProtocolError("embeddings must be [N,H]")
    if int(shard_size) <= 0:
        raise V5ProtocolError("embedding shard_size must be positive")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, h = int(embeddings.shape[0]), int(embeddings.shape[1])
    shards: list[dict[str, object]] = []
    for shard_idx, start in enumerate(range(0, n, int(shard_size))):
        end = min(n, start + int(shard_size))
        shard_path = out_dir / f"embeddings_{shard_idx:06d}.pt"
        profile_event(runtime_jsonl, "embedding_shard_write_start", graph_name=graph_name, shard_index=shard_idx, start=start, end=end)
        torch.save(
            {
                "start": int(start),
                "end": int(end),
                "embeddings": embeddings[start:end].detach().cpu().contiguous(),
                "labels_used": False,
                "ground_truth_used": False,
                "metrics_computed": False,
            },
            shard_path,
        )
        shards.append({"path": shard_path.name, "start": int(start), "end": int(end), "num_nodes": int(end - start)})
        profile_event(runtime_jsonl, "embedding_shard_write_end", graph_name=graph_name, shard_index=shard_idx, start=start, end=end)
    manifest = EmbeddingShardManifest(
        version="v5-embedding-shards-v1",
        graph_name=str(graph_name),
        num_nodes=n,
        embedding_dim=h,
        dtype=str(embeddings.dtype),
        shard_size=int(shard_size),
        shard_count=len(shards),
        shards=tuple(shards),
    )
    payload = manifest.to_dict()
    payload["metadata"] = dict(metadata or {})
    manifest_path = out_dir / EMBEDDING_SHARD_MANIFEST
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def save_embedding_chunk_stream(
    *,
    chunks: object,
    output_dir: str | Path,
    graph_name: str,
    embedding_dim: int,
    num_nodes: int,
    dtype: str = "torch.float32",
    metadata: Mapping[str, object] | None = None,
    runtime_jsonl: str | Path | None = None,
) -> Path:
    """Write embedding chunks yielded as ``(start, end, tensor)`` without concatenation."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    for shard_idx, item in enumerate(chunks):  # type: ignore[assignment]
        start, end, emb = item
        start_i, end_i = int(start), int(end)
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
            raise V5ProtocolError("embedding chunk stream must yield [C,H] tensors")
        if int(emb.shape[0]) != end_i - start_i or int(emb.shape[1]) != int(embedding_dim):
            raise V5ProtocolError("embedding chunk shape does not match stream range")
        shard_path = out_dir / f"embeddings_{shard_idx:06d}.pt"
        profile_event(runtime_jsonl, "embedding_shard_write_start", graph_name=graph_name, shard_index=shard_idx, start=start_i, end=end_i)
        torch.save(
            {
                "start": start_i,
                "end": end_i,
                "embeddings": emb.detach().cpu().contiguous(),
                "labels_used": False,
                "ground_truth_used": False,
                "metrics_computed": False,
            },
            shard_path,
        )
        shards.append({"path": shard_path.name, "start": start_i, "end": end_i, "num_nodes": end_i - start_i})
        profile_event(runtime_jsonl, "embedding_shard_write_end", graph_name=graph_name, shard_index=shard_idx, start=start_i, end=end_i)
    if not shards:
        raise V5ProtocolError("embedding chunk stream yielded no chunks")
    if int(shards[0]["start"]) != 0 or int(shards[-1]["end"]) != int(num_nodes):
        raise V5ProtocolError("embedding chunk stream does not cover [0,N)")
    manifest = EmbeddingShardManifest(
        version="v5-embedding-shards-v1",
        graph_name=str(graph_name),
        num_nodes=int(num_nodes),
        embedding_dim=int(embedding_dim),
        dtype=str(dtype),
        shard_size=max(int(row["num_nodes"]) for row in shards),
        shard_count=len(shards),
        shards=tuple(shards),
    )
    payload = manifest.to_dict()
    payload["metadata"] = dict(metadata or {})
    manifest_path = out_dir / EMBEDDING_SHARD_MANIFEST
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def resolve_embedding_manifest(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_dir():
        candidate = p / EMBEDDING_SHARD_MANIFEST
        return candidate if candidate.exists() else None
    if p.name == EMBEDDING_SHARD_MANIFEST and p.exists():
        return p
    return None


def load_embedding_manifest(path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = resolve_embedding_manifest(path)
    if manifest_path is None:
        raise V5ProtocolError(f"embedding shard manifest not found: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != "v5-embedding-shards-v1":
        raise V5ProtocolError("unsupported embedding shard manifest version")
    if bool(payload.get("labels_used", False)) or bool(payload.get("ground_truth_used", False)) or bool(payload.get("metrics_computed", False)):
        raise V5ProtocolError("embedding shard manifest has unsafe supervision flags")
    return manifest_path, payload


def iter_embedding_shards(path: str | Path) -> tuple[dict[str, object], ...]:
    manifest_path, payload = load_embedding_manifest(path)
    base = manifest_path.parent
    rows: list[dict[str, object]] = []
    for row in payload.get("shards", []):
        if not isinstance(row, dict):
            raise V5ProtocolError("invalid embedding shard row")
        shard_path = base / str(row.get("path"))
        obj = torch.load(shard_path, map_location="cpu")
        emb = obj.get("embeddings") if isinstance(obj, dict) else None
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
            raise V5ProtocolError(f"invalid embedding shard tensor: {shard_path}")
        rows.append({"path": shard_path, "start": int(row["start"]), "end": int(row["end"]), "embeddings": emb.float()})
    return tuple(rows)


def load_embedding_artifact(path: str | Path) -> torch.Tensor:
    manifest_path = resolve_embedding_manifest(path)
    if manifest_path is None:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for key in ("embeddings", "node_embeddings", "refined_embedding", "embedding"):
                value = obj.get(key)
                if isinstance(value, torch.Tensor):
                    return value
        raise V5ProtocolError(f"could not find embedding tensor in {path}")
    shards = iter_embedding_shards(manifest_path)
    if not shards:
        raise V5ProtocolError("embedding shard manifest contains no shards")
    return torch.cat([row["embeddings"] for row in shards], dim=0)


def chunked_query_cosine_from_embedding_shards(
    *,
    manifest_path: str | Path,
    query_node_sets: SeniorQueryBatch,
    chunk_size: int = 4096,
    top_k_pool: int | None = None,
    eps: float = 1e-12,
    runtime_jsonl: str | Path | None = None,
    score_calibration: str = "none",
) -> ChunkedScoreResult:
    manifest_file, manifest = load_embedding_manifest(manifest_path)
    n = int(manifest["num_nodes"])
    h = int(manifest["embedding_dim"])
    if int(chunk_size) <= 0:
        raise V5ProtocolError("chunk_size must be positive")
    qsets = _normalize_query_sets(query_node_sets, num_nodes=n)
    q_count = len(qsets)
    max_q = max(len(q) for q in qsets)
    k_pool = default_top_k_pool(n, max_q) if top_k_pool is None else min(int(top_k_pool), n)
    seed_to_queries: dict[int, list[int]] = {}
    for qi, qset in enumerate(qsets):
        for seed in qset:
            seed_to_queries.setdefault(int(seed), []).append(qi)
    seed_vectors: dict[int, torch.Tensor] = {}
    shards = tuple(dict(row) for row in manifest.get("shards", []))
    for shard_index, row in enumerate(shards):
        start, end = int(row["start"]), int(row["end"])
        needed = [seed for seed in seed_to_queries if start <= seed < end]
        if not needed:
            continue
        emb = _load_shard_tensor(manifest_file.parent / str(row["path"]), start=start, end=end)
        emb_norm = F.normalize(emb.float(), p=2, dim=-1, eps=eps)
        for seed in needed:
            seed_vectors[int(seed)] = emb_norm[int(seed) - start].detach().cpu()
        profile_event(runtime_jsonl, "embedding_shard_query_seed_read", shard_index=shard_index, start=start, end=end, seed_count=len(needed))
    query_embs = []
    for qset in qsets:
        vectors = [seed_vectors[int(seed)] for seed in qset]
        query_embs.append(torch.stack(vectors, dim=0).mean(dim=0))
    query_embs = F.normalize(torch.stack(query_embs, dim=0).float(), p=2, dim=-1, eps=eps)

    calibration = str(score_calibration or "none").lower()
    if calibration not in {"none", "center", "zscore"}:
        raise V5ProtocolError(f"unsupported score_calibration={score_calibration!r}")
    cal_mean = torch.zeros(q_count, dtype=torch.float64)
    cal_std = torch.ones(q_count, dtype=torch.float64)
    if calibration != "none":
        sum1 = torch.zeros(q_count, dtype=torch.float64)
        sum2 = torch.zeros(q_count, dtype=torch.float64)
        for row in shards:
            shard_start, shard_end = int(row["start"]), int(row["end"])
            shard = _load_shard_tensor(manifest_file.parent / str(row["path"]), start=shard_start, end=shard_end).float()
            for local_start in range(0, int(shard.shape[0]), int(chunk_size)):
                local_end = min(int(shard.shape[0]), local_start + int(chunk_size))
                chunk = F.normalize(shard[local_start:local_end], p=2, dim=-1, eps=eps)
                scores = query_embs @ chunk.t()
                sum1 += scores.double().sum(dim=1)
                sum2 += scores.double().pow(2).sum(dim=1)
        cal_mean = sum1 / float(n)
        if calibration == "zscore":
            cal_std = torch.sqrt(torch.clamp(sum2 / float(n) - cal_mean.pow(2), min=1e-12))

    top_scores = torch.empty((q_count, 0), dtype=torch.float32)
    top_nodes = torch.empty((q_count, 0), dtype=torch.long)
    full_sums = torch.zeros(q_count, dtype=torch.float64)
    query_seed_scores: list[dict[int, float]] = [dict() for _ in range(q_count)]
    chunk_count = 0
    max_live_c = 0

    for shard_index, row in enumerate(shards):
        shard_start, shard_end = int(row["start"]), int(row["end"])
        shard = _load_shard_tensor(manifest_file.parent / str(row["path"]), start=shard_start, end=shard_end).float()
        for local_start in range(0, int(shard.shape[0]), int(chunk_size)):
            local_end = min(int(shard.shape[0]), local_start + int(chunk_size))
            start = shard_start + local_start
            end = shard_start + local_end
            profile_event(runtime_jsonl, "embedding_shard_score_chunk_start", shard_index=shard_index, start=start, end=end)
            chunk = F.normalize(shard[local_start:local_end], p=2, dim=-1, eps=eps)
            scores = query_embs @ chunk.t()
            if calibration != "none":
                scores = ((scores.double() - cal_mean.view(-1, 1)) / cal_std.view(-1, 1)).float()
            chunk_count += 1
            max_live_c = max(max_live_c, int(scores.shape[1]))
            full_sums += scores.double().sum(dim=1)
            node_ids = torch.arange(start, end, dtype=torch.long).unsqueeze(0).expand(q_count, -1)
            merged_scores = torch.cat([top_scores, scores], dim=1)
            merged_nodes = torch.cat([top_nodes, node_ids], dim=1)
            keep = min(k_pool, int(merged_scores.shape[1]))
            top_scores, pos = torch.topk(merged_scores, k=keep, dim=1, largest=True, sorted=True)
            top_nodes = torch.gather(merged_nodes, dim=1, index=pos)
            for qi, qset in enumerate(qsets):
                for seed in qset:
                    if start <= seed < end:
                        query_seed_scores[qi][int(seed)] = float(scores[qi, int(seed) - start].detach().cpu().item())
            profile_event(runtime_jsonl, "embedding_shard_score_chunk_end", shard_index=shard_index, start=start, end=end)

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


def _load_shard_tensor(path: Path, *, start: int, end: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise V5ProtocolError(f"embedding shard must be a mapping: {path}")
    if bool(obj.get("labels_used", False)) or bool(obj.get("ground_truth_used", False)) or bool(obj.get("metrics_computed", False)):
        raise V5ProtocolError(f"embedding shard has unsafe flags: {path}")
    emb = obj.get("embeddings")
    if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
        raise V5ProtocolError(f"embedding shard missing [C,H] tensor: {path}")
    if int(obj.get("start", start)) != int(start) or int(obj.get("end", end)) != int(end):
        raise V5ProtocolError(f"embedding shard range mismatch: {path}")
    return emb.detach().cpu()
