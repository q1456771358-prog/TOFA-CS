#!/usr/bin/env python3
"""Build strict senior preprocessing caches with dense or sparse-safe exact semantics.

For small graphs this keeps the original dense senior feature-alignment branch.
For large graphs it uses a mathematically equivalent sparse-safe normalized-
Laplacian energy computation and never materializes [N,N] dense adjacency.
It still builds the strict cache:

    <graph>_d64_khop4_Nl10_pos10_neg10.pt

The command is single-graph friendly and writes explicit metadata so sparse-safe
exact caches are not mistaken for sampled or approximate caches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from senior_core import aa_utils_original as senior_utils
from ofacs.errors import V5ProtocolError
from ofacs.senior_data import (
    _coerce_features,
    _load_torch,
    _unpack_raw_graph,
    adjacency_to_sparse_and_edge_index,
    discover_senior_dataset_files,
    validate_dataset_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--graphs", required=True, help="comma-separated graph names")
    p.add_argument("--dims", type=int, default=64)
    p.add_argument("--hops", type=int, default=4)
    p.add_argument("--local-topk", type=int, default=10)
    p.add_argument("--pos-count", type=int, default=10)
    p.add_argument("--neg-count", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=None, help="default: dataset root")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-output", action="store_true")
    p.add_argument("--metadata-json", default=None)
    p.add_argument("--sparse-safe", action="store_true", help="force sparse-safe exact feature alignment")
    p.add_argument("--dense-threshold-gb", type=float, default=8.0, help="auto-use sparse-safe alignment when dense [N,N] adjacency exceeds this GiB threshold")
    p.add_argument("--chunk-size", type=int, default=8192, help="reserved for block-safe large-graph utilities; kept for CLI compatibility")
    p.add_argument("--disable-large-graph-fast-path", action="store_true", help="disable equivalent vectorized cache-build accelerations for large sparse graphs")
    p.add_argument("--fast-threshold-nodes", type=int, default=50000, help="enable large-graph fast path at or above this node count")
    p.add_argument("--edge-score-chunk-size", type=int, default=1000000, help="edge chunk size for vectorized local-token cosine scoring")
    p.add_argument("--local-row-block-size", type=int, default=4096, help="row block size for large-graph local-token scoring; avoids global scores[E] allocation")
    p.add_argument("--progress-every", type=int, default=50, help="print progress every N row blocks during large-graph cache build")
    p.add_argument("--pos-neg-sampler", choices=["auto", "fast", "original"], default="auto", help="pos/neg sample implementation for large graphs; original keeps legacy RNG loop, fast keeps sampling constraints")
    p.add_argument("--local-score-device", choices=["auto", "cpu", "cuda"], default="auto", help="device for large-graph local edge cosine scoring")
    p.add_argument("--global-token-device", choices=["auto", "cpu", "cuda"], default="auto", help="device for sparse global-token propagation")
    p.add_argument("--preflight-only", action="store_true", help="load graph metadata and report sparse/dense decision without building cache")
    return p.parse_args()


def _strict_cache_path(root: Path, graph: str, args: argparse.Namespace) -> Path:
    return root / f"{graph}_d{args.dims}_khop{args.hops}_Nl{args.local_topk}_pos{args.pos_count}_neg{args.neg_count}.pt"


def _alignment_mode_for_graph(num_nodes: int, adjacency: torch.Tensor, args: argparse.Namespace) -> tuple[bool, float, str]:
    dense_gb = senior_utils.estimate_dense_alignment_gb(num_nodes, dtype=adjacency.dtype)
    if bool(getattr(args, "sparse_safe", False)):
        return True, dense_gb, "forced_sparse_safe"
    if dense_gb > float(args.dense_threshold_gb):
        return True, dense_gb, "auto_sparse_safe_dense_threshold"
    return False, dense_gb, "dense_within_threshold"


def _build_one(graph: str, args: argparse.Namespace) -> dict[str, Any]:
    name = validate_dataset_name(graph)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root) if args.output_root else dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = _strict_cache_path(output_root, name, args)
    if out_path.exists() and not args.overwrite:
        return {"graph": name, "status": "exists", "cache_path": str(out_path)}

    files = discover_senior_dataset_files(
        dataset_root,
        name,
        dims=int(args.dims),
        hops=int(args.hops),
        local_topk=int(args.local_topk),
        pos_count=int(args.pos_count),
        neg_count=int(args.neg_count),
    )
    if not files.graph_pt_exists:
        raise V5ProtocolError(f"raw graph file missing for {name}: {files.graph_pt_path}")
    raw_obj = _load_torch(files.graph_pt_path, map_location="cpu")
    adjacency_raw, features_raw, labels_raw, labels_present = _unpack_raw_graph(raw_obj)
    if not labels_present or labels_raw is None:
        raise V5ProtocolError(f"strict cache generation requires source/community labels for {name}")
    features = _coerce_features(features_raw)
    labels = torch.as_tensor(labels_raw).long().contiguous()
    if labels.ndim != 1 or int(labels.shape[0]) != int(features.shape[0]):
        raise V5ProtocolError(f"labels must be [N], got {tuple(labels.shape)} for N={features.shape[0]}")
    adjacency, edge_index, adjacency_format = adjacency_to_sparse_and_edge_index(adjacency_raw, num_nodes=int(features.shape[0]), allow_dense=True)
    adjacency = adjacency.coalesce().cpu()
    edge_tuple = (edge_index[0].cpu(), edge_index[1].cpu())

    use_sparse_safe, dense_gb, alignment_mode_reason = _alignment_mode_for_graph(int(features.shape[0]), adjacency, args)
    preflight_report = {
        "graph": name,
        "status": "preflight",
        "num_nodes": int(features.shape[0]),
        "raw_feature_dim": int(features.shape[1]),
        "adjacency_nnz": int(adjacency._nnz()),
        "estimated_dense_adj_gb": float(dense_gb),
        "dense_threshold_gb": float(args.dense_threshold_gb),
        "sparse_safe_alignment": bool(use_sparse_safe),
        "alignment_mode_reason": alignment_mode_reason,
        "will_call_adj_to_dense": bool(not use_sparse_safe),
        "large_graph_fast_path": bool((not args.disable_large_graph_fast_path) and int(features.shape[0]) >= int(args.fast_threshold_nodes)),
        "fast_threshold_nodes": int(args.fast_threshold_nodes),
        "edge_score_chunk_size": int(args.edge_score_chunk_size),
        "local_row_block_size": int(args.local_row_block_size),
        "progress_every": int(args.progress_every),
        "pos_neg_sampler": str(args.pos_neg_sampler),
        "local_score_device": str(args.local_score_device),
        "global_token_device": str(args.global_token_device),
        "alignment_mode": "sparse_exact_quadratic_form" if bool(use_sparse_safe) else "dense_full_eigendecomposition",
        "alignment_formula": "diag(X^T L X) = sum_k lambda_k (u_k^T x_j)^2 under the full Laplacian eigenbasis" if bool(use_sparse_safe) else "torch.linalg.eigh(L) full dense basis",
        "equivalent_to_full_basis_eigendecomposition": bool(use_sparse_safe),
        "uses_low_rank_eig": False,
        "uses_randomized_eig": False,
        "uses_sampling": False,
        "drops_nodes": False,
        "drops_edges": False,
    }
    print(json.dumps(preflight_report, ensure_ascii=False, sort_keys=True), flush=True)
    if args.preflight_only:
        return preflight_report

    # The strict step: feature alignment.  For large graphs, keep adj/features on
    # CPU and force sparse-safe exact energy computation so --device cuda cannot
    # accidentally allocate a huge dense adjacency on GPU.
    if args.device == "cuda" and not use_sparse_safe:
        if not torch.cuda.is_available():
            raise V5ProtocolError("--device cuda requested but CUDA is unavailable")
        feat_in = features.cuda()
        adj_in = adjacency.cuda()
        edge_tuple_in = (edge_tuple[0].cuda(), edge_tuple[1].cuda())
    else:
        if args.device == "cuda" and use_sparse_safe:
            print("[sparse-safe-align] --device cuda requested, but alignment stays on CPU to avoid dense GPU adjacency", flush=True)
        feat_in = features.cpu()
        adj_in = adjacency.cpu()
        edge_tuple_in = edge_tuple
    aligned_feat = senior_utils.my_feat_alignment(
        feat_in,
        adj_in,
        edge_tuple_in,
        int(args.dims),
        sparse_safe=bool(use_sparse_safe),
        dense_threshold_gb=float(args.dense_threshold_gb),
    ).cpu().float().contiguous()

    helper = senior_utils.Dataset.__new__(senior_utils.Dataset)
    helper.name = name
    helper.dims = int(args.dims)
    helper.k_hop = int(args.hops)
    helper.local_topk = int(args.local_topk)
    helper.pos_count = int(args.pos_count)
    helper.neg_count = int(args.neg_count)
    helper.seed = int(args.seed)
    indptr, indices = helper._build_csr_from_coo(adjacency, num_nodes=int(aligned_feat.shape[0]))
    use_fast_large = bool((not args.disable_large_graph_fast_path) and int(aligned_feat.shape[0]) >= int(args.fast_threshold_nodes))
    local_score_device_actual = str(args.local_score_device)
    global_token_device_actual = str(args.global_token_device)
    pos_neg_sampler_actual = str(args.pos_neg_sampler)
    if use_fast_large:
        print(
            f"[large-fast] enabled for {name}: N={aligned_feat.shape[0]}, "
            f"threshold={args.fast_threshold_nodes}",
            flush=True,
        )
        local_score_device = str(args.local_score_device if args.local_score_device != "auto" else ("cuda" if args.device == "cuda" else "cpu"))
        try:
            local_token_seqs = helper._build_local_token_sequences_fast(
                aligned_feat,
                indptr,
                indices,
                topk=int(args.local_topk),
                edge_score_chunk_size=int(args.edge_score_chunk_size),
                score_device=local_score_device,
                seed=int(args.seed),
                row_block_size=int(args.local_row_block_size),
                progress_every=int(args.progress_every),
            ).float().contiguous()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or local_score_device == "cpu":
                raise
            print("[fast-local] CUDA edge scoring OOM; falling back to CPU local-token scoring", flush=True)
            torch.cuda.empty_cache()
            local_score_device = "cpu"
            local_token_seqs = helper._build_local_token_sequences_fast(
                aligned_feat,
                indptr,
                indices,
                topk=int(args.local_topk),
                edge_score_chunk_size=int(args.edge_score_chunk_size),
                score_device="cpu",
                seed=int(args.seed),
                row_block_size=int(args.local_row_block_size),
                progress_every=int(args.progress_every),
            ).float().contiguous()
        local_score_device_actual = local_score_device
        global_device = str(args.global_token_device)
        if global_device == "auto":
            global_device = "cuda" if args.device == "cuda" else "cpu"
        global_token_device_actual = global_device
        global_token_seqs = senior_utils.re_features_auto_device(
            adjacency.to(torch.float32),
            aligned_feat,
            int(args.hops),
            device=global_device,
        ).float().contiguous()
        pos_neg_sampler = str(args.pos_neg_sampler)
        if pos_neg_sampler == "auto":
            pos_neg_sampler = "fast"
        pos_neg_sampler_actual = pos_neg_sampler
        if pos_neg_sampler == "original":
            print("[pos-neg] using original senior RNG loop; this is stricter but may be slow on Reddit", flush=True)
            pos_samples, neg_samples = helper._build_pos_neg_samples(
                label=labels,
                indptr=indptr,
                indices=indices,
                pos_count=int(args.pos_count),
                neg_count=int(args.neg_count),
                seed=int(args.seed),
            )
        else:
            print("[pos-neg] using fast constraint-equivalent sampler", flush=True)
            pos_samples, neg_samples = helper._build_pos_neg_samples_fast(
                label=labels,
                indptr=indptr,
                indices=indices,
                pos_count=int(args.pos_count),
                neg_count=int(args.neg_count),
                seed=int(args.seed),
                progress_every_nodes=max(1, int(args.local_row_block_size) * int(args.progress_every)),
            )
    else:
        local_token_seqs = helper._build_local_token_sequences(aligned_feat, indptr, indices, topk=int(args.local_topk)).float().contiguous()
        global_token_seqs = senior_utils.re_features(adjacency.to(torch.float32), aligned_feat, int(args.hops)).float().contiguous()
        pos_samples, neg_samples = helper._build_pos_neg_samples(
            label=labels,
            indptr=indptr,
            indices=indices,
            pos_count=int(args.pos_count),
            neg_count=int(args.neg_count),
            seed=int(args.seed),
        )

    payload = {
        "adj": adjacency.coalesce(),
        "feat": aligned_feat,
        "label": labels,
        "local_token_seqs": local_token_seqs,
        "global_token_seqs": global_token_seqs,
        "pos_samples": pos_samples.long().contiguous(),
        "neg_samples": neg_samples.long().contiguous(),
        "cache_metadata": {
            "protocol": "senior-strict-cache-exact-sparse-safe" if bool(use_sparse_safe) else "senior-strict-cache-exact-dense",
            "graph_name": name,
            "dims": int(args.dims),
            "hops": int(args.hops),
            "local_topk": int(args.local_topk),
            "pos_count": int(args.pos_count),
            "neg_count": int(args.neg_count),
            "seed": int(args.seed),
            "feature_alignment": "senior_original_my_feat_alignment_sparse_exact_laplacian_energy" if bool(use_sparse_safe) else "senior_original_my_feat_alignment_full_dense_eigh",
            "alignment_mode": "sparse_exact_quadratic_form" if bool(use_sparse_safe) else "dense_full_eigendecomposition",
            "alignment_formula": "diag(X^T L X) = sum_k lambda_k (u_k^T x_j)^2 under the full Laplacian eigenbasis" if bool(use_sparse_safe) else "torch.linalg.eigh(L) full dense basis",
            "equivalent_to_full_basis_eigendecomposition": bool(use_sparse_safe),
            "approximate_eigendecomposition": False,
            "sparse_low_rank": False,
            "uses_low_rank_eig": False,
            "uses_randomized_eig": False,
            "uses_sampling": False,
            "drops_nodes": False,
            "drops_edges": False,
            "sparse_safe_alignment": bool(use_sparse_safe),
            "large_graph_fast_path": bool(use_fast_large),
            "fast_path_semantics": "equivalent_vectorized_sparse_safe_no_node_or_edge_drop" if bool(use_fast_large) else "original_python_loops",
            "edge_score_chunk_size": int(args.edge_score_chunk_size),
            "local_row_block_size": int(args.local_row_block_size),
            "progress_every": int(args.progress_every),
            "pos_neg_sampler": str(args.pos_neg_sampler),
            "pos_neg_sampler_actual": str(pos_neg_sampler_actual),
            "local_score_device": str(args.local_score_device),
            "local_score_device_actual": str(local_score_device_actual),
            "global_token_device": str(args.global_token_device),
            "global_token_device_actual": str(global_token_device_actual),
            "alignment_mode_reason": alignment_mode_reason,
            "estimated_dense_adj_gb": float(dense_gb),
            "dense_threshold_gb": float(args.dense_threshold_gb),
            "adjacency_format": adjacency_format,
            "num_nodes": int(aligned_feat.shape[0]),
            "raw_feature_dim": int(features.shape[1]),
        },
    }
    torch.save(payload, out_path)

    report = {
        "graph": name,
        "status": "built",
        "cache_path": str(out_path),
        "num_nodes": int(aligned_feat.shape[0]),
        "raw_feature_dim": int(features.shape[1]),
        "feat_shape": list(aligned_feat.shape),
        "global_token_shape": list(global_token_seqs.shape),
        "local_token_shape": list(local_token_seqs.shape),
        "pos_shape": list(pos_samples.shape),
        "neg_shape": list(neg_samples.shape),
        "strict_exact_dense": bool(not use_sparse_safe),
        "strict_exact_sparse_safe": bool(use_sparse_safe),
        "large_graph_fast_path": bool(use_fast_large),
        "estimated_dense_adj_gb": float(dense_gb),
        "alignment_mode_reason": alignment_mode_reason,
    }
    if args.validate_output:
        loaded = torch.load(out_path, map_location="cpu", weights_only=False)
        required = ("adj", "feat", "label", "local_token_seqs", "global_token_seqs", "pos_samples", "neg_samples")
        missing = [k for k in required if k not in loaded]
        if missing:
            raise V5ProtocolError(f"built cache missing keys: {missing}")
        if tuple(loaded["global_token_seqs"].shape) != (int(aligned_feat.shape[0]), int(args.hops) + 1, int(args.dims)):
            raise V5ProtocolError("global_token_seqs shape mismatch after save/load")
        report["validated"] = True
    return report


def main() -> None:
    args = parse_args()
    reports = []
    for raw in str(args.graphs).split(","):
        graph = raw.strip()
        if not graph:
            continue
        report = _build_one(graph, args)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if args.metadata_json:
        out = Path(args.metadata_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
