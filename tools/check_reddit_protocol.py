#!/usr/bin/env python3
"""Validate a fixed TransZero-style Reddit protocol before expensive cache builds.

The checker is intentionally strict about protocol identity, not model quality.  It
verifies that the generated reddit_pyg.pt / reddit.query / reddit.gt /
reddit.edges files are internally consistent with TransZero's Reddit semantics:
queries are sampled from one label class, each ground truth is the full label
class, and edges contain no self-loops.  It also checks recorded SHA256 hashes so
future runs can prove they reuse exactly the same protocol files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


class RedditProtocolError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1024 * 1024 * 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_index_rows(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                rows.append([])
                continue
            try:
                rows.append([int(x) for x in stripped.split()])
            except ValueError as exc:
                raise RedditProtocolError(f"{path} line {lineno} contains non-integer tokens") from exc
    return rows


def load_recorded_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise RedditProtocolError(f"bad sha256 line {lineno}: {line!r}")
            digest, rel = parts[0], parts[-1]
            out[rel] = digest
    return out


def _coerce_sparse_adj(adj: torch.Tensor) -> torch.Tensor:
    if not isinstance(adj, torch.Tensor):
        raise RedditProtocolError("reddit_pyg.pt[0] must be a torch Tensor sparse COO adjacency")
    if not adj.is_sparse:
        raise RedditProtocolError("reddit_pyg.pt[0] must be sparse COO, matching TransZero reddit_pyg.pt")
    return adj.coalesce().cpu()


def _check_edges(edge_path: Path, *, num_nodes: int, max_bad_examples: int = 5) -> dict[str, Any]:
    if not edge_path.exists():
        raise RedditProtocolError(f"missing edge file: {edge_path}")
    edge_lines = 0
    self_loops = 0
    out_of_range = 0
    bad_examples: list[str] = []
    with edge_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append(f"line {lineno}: expected 2 columns, got {len(parts)}")
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append(f"line {lineno}: non-integer edge")
                continue
            edge_lines += 1
            if u == v:
                self_loops += 1
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append(f"line {lineno}: self-loop {u} {v}")
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                out_of_range += 1
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append(f"line {lineno}: out-of-range edge {u} {v}")
    return {
        "edge_lines": int(edge_lines),
        "self_loops": int(self_loops),
        "out_of_range": int(out_of_range),
        "bad_examples": bad_examples,
    }


def check_protocol(data_root: str | Path, *, expected_queries: int = 100, check_sha256: bool = True, check_edges: bool = True) -> dict[str, Any]:
    root = Path(data_root).resolve()
    reddit_pyg = root / "reddit_pyg.pt"
    dataset_dir = root / "reddit"
    query_path = dataset_dir / "reddit.query"
    gt_path = dataset_dir / "reddit.gt"
    edge_path = dataset_dir / "reddit.edges"
    sha_path = dataset_dir / "reddit_protocol_sha256.txt"
    manifest_path = dataset_dir / "reddit_protocol_manifest.json"

    missing = [str(p) for p in [reddit_pyg, query_path, gt_path] if not p.exists()]
    if missing:
        raise RedditProtocolError("missing required protocol files: " + ", ".join(missing))

    obj = torch.load(reddit_pyg, map_location="cpu", weights_only=False)
    if not isinstance(obj, (list, tuple)) or len(obj) < 3:
        raise RedditProtocolError("reddit_pyg.pt must be [adj, x, y] as in TransZero")
    adj = _coerce_sparse_adj(obj[0])
    x = torch.as_tensor(obj[1]).cpu()
    y = torch.as_tensor(obj[2]).long().cpu().contiguous()
    if y.ndim != 1:
        raise RedditProtocolError(f"labels must be 1-D, got {tuple(y.shape)}")
    num_nodes = int(y.shape[0])
    if int(x.shape[0]) != num_nodes:
        raise RedditProtocolError(f"x/y node count mismatch: {x.shape[0]} vs {num_nodes}")
    if adj.shape != (num_nodes, num_nodes):
        raise RedditProtocolError(f"adjacency shape {tuple(adj.shape)} does not match num_nodes={num_nodes}")
    if x.dtype != torch.long:
        raise RedditProtocolError(f"reddit_pyg.pt[1] feature dtype should be torch.long to match TransZero, got {x.dtype}")
    if y.dtype != torch.long:
        raise RedditProtocolError(f"reddit_pyg.pt[2] label dtype should be torch.long, got {y.dtype}")

    queries = read_index_rows(query_path)
    gts = read_index_rows(gt_path)
    if len(queries) != int(expected_queries):
        raise RedditProtocolError(f"query count {len(queries)} != expected {expected_queries}")
    if len(gts) != len(queries):
        raise RedditProtocolError(f"gt count {len(gts)} != query count {len(queries)}")

    labels = y.tolist()
    label_to_nodes: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        label_to_nodes.setdefault(int(lab), []).append(idx)
    label_to_set = {lab: set(nodes) for lab, nodes in label_to_nodes.items()}

    bad_queries: list[dict[str, Any]] = []
    query_sizes: list[int] = []
    gt_sizes: list[int] = []
    selected_labels: list[int] = []
    for qi, (query, gt) in enumerate(zip(queries, gts)):
        if len(query) < 1 or len(query) > 3:
            bad_queries.append({"query_id": qi, "reason": "query_size_not_1_to_3", "query": query[:10]})
            continue
        if not gt:
            bad_queries.append({"query_id": qi, "reason": "empty_gt"})
            continue
        if any(v < 0 or v >= num_nodes for v in query):
            bad_queries.append({"query_id": qi, "reason": "query_node_out_of_range", "query": query[:10]})
            continue
        if any(v < 0 or v >= num_nodes for v in gt):
            bad_queries.append({"query_id": qi, "reason": "gt_node_out_of_range", "gt_head": gt[:10]})
            continue
        query_labels = {int(labels[v]) for v in query}
        gt_labels = {int(labels[v]) for v in gt}
        if len(query_labels) != 1:
            bad_queries.append({"query_id": qi, "reason": "query_nodes_have_multiple_labels", "labels": sorted(query_labels)})
            continue
        if len(gt_labels) != 1:
            bad_queries.append({"query_id": qi, "reason": "gt_nodes_have_multiple_labels", "labels": sorted(gt_labels)[:10]})
            continue
        q_label = next(iter(query_labels))
        gt_label = next(iter(gt_labels))
        if q_label != gt_label:
            bad_queries.append({"query_id": qi, "reason": "query_label_not_equal_gt_label", "query_label": q_label, "gt_label": gt_label})
            continue
        if set(gt) != label_to_set[q_label]:
            bad_queries.append({
                "query_id": qi,
                "reason": "gt_is_not_full_label_class",
                "label": q_label,
                "gt_size": len(gt),
                "class_size": len(label_to_set[q_label]),
            })
            continue
        query_sizes.append(len(query))
        gt_sizes.append(len(gt))
        selected_labels.append(q_label)
    if bad_queries:
        raise RedditProtocolError("bad query/gt rows: " + json.dumps(bad_queries[:5], ensure_ascii=False))

    edge_report: dict[str, Any] | None = None
    if check_edges:
        edge_report = _check_edges(edge_path, num_nodes=num_nodes)
        if edge_report["self_loops"] or edge_report["out_of_range"] or edge_report["bad_examples"]:
            raise RedditProtocolError("reddit.edges failed validation: " + json.dumps(edge_report, ensure_ascii=False))

    hash_report: dict[str, Any] = {"checked": bool(check_sha256), "matches": {}, "mismatches": {}}
    if check_sha256:
        recorded = load_recorded_hashes(sha_path)
        rels = ["reddit_pyg.pt", "reddit/reddit.query", "reddit/reddit.gt"]
        if edge_path.exists():
            rels.append("reddit/reddit.edges")
        for rel in rels:
            p = root / rel
            actual = sha256_file(p)
            expected = recorded.get(rel)
            if expected is None:
                hash_report["mismatches"][rel] = {"expected": None, "actual": actual, "reason": "missing_recorded_hash"}
            elif actual != expected:
                hash_report["mismatches"][rel] = {"expected": expected, "actual": actual}
            else:
                hash_report["matches"][rel] = actual
        if hash_report["mismatches"]:
            raise RedditProtocolError("sha256 mismatch: " + json.dumps(hash_report["mismatches"], ensure_ascii=False))

    manifest_protocol = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_protocol = manifest.get("protocol")
        except Exception:
            manifest_protocol = "unreadable"

    report: dict[str, Any] = {
        "status": "ok",
        "protocol": manifest_protocol or "transzero_reddit_pyg_seeded_query_gt_v1",
        "data_root": str(root),
        "num_nodes": num_nodes,
        "feature_shape": list(x.shape),
        "feature_dtype": str(x.dtype),
        "label_dtype": str(y.dtype),
        "label_min": int(y.min().item()),
        "label_max": int(y.max().item()),
        "num_classes": int(len(label_to_nodes)),
        "adj_shape": list(adj.shape),
        "adj_nnz": int(adj._nnz()),
        "query_count": int(len(queries)),
        "query_size_min": int(min(query_sizes) if query_sizes else 0),
        "query_size_max": int(max(query_sizes) if query_sizes else 0),
        "gt_size_min": int(min(gt_sizes) if gt_sizes else 0),
        "gt_size_max": int(max(gt_sizes) if gt_sizes else 0),
        "gt_size_mean": float(sum(gt_sizes) / max(1, len(gt_sizes))),
        "selected_label_hist": {str(lab): int(selected_labels.count(lab)) for lab in sorted(set(selected_labels))},
        "edge_report": edge_report,
        "sha256_report": hash_report,
        "checks": {
            "queries_are_1_to_3_nodes": True,
            "query_nodes_belong_to_gt_label": True,
            "gt_equals_full_label_class": True,
            "edge_file_has_no_self_loops": bool(edge_report is None or edge_report["self_loops"] == 0),
            "sha256_matches_recorded": bool(not hash_report.get("mismatches")),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="root containing reddit_pyg.pt and reddit/reddit.*")
    p.add_argument("--expected-queries", type=int, default=100)
    p.add_argument("--skip-sha256", action="store_true", help="skip hashing large files; not recommended for protocol lock")
    p.add_argument("--skip-edge-check", action="store_true", help="skip full reddit.edges validation")
    p.add_argument("--report-json", default=None, help="optional output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = check_protocol(
        args.data_root,
        expected_queries=int(args.expected_queries),
        check_sha256=not bool(args.skip_sha256),
        check_edges=not bool(args.skip_edge_check),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
