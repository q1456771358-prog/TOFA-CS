#!/usr/bin/env python3
"""Formal V6 entry for the hierarchical target-conditioned method framework.

The public framework is part of the real execution path, but the numerical
prediction remains delegated to the immutable V575 evaluator.  After V575 has
frozen its predictions, this entry extracts the effective target condition,
query fusion, query core, score composition, and bidirectional boundary trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ofacs.hierarchical_target_conditioned_framework import (
    IMMUTABLE_V575_PRESET,
    METHOD_IDENTITY,
    RELEASE_IDENTITY,
    METHOD_PROTOCOL,
    PUBLIC_METHOD_PRESET,
    TRAINING_CHECKPOINT_SOURCE,
    MethodContract,
    build_immutable_v575_execution_plan,
    write_hierarchical_search_trace,
)
from ofacs.v575_kernel_lock import verify_kernel_lock
from ofacs.v6_posthoc_diagnostics import write_posthoc_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stable-evidence-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--method-preset", default=PUBLIC_METHOD_PRESET)
    parser.add_argument(
        "--method-config",
        default=str(PROJECT_ROOT / "configs" / "v6_method_contract.json"),
    )
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def build_v575_command(args: argparse.Namespace) -> list[str]:
    """Build the unchanged V575 child command used by the validated wrapper."""

    plan = build_immutable_v575_execution_plan(
        project_root=PROJECT_ROOT,
        python_bin=str(args.python_bin),
        graph_name=str(args.graph_name),
        checkpoint=args.checkpoint,
        stable_evidence_checkpoint=args.stable_evidence_checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=str(args.device),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        deterministic=bool(args.deterministic),
    )
    return list(plan.command)


def main() -> None:
    args = parse_args()
    if args.method_preset != PUBLIC_METHOD_PRESET:
        raise SystemExit(f"formal framework accepts only --method-preset {PUBLIC_METHOD_PRESET}")

    config_path = Path(args.method_config).resolve()
    contract = MethodContract.from_json(config_path)
    kernel_lock = verify_kernel_lock(PROJECT_ROOT)

    checkpoint = Path(args.checkpoint).resolve()
    stable_checkpoint = Path(args.stable_evidence_checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    for path, label in (
        (checkpoint, "checkpoint"),
        (stable_checkpoint, "stable evidence checkpoint"),
        (dataset_root, "dataset root"),
    ):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_immutable_v575_execution_plan(
        project_root=PROJECT_ROOT,
        python_bin=str(args.python_bin),
        graph_name=str(args.graph_name),
        checkpoint=checkpoint,
        stable_evidence_checkpoint=stable_checkpoint,
        dataset_root=dataset_root,
        output_dir=output_dir,
        device=str(args.device),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        deterministic=bool(args.deterministic),
    )
    plan.run(cwd=PROJECT_ROOT)

    prediction = output_dir / f"{args.graph_name}.pred"
    prediction_jsonl = output_dir / f"{args.graph_name}.pred.jsonl"
    resolved = output_dir / "fixed_inference_resolved_args.json"
    if not prediction.is_file() or not prediction_jsonl.is_file() or not resolved.is_file():
        raise SystemExit("immutable V575 evaluator returned without frozen prediction artifacts")

    interface_payload = {
        "protocol": METHOD_PROTOCOL,
        "release_identity": RELEASE_IDENTITY,
        "method_identity": METHOD_IDENTITY,
        "training_checkpoint_source": TRAINING_CHECKPOINT_SOURCE,
        "public_method_preset": PUBLIC_METHOD_PRESET,
        "immutable_v575_preset": IMMUTABLE_V575_PRESET,
        "method_config": str(config_path),
        "method_contract": contract.to_dict(),
        "execution_plan": plan.to_dict(),
        "diagnostics_enabled": bool(args.diagnostics),
        "diagnostics_are_posthoc": True,
        "numerical_reimplementation": False,
        "kernel_lock": kernel_lock,
    }
    interface_path = output_dir / "v58_method_framework_resolved.json"
    interface_path.write_text(
        json.dumps(interface_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.diagnostics:
        trace_path = write_hierarchical_search_trace(
            output_dir=output_dir,
            graph_name=args.graph_name,
            method_contract=contract,
            execution_plan=plan,
        )
        diagnostic_path = write_posthoc_diagnostics(
            output_dir=output_dir,
            graph_name=args.graph_name,
            interface_payload=interface_payload,
        )
        print(f"wrote {trace_path}")
        print(f"wrote {diagnostic_path}")

    print(f"wrote {interface_path}")
    print(f"prediction frozen by immutable V575 kernel: {prediction}")


if __name__ == "__main__":
    main()
