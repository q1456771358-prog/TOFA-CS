#!/usr/bin/env python3
"""Public entry point for the TOFA-CS paper main experiment.

The wrapper intentionally fixes the three source graphs and the paper method
configuration.  Users may choose target graphs and independently trained
seeds, but cannot silently change the source protocol through this interface.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER_CONFIG = ROOT / "configs" / "paper_main.json"
TARGETS = ("wisconsin", "cora", "citeseer", "cs", "physics", "reddit")
SOURCES = ("texas", "photo", "dblp")


def comma_list(value: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if values == ["all"]:
        return list(TARGETS)
    aliases = {"cocs": "cs"}
    values = [aliases.get(item, item) for item in values]
    unknown = sorted(set(values) - set(TARGETS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown target(s): {', '.join(unknown)}; choose from {', '.join(TARGETS)}"
        )
    if not values:
        raise argparse.ArgumentTypeError("at least one target is required")
    return list(dict.fromkeys(values))


def seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative integers")
    return list(dict.fromkeys(seeds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pretrained = subparsers.add_parser(
        "pretrained", help="evaluate one verified source-trained checkpoint bundle"
    )
    _add_common(pretrained)
    pretrained.add_argument("--model-root", type=Path, required=True)
    pretrained.add_argument(
        "--seed",
        type=int,
        default=None,
        help="training seed recorded for this model; inferred from model_manifest.json or seed_N directory",
    )

    train = subparsers.add_parser(
        "train", help="jointly train on Texas, Photo, and DBLP, then evaluate selected targets"
    )
    _add_common(train)
    train.add_argument("--seeds", type=seed_list, default=[0], help="e.g. 0 or 0,1,2")
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--targets", type=comma_list, default=["wisconsin"], help="comma list or all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")


def require_data(data_root: Path, graphs: list[str], *, require_gt: bool = True) -> None:
    missing: list[str] = []
    for graph in graphs:
        cache = data_root / f"{graph}_d64_khop4_Nl10_pos10_neg10.pt"
        graph_files = [
            data_root / f"{graph}_pyg.pt",
            data_root / f"{graph}_dgl.pt",
            data_root / graph / f"{graph}_pyg.pt",
            data_root / graph / f"{graph}_dgl.pt",
        ]
        query = data_root / graph / f"{graph}.query"
        gt = data_root / graph / f"{graph}.gt"
        if not cache.is_file():
            missing.append(str(cache))
        if not any(path.is_file() for path in graph_files):
            missing.append("one of: " + ", ".join(str(path) for path in graph_files))
        if not query.is_file():
            missing.append(str(query))
        if require_gt and not gt.is_file():
            missing.append(str(gt))
    if missing:
        raise SystemExit("missing required dataset files:\n  - " + "\n  - ".join(missing))


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = shlex.join(command)
    print(f"[command] {printable}")
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def infer_seed(model_root: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    manifest = model_root / "model_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "seed" in payload:
            return int(payload["seed"])
    if model_root.name.startswith("seed_"):
        try:
            return int(model_root.name.removeprefix("seed_"))
        except ValueError:
            pass
    raise SystemExit("cannot infer model seed; pass --seed or add model_manifest.json")


def pretrained_mode(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root.resolve()
    model_root = args.model_root.resolve()
    output_root = args.output_root.resolve()
    targets: list[str] = args.targets
    seed = infer_seed(model_root, args.seed)
    checkpoint = model_root / "trained_model.pt"
    stable = model_root / "frozen_reference.pt"
    if not args.dry_run:
        require_data(data_root, targets)
        for path in (checkpoint, stable):
            if not path.is_file():
                raise SystemExit(f"missing verified model file: {path}")

    run_root = output_root / f"pretrained_seed_{seed}"
    predictions_root = run_root / "predictions"
    commands: list[list[str]] = []
    for target in targets:
        command = [
            sys.executable,
            str(ROOT / "tools" / "evaluate_fixed_checkpoint_v6.py"),
            "--graph-name", target,
            "--checkpoint", str(checkpoint),
            "--stable-evidence-checkpoint", str(stable),
            "--dataset-root", str(data_root),
            "--output-dir", str(predictions_root / target),
            "--device", args.device,
            "--seed", str(seed),
            "--deterministic",
            "--diagnostics",
        ]
        commands.append(command)
        run_command(command, dry_run=args.dry_run)

    offline = [
        sys.executable,
        str(ROOT / "tools" / "offline_evaluate_predictions_ofacs.py"),
        "--dataset-root", str(data_root),
        "--graphs", ",".join(targets),
        "--predictions-root", str(predictions_root),
        "--output-json", str(run_root / "offline_eval.json"),
        "--output-csv", str(run_root / "offline_eval.csv"),
        "--allow-unseen-ground-truth-eval",
    ]
    commands.append(offline)
    run_command(offline, dry_run=args.dry_run)
    return {
        "mode": "pretrained",
        "seed": seed,
        "targets": targets,
        "model_root": str(model_root),
        "commands": commands,
        "output_root": str(run_root),
    }


def train_mode(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve() / "training"
    targets: list[str] = args.targets
    seeds: list[int] = args.seeds
    if not args.dry_run:
        require_data(data_root, list(SOURCES) + targets)

    commands: list[list[str]] = []
    for seed in seeds:
        command = [
            sys.executable,
            str(ROOT / "tools" / "run_senior_exact_paper_table_v5.py"),
            "--dataset-root", str(data_root),
            "--source-graphs", ",".join(SOURCES),
            "--target-graphs", ",".join(targets),
            "--output-dir", str(output_root / f"seed_{seed}"),
            "--device", args.device,
            "--epochs", "10",
            "--dims", "64",
            "--hops", "4",
            "--local-topk", "10",
            "--pos-count", "10",
            "--neg-count", "10",
            "--hidden-dim", "512",
            "--n-layers", "1",
            "--n-heads", "8",
            "--diffusion-T", "200",
            "--lr", "0.0005",
            "--batch-size", "10000",
            "--source-weighting", "step_weighted",
            "--source-repeat-weights", "texas=1,photo=1,dblp=3",
            "--chunk-size", "4096",
            "--t-eval", "50",
            "--eval-before-training",
            "--eval-every", "1",
            "--eval-start-epoch", "1",
            "--allow-unseen-ground-truth-eval",
            "--score-embedding", "guided_refined",
            "--v55-preset", "v574_state_cpic_train_final",
            "--preset-override-json",
            "@" + str((ROOT / "configs" / "final_training_override.json").resolve()),
            "--seed", str(seed),
            "--deterministic",
        ]
        commands.append(command)
        run_command(command, dry_run=args.dry_run)
    return {
        "mode": "train",
        "sources": list(SOURCES),
        "targets": targets,
        "seeds": seeds,
        "commands": commands,
        "output_root": str(output_root),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "pretrained":
        manifest = pretrained_mode(args)
    else:
        manifest = train_mode(args)
    manifest.update(
        {
            "paper_config": str(PAPER_CONFIG),
            "dry_run": bool(args.dry_run),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not args.dry_run:
        output = Path(str(manifest["output_root"]))
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
