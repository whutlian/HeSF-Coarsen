from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import (
    repo_root,
    run_subprocess_with_log,
    write_command_metadata,
    write_config_snapshot,
)
from hesf_coarsen.config import DEFAULT_CONFIG


NEXT4_VARIANTS = ("H0", "H1", "H2", "H3", "H4", "H5", "H6")
NEXT4_AGGRESSIVE_VARIANTS = ("A0", "A1", "A2", "A3", "A4")


@dataclass(frozen=True)
class Next4Config:
    run_name: str
    dataset: str
    variant: str
    target_ratio: float
    seed: int
    config: dict
    unique_run_key: str


def _candidate_flags(source: str) -> dict:
    return {
        "enable_onehop": "onehop" in source,
        "enable_capped_twohop": "twohop" in source,
        "enable_bucket": "bucket" in source,
        "enable_partition_ann": source.endswith("_ann") or "ann" in source,
    }


def _variant_settings(variant: str) -> dict[str, float | int | str]:
    settings: dict[str, dict[str, float | int | str]] = {
        "H0": {"matching_method": "mutual_best", "max_cluster_size": 2, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "H1": {"matching_method": "greedy_cluster", "max_cluster_size": 3, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "H2": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "H3": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.35, "lambda_spec": 1.0},
        "H4": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.0, "lambda_spec": 1.0},
        "H5": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.0, "lambda_spec": 1.0},
        "H6": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 0.0},
        "A0": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "A1": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "A2": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "A3": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "A4": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
    }
    if variant not in settings:
        raise ValueError(f"unsupported Next4 variant: {variant}")
    return settings[variant]


def _terminal_guard_settings(variant: str) -> dict:
    base = {
        "enabled": False,
        "protect_hubs": False,
        "protect_rare_relation_carriers": False,
        "protect_boundary_nodes": False,
        "protect_train_label_conflict_nodes": False,
        "hub_degree_percentile": 95,
        "rare_relation_min_count": 1,
        "label_entropy_threshold": 0.0,
        "max_terminal_cluster_size": 2,
    }
    if variant == "A1":
        base.update({"enabled": True, "protect_hubs": True})
    elif variant == "A2":
        base.update({"enabled": True, "protect_rare_relation_carriers": True})
    elif variant == "A3":
        base.update({"enabled": True, "protect_train_label_conflict_nodes": True})
    elif variant == "A4":
        base.update(
            {
                "enabled": True,
                "protect_hubs": True,
                "protect_rare_relation_carriers": True,
                "protect_boundary_nodes": True,
                "protect_train_label_conflict_nodes": True,
            }
        )
    return base


def _run_key(dataset: str, variant: str, target_ratio: float, seed: int, max_levels: int, candidate_k: int) -> str:
    text = f"next4|{dataset}|{variant}|{target_ratio:.6g}|{seed}|{max_levels}|{candidate_k}"
    return f"next4:{dataset}:{variant}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def _make_config(
    *,
    variant: str,
    target_ratio: float,
    seed: int,
    max_levels: int,
    candidate_source: str,
    candidate_k: int,
) -> dict:
    settings = _variant_settings(variant)
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["seed"] = int(seed)
    cfg["coarsening"] = dict(
        cfg["coarsening"],
        target_ratio=float(target_ratio),
        max_levels=int(max_levels),
        matching_method=str(settings["matching_method"]),
        max_cluster_size=int(settings["max_cluster_size"]),
    )
    if str(variant).startswith("A"):
        cfg["coarsening"]["terminal_guard"] = _terminal_guard_settings(str(variant))
    cfg["sketch"] = dict(
        cfg["sketch"],
        method="chebyshev_heat",
        dim=16,
        order=5,
        dtype="float16",
        row_normalize=True,
    )
    cfg["fusion"] = dict(cfg.get("fusion", {}))
    cfg["fusion"]["relation_weighting"] = dict(
        cfg["fusion"].get("relation_weighting", {}),
        method="uniform",
    )
    cfg["metapath_sketch"] = dict(
        cfg.get("metapath_sketch", {}),
        enabled=False,
        preset="off",
        operator_weight_total=0.0,
        paths=[],
        auto_paths=False,
    )
    cfg["candidates"] = dict(
        cfg["candidates"],
        total_budget_K=int(candidate_k),
        twohop_budget_K2=max(1, int(candidate_k) // 2),
        ann_budget_K=int(candidate_k),
        enable_fallback=True,
        fallback_penalty=1.0e6,
        fallback_max_fraction=0.05,
        **_candidate_flags(candidate_source),
    )
    cfg["scoring"] = dict(
        cfg["scoring"],
        normalization="p95",
        normalization_scope="level",
        lambda_spec=float(settings["lambda_spec"]),
        lambda_conv=float(settings["lambda_conv"]),
    )
    return cfg


def generate_next4_configs(
    *,
    datasets: Iterable[str],
    variants: Iterable[str],
    target_ratios: Iterable[float],
    seeds: Iterable[int],
    max_levels: int,
    candidate_source: str,
    candidate_k: int,
) -> Iterable[Next4Config]:
    for dataset, variant, target_ratio, seed in product(datasets, variants, target_ratios, seeds):
        cfg = _make_config(
            variant=str(variant),
            target_ratio=float(target_ratio),
            seed=int(seed),
            max_levels=max_levels,
            candidate_source=candidate_source,
            candidate_k=candidate_k,
        )
        ratio_token = str(float(target_ratio)).replace(".", "p")
        run_name = (
            f"next4_{dataset}_{variant}_r{ratio_token}_L{int(max_levels)}_"
            f"d16_K{int(candidate_k)}_{candidate_source}_seed{int(seed)}"
        )
        yield Next4Config(
            run_name=run_name,
            dataset=str(dataset),
            variant=str(variant),
            target_ratio=float(target_ratio),
            seed=int(seed),
            config=cfg,
            unique_run_key=_run_key(str(dataset), str(variant), float(target_ratio), int(seed), int(max_levels), int(candidate_k)),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Next4 HGB mainline variants H0-H6 and aggressive A0-A4.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/hgb_next4"))
    parser.add_argument("--variants", nargs="+", default=list(NEXT4_VARIANTS))
    parser.add_argument("--target-ratio", type=float, action="append", dest="target_ratio")
    parser.add_argument("--target-ratios", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--candidate-source", default="onehop_twohop_bucket")
    parser.add_argument("--candidate-K", "--candidate-k", type=int, default=8, dest="candidate_K")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-backend", choices=["auto", "plain", "tqdm"], default="plain")
    parser.add_argument("--progress-interval", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _target_ratios(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    if args.target_ratios:
        values.extend(float(value) for value in args.target_ratios)
    if args.target_ratio:
        values.extend(float(value) for value in args.target_ratio)
    return values or [0.5]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_root = args.data_root or args.root
    for item in generate_next4_configs(
        datasets=args.datasets,
        variants=args.variants,
        target_ratios=_target_ratios(args),
        seeds=args.seeds,
        max_levels=args.max_levels,
        candidate_source=args.candidate_source,
        candidate_k=args.candidate_K,
    ):
        graph_dir = (
            (args.graph_root / item.dataset.lower())
            if args.graph_root
            else (args.root / f"{item.dataset.lower()}_hesf")
        )
        if not (graph_dir / "schema.json").exists() and not args.dry_run:
            run_subprocess_with_log(
                [
                    args.python,
                    "-m",
                    "hesf_coarsen.cli.main",
                    "import-hgb",
                    "--name",
                    item.dataset,
                    "--root",
                    str(raw_root),
                    "--output",
                    str(graph_dir),
                ],
                cwd=root,
                log_path=args.output / "_imports" / f"{item.dataset}.log",
                stream_output=args.progress,
            )
        run_dir = args.output / item.run_name
        config = deepcopy(item.config)
        if args.progress:
            config.setdefault("progress", {})["enabled"] = True
        config.setdefault("progress", {})["backend"] = args.progress_backend
        if args.progress_interval is not None:
            config.setdefault("progress", {})["min_interval_seconds"] = args.progress_interval
        config["output"] = {"dir": str(run_dir)}
        write_config_snapshot(run_dir / "config.yaml", config)
        metadata = {
            "dataset": item.dataset,
            "variant": item.variant,
            "target_ratio": item.target_ratio,
            "seed": item.seed,
            "candidate_source": args.candidate_source,
            "sketch_dim": 16,
            "sketch_order": 5,
            "sketch_method": "chebyshev_heat",
            "matching_method": config["coarsening"]["matching_method"],
            "max_cluster_size": config["coarsening"]["max_cluster_size"],
            "relation_weighting_method": "uniform",
            "metapath_preset": "off",
            "metapath_operator_weight_total": 0.0,
            "experiment_block": "next4_mainline",
            "unique_run_key": item.unique_run_key,
        }
        write_command_metadata(run_dir, run_name=item.run_name, status="created", **metadata)
        if args.dry_run:
            continue
        command = [
            args.python,
            "-m",
            "hesf_coarsen.cli.main",
            "coarsen",
            "--config",
            str(run_dir / "config.yaml"),
            "--input",
            str(graph_dir),
            "--output",
            str(run_dir),
        ]
        if args.progress:
            command.extend(["--progress", "--progress-backend", args.progress_backend])
            if args.progress_interval is not None:
                command.extend(["--progress-interval", str(args.progress_interval)])
        write_command_metadata(run_dir, run_name=item.run_name, command=command, status="running", **metadata)
        completed = run_subprocess_with_log(
            command,
            cwd=root,
            log_path=run_dir / "run.log",
            stream_output=args.progress,
        )
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            command=command,
            status="success" if completed.returncode == 0 else "failed",
            returncode=completed.returncode,
            **metadata,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
