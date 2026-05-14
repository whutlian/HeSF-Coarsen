from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class StageBAblationConfig:
    run_name: str
    dataset: str
    variant: str
    target_ratio: float
    seed: int
    candidate_source: str
    sketch_dim: int
    sketch_order: int
    config: dict


def _candidate_flags(source: str) -> dict:
    return {
        "enable_onehop": "onehop" in source,
        "enable_capped_twohop": "twohop" in source,
        "enable_bucket": "bucket" in source,
        "enable_partition_ann": source.endswith("_ann") or "ann" in source,
    }


def _variant_config(config: dict, variant: str) -> dict:
    cfg = deepcopy(config)
    if variant == "base":
        return cfg
    if variant == "uniform_weight":
        cfg.setdefault("fusion", {}).setdefault("relation_weighting", {})["method"] = "uniform"
        cfg.setdefault("metapath_sketch", {}).setdefault("weighting", {})["method"] = "uniform"
        return cfg
    if variant == "no_metapath":
        cfg.setdefault("metapath_sketch", {})["enabled"] = False
        cfg.setdefault("metapath_sketch", {})["operator_weight_total"] = 0.0
        return cfg
    if variant == "lazy_no_metapath":
        cfg.setdefault("sketch", {})["method"] = "lazy"
        cfg.setdefault("metapath_sketch", {})["enabled"] = False
        cfg.setdefault("metapath_sketch", {})["operator_weight_total"] = 0.0
        return cfg
    if variant == "no_conv":
        cfg.setdefault("scoring", {})["lambda_conv"] = 0.0
        return cfg
    raise ValueError(f"unsupported Stage B variant: {variant}")


def generate_stage_b_configs(
    *,
    datasets: Iterable[str],
    target_ratios: Iterable[float],
    max_levels: int,
    candidate_sources: Iterable[str],
    candidate_k: int,
    sketch_dims: Iterable[int],
    sketch_orders: Iterable[int],
    seeds: Iterable[int],
    variants: Iterable[str],
    normalization: str = "p95",
    normalization_scope: str = "level",
) -> Iterable[StageBAblationConfig]:
    for dataset, target_ratio, source, sketch_dim, sketch_order, seed, variant in product(
        datasets,
        target_ratios,
        candidate_sources,
        sketch_dims,
        sketch_orders,
        seeds,
        variants,
    ):
        config = deepcopy(DEFAULT_CONFIG)
        config["seed"] = int(seed)
        config["coarsening"] = dict(
            config["coarsening"],
            target_ratio=float(target_ratio),
            max_levels=int(max_levels),
            matching_method="mutual_best",
        )
        config["sketch"] = dict(
            config["sketch"],
            dim=int(sketch_dim),
            order=int(sketch_order),
            method="chebyshev_heat",
        )
        config["candidates"] = dict(
            config["candidates"],
            total_budget_K=int(candidate_k),
            twohop_budget_K2=max(1, int(candidate_k) // 2),
            ann_budget_K=int(candidate_k),
            enable_fallback=True,
            fallback_penalty=1.0e6,
            fallback_max_fraction=0.05,
            **_candidate_flags(source),
        )
        config["scoring"] = dict(
            config["scoring"],
            normalization=str(normalization),
            normalization_scope=str(normalization_scope),
            lambda_spec=1.0,
            lambda_rel=0.5,
            lambda_feat=0.2,
            lambda_conv=0.5,
            lambda_boundary=0.2,
        )
        config["diagnostics"] = dict(config["diagnostics"], enable_large_graph_envelope=True)
        config = _variant_config(config, variant)
        ratio_token = str(float(target_ratio)).replace(".", "p")
        run_name = (
            f"stageB_{dataset}_{variant}_r{ratio_token}_L{int(max_levels)}_"
            f"d{int(sketch_dim)}_K{int(candidate_k)}_{source}_seed{int(seed)}"
        )
        yield StageBAblationConfig(
            run_name=run_name,
            dataset=dataset,
            variant=variant,
            target_ratio=float(target_ratio),
            seed=int(seed),
            candidate_source=source,
            sketch_dim=int(sketch_dim),
            sketch_order=int(sketch_order),
            config=config,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused Stage B HGB ablations.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/hgb_stageB"))
    parser.add_argument("--target-ratio", type=float, action="append", dest="target_ratio")
    parser.add_argument("--target-ratios", type=float, nargs="+", default=None)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--candidate-source", default="onehop_twohop_bucket")
    parser.add_argument("--candidate-sources", nargs="+", default=None)
    parser.add_argument("--candidate-K", "--candidate-k", type=int, default=8, dest="candidate_K")
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--sketch-dims", type=int, nargs="+", default=None)
    parser.add_argument("--sketch-order", type=int, default=5)
    parser.add_argument("--sketch-orders", type=int, nargs="+", default=None)
    parser.add_argument("--normalization", default="p95")
    parser.add_argument("--normalization-scope", default="level")
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "uniform_weight", "no_metapath", "lazy_no_metapath", "no_conv"],
    )
    parser.add_argument("--python", default=sys.executable)
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


def _candidate_sources(args: argparse.Namespace) -> list[str]:
    return list(args.candidate_sources or [args.candidate_source])


def _sketch_dims(args: argparse.Namespace) -> list[int]:
    return [int(value) for value in (args.sketch_dims or [args.sketch_dim])]


def _sketch_orders(args: argparse.Namespace) -> list[int]:
    return [int(value) for value in (args.sketch_orders or [args.sketch_order])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_root = args.data_root or args.root
    for item in generate_stage_b_configs(
        datasets=args.datasets,
        target_ratios=_target_ratios(args),
        max_levels=args.max_levels,
        candidate_sources=_candidate_sources(args),
        candidate_k=args.candidate_K,
        sketch_dims=_sketch_dims(args),
        sketch_orders=_sketch_orders(args),
        seeds=args.seeds,
        variants=args.variants,
        normalization=args.normalization,
        normalization_scope=args.normalization_scope,
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
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            dataset=item.dataset,
            variant=item.variant,
            target_ratio=item.target_ratio,
            seed=item.seed,
            candidate_source=item.candidate_source,
            sketch_dim=item.sketch_dim,
            sketch_order=item.sketch_order,
            status="created",
        )
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
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            dataset=item.dataset,
            variant=item.variant,
            target_ratio=item.target_ratio,
            seed=item.seed,
            candidate_source=item.candidate_source,
            sketch_dim=item.sketch_dim,
            sketch_order=item.sketch_order,
            command=command,
            status="running",
        )
        completed = run_subprocess_with_log(
            command,
            cwd=root,
            log_path=run_dir / "run.log",
            stream_output=args.progress,
        )
        status = "success" if completed.returncode == 0 else "failed"
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            dataset=item.dataset,
            variant=item.variant,
            target_ratio=item.target_ratio,
            seed=item.seed,
            candidate_source=item.candidate_source,
            sketch_dim=item.sketch_dim,
            sketch_order=item.sketch_order,
            command=command,
            status=status,
            returncode=completed.returncode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
