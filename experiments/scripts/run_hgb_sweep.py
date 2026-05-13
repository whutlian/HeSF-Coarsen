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

from experiments.scripts._common import repo_root, run_subprocess_with_log, write_command_metadata, write_config_snapshot
from hesf_coarsen.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class HgbSweepConfig:
    run_name: str
    dataset: str
    target_ratio: float
    max_levels: int
    sketch_dim: int
    candidate_K: int
    candidate_sources: str
    config: dict


def _candidate_flags(source: str) -> dict:
    return {
        "enable_onehop": True,
        "enable_capped_twohop": "twohop" in source,
        "enable_bucket": "bucket" in source,
        "enable_partition_ann": source.endswith("_ann"),
    }


def generate_hgb_sweep_configs(datasets: Iterable[str]) -> Iterable[HgbSweepConfig]:
    sources = [
        "onehop_only",
        "onehop_bucket",
        "onehop_twohop_bucket",
        "onehop_twohop_bucket_ann",
    ]
    for dataset, target_ratio, max_levels, sketch_dim, candidate_k, source in product(
        datasets,
        [0.5, 0.25],
        [1, 2],
        [16, 32],
        [8, 16],
        sources,
    ):
        config = deepcopy(DEFAULT_CONFIG)
        config["coarsening"] = dict(config["coarsening"], target_ratio=target_ratio, max_levels=max_levels)
        config["sketch"] = dict(config["sketch"], dim=sketch_dim)
        config["candidates"] = dict(
            config["candidates"],
            total_budget_K=candidate_k,
            twohop_budget_K2=max(1, candidate_k // 2),
            ann_budget_K=candidate_k,
            **_candidate_flags(source),
        )
        run_name = (
            f"hgb_{dataset}_r{str(target_ratio).replace('.', 'p')}_"
            f"L{max_levels}_d{sketch_dim}_K{candidate_k}_{source}"
        )
        yield HgbSweepConfig(run_name, dataset, target_ratio, max_levels, sketch_dim, candidate_k, source, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HGB sweep experiments.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root used by the plan commands.")
    parser.add_argument("--data-root", type=Path, help="Optional raw HGB root override.")
    parser.add_argument("--graph-root", type=Path, help="Optional converted graph root override.")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/hgb_sweep"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--progress", action="store_true", help="emit live progress from each coarsening run")
    parser.add_argument("--progress-backend", choices=["auto", "plain", "tqdm"], default="plain")
    parser.add_argument("--progress-interval", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_root = args.data_root or args.root
    for item in generate_hgb_sweep_configs(args.datasets):
        graph_dir = (args.graph_root / item.dataset.lower()) if args.graph_root else (args.root / f"{item.dataset.lower()}_hesf")
        if not (graph_dir / "schema.json").exists() and not args.dry_run:
            run_subprocess_with_log(
                [args.python, "-m", "hesf_coarsen.cli.main", "import-hgb", "--name", item.dataset, "--root", str(raw_root), "--output", str(graph_dir)],
                cwd=root,
                log_path=args.output / "_imports" / f"{item.dataset}.log",
                stream_output=args.progress,
            )
        run_dir = args.output / item.run_name
        config = deepcopy(item.config)
        if args.progress:
            config.setdefault("progress", {})["enabled"] = True
        if args.progress_backend is not None:
            config.setdefault("progress", {})["backend"] = args.progress_backend
        if args.progress_interval is not None:
            config.setdefault("progress", {})["min_interval_seconds"] = args.progress_interval
        config["output"] = {"dir": str(run_dir)}
        write_config_snapshot(run_dir / "config.yaml", config)
        write_command_metadata(run_dir, run_name=item.run_name, dataset=item.dataset, status="created")
        if args.dry_run:
            continue
        command = [args.python, "-m", "hesf_coarsen.cli.main", "coarsen", "--config", str(run_dir / "config.yaml"), "--input", str(graph_dir), "--output", str(run_dir)]
        if args.progress:
            command.extend(["--progress", "--progress-backend", args.progress_backend])
            if args.progress_interval is not None:
                command.extend(["--progress-interval", str(args.progress_interval)])
        completed = run_subprocess_with_log(
            command,
            cwd=root,
            log_path=run_dir / "run.log",
            stream_output=args.progress,
        )
        status = "success" if completed.returncode == 0 else "failed"
        write_command_metadata(run_dir, run_name=item.run_name, dataset=item.dataset, command=command, status=status, returncode=completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
