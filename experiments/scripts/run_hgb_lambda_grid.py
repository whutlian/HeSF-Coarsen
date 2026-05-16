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
from experiments.scripts.run_hgb_next4_mainline import _make_config


DEFAULT_LAMBDA_SPECS = (0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_LAMBDA_CONVS = (0.0, 0.25, 0.5, 1.0)


def _apply_source_policy(config: dict, source_policy: str | None) -> dict:
    if source_policy in {None, "", "none"}:
        return config
    if str(source_policy) != "p3-source-aware":
        raise ValueError(f"unsupported source policy: {source_policy}")
    cfg = deepcopy(config)
    candidates = cfg.setdefault("candidates", {})
    candidates["source_policies"] = {
        "bucket": {
            "priority": "high",
            "topk_per_node": 8,
        },
        "onehop": {
            "priority": "medium",
            "topk_per_node": 2,
            "reject_if_delta_spec_above": "bucket_q95",
        },
        "fallback": {
            "max_selected_share": 0.05,
        },
    }
    quotas = candidates.setdefault("quotas", {})
    quotas["enforce_on"] = "selected_matches"
    quotas["fallback_max_fraction"] = 0.05
    return cfg


@dataclass(frozen=True)
class LambdaGridConfig:
    run_name: str
    dataset: str
    variant: str
    target_ratio: float
    seed: int
    lambda_spec: float
    lambda_conv: float
    lambda_rel: float
    config: dict
    unique_run_key: str


def _target_ratios(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    if args.target_ratios:
        values.extend(float(value) for value in args.target_ratios)
    if args.target_ratio:
        values.extend(float(value) for value in args.target_ratio)
    return values or [0.5]


def _token(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def _run_key(
    dataset: str,
    variant: str,
    target_ratio: float,
    seed: int,
    max_levels: int,
    candidate_k: int,
    candidate_source: str,
    lambda_spec: float,
    lambda_conv: float,
    lambda_rel: float,
) -> str:
    text = (
        f"lambda-grid|{dataset}|{variant}|{target_ratio:.6g}|{seed}|{max_levels}|"
        f"{candidate_k}|{candidate_source}|{lambda_spec:.6g}|{lambda_conv:.6g}|{lambda_rel:.6g}"
    )
    return f"lambda-grid:{dataset}:{variant}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def _make_lambda_grid_config(
    *,
    variant: str,
    target_ratio: float,
    seed: int,
    max_levels: int,
    candidate_source: str,
    candidate_k: int,
    lambda_spec: float,
    lambda_conv: float,
    lambda_rel: float,
    twohop_budget_per_node: int = 1,
    twohop_max_time_budget_sec: float | None = None,
) -> dict:
    cfg = _make_config(
        variant=variant,
        target_ratio=target_ratio,
        seed=seed,
        max_levels=max_levels,
        candidate_source=candidate_source,
        candidate_k=candidate_k,
        twohop_budget_per_node=twohop_budget_per_node,
        twohop_max_time_budget_sec=twohop_max_time_budget_sec,
    )
    cfg["scoring"] = dict(cfg.get("scoring", {}))
    cfg["scoring"]["lambda_spec"] = float(lambda_spec)
    cfg["scoring"]["lambda_conv"] = float(lambda_conv)
    cfg["scoring"]["lambda_rel"] = float(lambda_rel)
    if float(lambda_rel) == 0.0:
        cfg["scoring"].setdefault("relation_guard", {})["enabled"] = False
    return cfg


def generate_lambda_grid_configs(
    *,
    datasets: Iterable[str],
    variants: Iterable[str],
    target_ratios: Iterable[float],
    seeds: Iterable[int],
    max_levels: int,
    candidate_source: str,
    candidate_k: int,
    lambda_specs: Iterable[float],
    lambda_convs: Iterable[float],
    lambda_rel: float,
    twohop_budget_per_node: int = 1,
    twohop_max_time_budget_sec: float | None = None,
) -> Iterable[LambdaGridConfig]:
    for dataset, variant, target_ratio, seed, lambda_spec, lambda_conv in product(
        datasets, variants, target_ratios, seeds, lambda_specs, lambda_convs
    ):
        cfg = _make_lambda_grid_config(
            variant=str(variant),
            target_ratio=float(target_ratio),
            seed=int(seed),
            max_levels=max_levels,
            candidate_source=candidate_source,
            candidate_k=candidate_k,
            lambda_spec=float(lambda_spec),
            lambda_conv=float(lambda_conv),
            lambda_rel=float(lambda_rel),
            twohop_budget_per_node=int(twohop_budget_per_node),
            twohop_max_time_budget_sec=twohop_max_time_budget_sec,
        )
        ratio_token = _token(float(target_ratio))
        spec_token = _token(float(lambda_spec))
        conv_token = _token(float(lambda_conv))
        rel_token = _token(float(lambda_rel))
        run_name = (
            f"lambda_grid_{dataset}_{variant}_r{ratio_token}_L{int(max_levels)}_"
            f"ls{spec_token}_lc{conv_token}_lr{rel_token}_K{int(candidate_k)}_"
            f"{candidate_source}_seed{int(seed)}"
        )
        yield LambdaGridConfig(
            run_name=run_name,
            dataset=str(dataset),
            variant=str(variant),
            target_ratio=float(target_ratio),
            seed=int(seed),
            lambda_spec=float(lambda_spec),
            lambda_conv=float(lambda_conv),
            lambda_rel=float(lambda_rel),
            config=cfg,
            unique_run_key=_run_key(
                str(dataset),
                str(variant),
                float(target_ratio),
                int(seed),
                int(max_levels),
                int(candidate_k),
                str(candidate_source),
                float(lambda_spec),
                float(lambda_conv),
                float(lambda_rel),
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HGB H2/H3 lambda_spec x lambda_conv Pareto grid.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/hgb_lambda_grid"))
    parser.add_argument("--variants", nargs="+", default=["H2", "H3"])
    parser.add_argument("--target-ratio", type=float, action="append", dest="target_ratio")
    parser.add_argument("--target-ratios", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--lambda-specs", type=float, nargs="+", default=list(DEFAULT_LAMBDA_SPECS))
    parser.add_argument("--lambda-convs", type=float, nargs="+", default=list(DEFAULT_LAMBDA_CONVS))
    parser.add_argument("--lambda-rel", type=float, default=0.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--candidate-source", default="onehop_twohop_bucket")
    parser.add_argument("--candidate-K", "--candidate-k", type=int, default=8, dest="candidate_K")
    parser.add_argument("--twohop-budget-per-node", type=int, default=1)
    parser.add_argument("--twohop-max-time-budget-sec", type=float)
    parser.add_argument("--source-policy", choices=["none", "p3-source-aware"], default="none")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-backend", choices=["auto", "plain", "tqdm"], default="plain")
    parser.add_argument("--progress-interval", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_root = args.data_root or args.root
    for item in generate_lambda_grid_configs(
        datasets=args.datasets,
        variants=args.variants,
        target_ratios=_target_ratios(args),
        seeds=args.seeds,
        max_levels=args.max_levels,
        candidate_source=args.candidate_source,
        candidate_k=args.candidate_K,
        lambda_specs=args.lambda_specs,
        lambda_convs=args.lambda_convs,
        lambda_rel=args.lambda_rel,
        twohop_budget_per_node=int(args.twohop_budget_per_node),
        twohop_max_time_budget_sec=args.twohop_max_time_budget_sec,
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
        config = _apply_source_policy(deepcopy(item.config), args.source_policy)
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
            "lambda_spec": item.lambda_spec,
            "lambda_conv": item.lambda_conv,
            "lambda_rel": item.lambda_rel,
            "sketch_dim": 16,
            "sketch_order": 5,
            "sketch_method": "chebyshev_heat",
            "matching_method": config["coarsening"]["matching_method"],
            "max_cluster_size": config["coarsening"]["max_cluster_size"],
            "relation_weighting_method": "uniform",
            "metapath_preset": "off",
            "metapath_operator_weight_total": 0.0,
            "experiment_block": "lambda_grid",
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
