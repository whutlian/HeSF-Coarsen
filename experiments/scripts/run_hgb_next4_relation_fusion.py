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


RELATION_FUSION_VARIANTS = (
    "H2-full",
    "H2-single-relation-sum",
    "H2-no-rel-term",
    "H2-uniform-fused-only",
)


@dataclass(frozen=True)
class RelationFusionConfig:
    run_name: str
    dataset: str
    variant: str
    target_ratio: float
    seed: int
    config: dict
    unique_run_key: str


def _target_ratios(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    if args.target_ratios:
        values.extend(float(value) for value in args.target_ratios)
    if args.target_ratio:
        values.extend(float(value) for value in args.target_ratio)
    return values or [0.5]


def _run_key(dataset: str, variant: str, target_ratio: float, seed: int, max_levels: int, candidate_k: int) -> str:
    text = f"next4-relfusion|{dataset}|{variant}|{target_ratio:.6g}|{seed}|{max_levels}|{candidate_k}"
    return f"next4-relfusion:{dataset}:{variant}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def _make_relation_fusion_config(
    *,
    variant: str,
    target_ratio: float,
    seed: int,
    max_levels: int,
    candidate_source: str,
    candidate_k: int,
) -> dict:
    if variant not in RELATION_FUSION_VARIANTS:
        raise ValueError(f"unsupported relation-fusion variant: {variant}")
    cfg = _make_config(
        variant="H2",
        target_ratio=target_ratio,
        seed=seed,
        max_levels=max_levels,
        candidate_source=candidate_source,
        candidate_k=candidate_k,
    )
    cfg.setdefault("fusion", {})["relation_operator_mode"] = "relationwise"
    cfg.setdefault("scoring", {})["relation_profile_mode"] = "relationwise"
    cfg.setdefault("diagnostics", {})["spectral_relation_detail"] = True

    if variant == "H2-single-relation-sum":
        cfg["fusion"]["relation_operator_mode"] = "single_relation_sum"
        cfg["scoring"]["relation_profile_mode"] = "single_relation_sum"
    elif variant == "H2-no-rel-term":
        cfg["scoring"]["lambda_rel"] = 0.0
        cfg["scoring"].setdefault("relation_guard", {})["enabled"] = False
    elif variant == "H2-uniform-fused-only":
        cfg["scoring"].update(
            {
                "lambda_rel": 0.0,
                "lambda_conv": 0.0,
                "lambda_feat": 0.0,
                "lambda_boundary": 0.0,
            }
        )
        cfg["scoring"].setdefault("relation_guard", {})["enabled"] = False
        cfg["diagnostics"]["spectral_relation_detail"] = False

    return cfg


def generate_relation_fusion_configs(
    *,
    datasets: Iterable[str],
    variants: Iterable[str],
    target_ratios: Iterable[float],
    seeds: Iterable[int],
    max_levels: int,
    candidate_source: str,
    candidate_k: int,
) -> Iterable[RelationFusionConfig]:
    for dataset, variant, target_ratio, seed in product(datasets, variants, target_ratios, seeds):
        cfg = _make_relation_fusion_config(
            variant=str(variant),
            target_ratio=float(target_ratio),
            seed=int(seed),
            max_levels=max_levels,
            candidate_source=candidate_source,
            candidate_k=candidate_k,
        )
        ratio_token = str(float(target_ratio)).replace(".", "p")
        run_name = (
            f"next4_rel_{dataset}_{variant}_r{ratio_token}_L{int(max_levels)}_"
            f"d16_K{int(candidate_k)}_{candidate_source}_seed{int(seed)}"
        )
        yield RelationFusionConfig(
            run_name=run_name,
            dataset=str(dataset),
            variant=str(variant),
            target_ratio=float(target_ratio),
            seed=int(seed),
            config=cfg,
            unique_run_key=_run_key(str(dataset), str(variant), float(target_ratio), int(seed), int(max_levels), int(candidate_k)),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Next4 H2 relation-fusion ablations.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/hgb_next4_relation_fusion"))
    parser.add_argument("--variants", nargs="+", default=list(RELATION_FUSION_VARIANTS))
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_root = args.data_root or args.root
    for item in generate_relation_fusion_configs(
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
            "relation_operator_mode": config["fusion"].get("relation_operator_mode", "relationwise"),
            "relation_profile_mode": config["scoring"].get("relation_profile_mode", "relationwise"),
            "spectral_relation_detail": config["diagnostics"].get("spectral_relation_detail", True),
            "metapath_preset": "off",
            "metapath_operator_weight_total": 0.0,
            "experiment_block": "next4_relation_fusion",
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
