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


DEFAULT_VARIANTS = ("H2", "H3", "H4")


@dataclass(frozen=True)
class OGBNMediumConfig:
    run_name: str
    variant: str
    seed: int
    config: dict
    unique_run_key: str


def _variant_settings(variant: str) -> dict[str, float | int | str]:
    settings: dict[str, dict[str, float | int | str]] = {
        "H2": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.5, "lambda_spec": 1.0},
        "H3": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.35, "lambda_spec": 1.0},
        "H4": {"matching_method": "greedy_cluster", "max_cluster_size": 4, "lambda_conv": 0.0, "lambda_spec": 1.0},
    }
    if variant not in settings:
        raise ValueError(f"unsupported OGBN-MAG medium variant: {variant}")
    return settings[variant]


def _run_key(variant: str, seed: int, target_ratio: float, optimized_candidates: bool) -> str:
    text = f"ogbn-mag-medium|{variant}|{seed}|{target_ratio:.6g}|{optimized_candidates}"
    return f"ogbn-medium:{variant}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def _make_config(
    *,
    run_dir: Path,
    variant: str,
    seed: int,
    target_ratio: float,
    max_levels: int,
    optimized_candidates: bool,
    device: str,
) -> dict:
    settings = _variant_settings(str(variant))
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["seed"] = int(seed)
    cfg["acceleration"] = dict(
        cfg["acceleration"],
        dense_backend="torch" if str(device).startswith("cuda") else "numpy",
        device=str(device),
        fallback_to_numpy=True,
        max_dense_bytes=12_000_000_000,
    )
    cfg["progress"] = dict(cfg["progress"], enabled=True, backend="plain", min_interval_seconds=5.0)
    cfg["coarsening"] = dict(
        cfg["coarsening"],
        target_ratio=float(target_ratio),
        max_levels=int(max_levels),
        matching_method=str(settings["matching_method"]),
        max_cluster_size=int(settings["max_cluster_size"]),
        same_type_only=True,
        same_partition_only=True,
    )
    cfg["sketch"] = dict(
        cfg["sketch"],
        method="chebyshev_heat",
        dim=16,
        order=5,
        dtype="float16",
        row_normalize=True,
    )
    cfg["fusion"] = dict(cfg.get("fusion", {}))
    cfg["fusion"]["relation_operator_mode"] = "relationwise"
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
    cfg["scoring"] = dict(
        cfg["scoring"],
        normalization="p95",
        normalization_scope="level",
        lambda_spec=float(settings["lambda_spec"]),
        lambda_conv=float(settings["lambda_conv"]),
        relation_profile_mode="relationwise",
    )
    cfg["candidates"] = dict(
        cfg["candidates"],
        store_backend="array",
        use_chunked_generation=True,
        total_budget_K=8,
        twohop_budget_K2=4,
        edge_chunk_size=1_000_000,
        middle_chunk_size=100_000,
        node_chunk_size=500_000,
        mmap_dir=str(run_dir / "candidate_mmap"),
        incident_index_mmap_dir=None if optimized_candidates else str(run_dir / "incident_index_mmap"),
        enable_onehop=True,
        enable_capped_twohop=not bool(optimized_candidates),
        enable_bucket=True,
        enable_partition_ann=False,
        enable_fallback=True,
        fallback_penalty=1.0e6,
        fallback_max_fraction=0.05,
        per_middle_pair_cap=64,
        bucket_pair_cap=64,
        simhash_bits=16,
    )
    cfg["features"] = dict(
        cfg["features"],
        projection_mmap_dir=str(run_dir / "projected_features"),
        projection_chunk_size=100_000,
        projection_dtype="float16",
    )
    cfg["diagnostics"] = dict(
        cfg["diagnostics"],
        enable_large_graph_envelope=True,
        enable_spectral=False,
        spectral_relation_detail=False,
        spectral_baselines=[],
        cumulative_spectral_baselines=[],
        spectral_baseline_max_nodes=1,
        spectral_exact_eigenvalue_max_nodes=0,
        cumulative_spectral_exact_eigenvalue_max_nodes=0,
    )
    cfg["output"] = {"dir": str(run_dir)}
    return cfg


def generate_configs(
    *,
    output: Path,
    variants: Iterable[str],
    seeds: Iterable[int],
    target_ratio: float,
    max_levels: int,
    optimized_candidates: bool,
    device: str,
) -> Iterable[OGBNMediumConfig]:
    mode = "optimized" if optimized_candidates else "full_candidates"
    for variant, seed in product(variants, seeds):
        run_name = f"ogbn_mag_medium_{variant}_r{str(float(target_ratio)).replace('.', 'p')}_{mode}_seed{int(seed)}"
        run_dir = output / run_name
        yield OGBNMediumConfig(
            run_name=run_name,
            variant=str(variant),
            seed=int(seed),
            config=_make_config(
                run_dir=run_dir,
                variant=str(variant),
                seed=int(seed),
                target_ratio=float(target_ratio),
                max_levels=int(max_levels),
                optimized_candidates=bool(optimized_candidates),
                device=str(device),
            ),
            unique_run_key=_run_key(str(variant), int(seed), float(target_ratio), bool(optimized_candidates)),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Next4 H2/H3/H4 on a medium OGBN-MAG subset.")
    parser.add_argument("--input", type=Path, default=Path("data/ogbn_mag_subsets_h2_cuda_smoke/subset_200k"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/ogbn_mag_next4_medium"))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345])
    parser.add_argument("--target-ratio", type=float, default=0.5)
    parser.add_argument("--max-levels", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--optimized-candidates", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    rows: list[dict[str, object]] = []
    for item in generate_configs(
        output=args.output,
        variants=args.variants,
        seeds=args.seeds,
        target_ratio=float(args.target_ratio),
        max_levels=int(args.max_levels),
        optimized_candidates=bool(args.optimized_candidates),
        device=str(args.device),
    ):
        run_dir = args.output / item.run_name
        config = deepcopy(item.config)
        config.setdefault("resume", {})["enabled"] = bool(args.resume)
        write_config_snapshot(run_dir / "config.yaml", config)
        command = [
            args.python,
            "-m",
            "hesf_coarsen.cli.main",
            "coarsen",
            "--config",
            str(run_dir / "config.yaml"),
            "--input",
            str(args.input),
            "--output",
            str(run_dir),
            "--progress",
            "--progress-backend",
            "plain",
        ]
        if args.resume:
            command.append("--resume")
        metadata = {
            "dataset": "OGBN-MAG-subset",
            "variant": item.variant,
            "seed": item.seed,
            "target_ratio": float(args.target_ratio),
            "experiment_block": "ogbn_mag_next4_medium",
            "input_graph_dir": str(args.input),
            "candidate_source": "onehop_bucket_fallback" if args.optimized_candidates else "onehop_twohop_bucket",
            "optimized_candidates": bool(args.optimized_candidates),
            "compute_device": str(args.device),
            "unique_run_key": item.unique_run_key,
        }
        write_command_metadata(run_dir, run_name=item.run_name, command=command, status="created", **metadata)
        if args.dry_run:
            rows.append({"run_name": item.run_name, "status": "created", **metadata})
            continue
        completed = run_subprocess_with_log(
            command,
            cwd=root,
            log_path=run_dir / "run.log",
            stream_output=True,
        )
        status = "success" if completed.returncode == 0 else "failed"
        write_command_metadata(
            run_dir,
            run_name=item.run_name,
            command=command,
            status=status,
            returncode=completed.returncode,
            **metadata,
        )
        rows.append({"run_name": item.run_name, "status": status, "returncode": completed.returncode, **metadata})
    write_command_metadata(args.output, run_name=args.output.name, status="success", experiment_block="ogbn_mag_next4_medium")
    return 0 if all(row.get("status") != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
