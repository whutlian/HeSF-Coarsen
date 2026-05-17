from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import repo_root, run_subprocess_with_log, write_command_metadata, write_config_snapshot
from experiments.scripts.summarize_next9_ogbn_aggregation import summarize_next9_ogbn_aggregation


SIZE_INPUTS = {
    "200k": Path("data/ogbn_mag_subsets_h2_cuda_smoke/subset_200k"),
    "500k": Path("data/ogbn_mag_subsets_next6_20260516/subset_500k"),
    "1m": Path("data/ogbn_mag_subsets_20260516/subset_1m_fullrels"),
    "full-local": Path("data/ogbn_mag_hesf"),
}


METHOD_LAMBDA_SPEC = {
    "HeSF-LVC-P": 0.25,
    "HeSF-LVC-S": 0.5,
}


def _load_base_config(root: Path) -> dict:
    with (root / "configs/paper/ogbn_mag_next9_opt_aggregation.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_name(size: str, method: str) -> str:
    return f"next10_ogbn_aggregation_{size}_{method.replace('-', '_')}"


def _prepare_config(root: Path, size: str, method: str, run_dir: Path, progress: bool) -> dict:
    config = deepcopy(_load_base_config(root))
    config.setdefault("paper", {})["method"] = method
    config.setdefault("paper", {})["variant"] = "next10_aggregation_instrumented"
    config.setdefault("coarsening", {})["max_levels"] = 1
    config.setdefault("coarsening", {})["aggregation_reducer"] = "sort"
    config.setdefault("coarsening", {})["aggregation_chunk_size"] = 1_000_000
    config.setdefault("scoring", {})["lambda_spec"] = METHOD_LAMBDA_SPEC[method]
    config.setdefault("scoring", {})["lambda_conv"] = 0.0
    config.setdefault("scoring", {})["lambda_rel"] = 0.0
    config.setdefault("diagnostics", {})["enable_large_graph_envelope"] = True
    config.setdefault("diagnostics", {})["enable_relation_diagnostics"] = False
    config.setdefault("progress", {})["enabled"] = bool(progress)
    config.setdefault("progress", {})["backend"] = "plain"
    config.setdefault("output", {})["dir"] = str(run_dir)
    config.setdefault("candidates", {})["mmap_dir"] = str(run_dir / "_candidate_mmap")
    config.setdefault("candidates", {})["incident_index_mmap_dir"] = str(run_dir / "_incident_index_mmap")
    return config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=["200k", "500k", "1m", "full-local"])
    parser.add_argument("--methods", nargs="*", default=["HeSF-LVC-P", "HeSF-LVC-S"])
    parser.add_argument("--candidate-mode", default="optimized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input", default="outputs/exp_next8_ogbn_system_scale_20260517_summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    command = (
        "run_next9_ogbn_aggregation_benchmark "
        f"--sizes {' '.join(args.sizes)} --methods {' '.join(args.methods)} "
        f"--candidate-mode {args.candidate_mode} --device {args.device}"
    )
    output = Path(args.output)
    runs_root = output / "runs"
    root = repo_root()
    if not args.summary_only:
        runs_root.mkdir(parents=True, exist_ok=True)
        base_inputs = {size: SIZE_INPUTS[size] for size in args.sizes}
        for method in args.methods:
            if method not in METHOD_LAMBDA_SPEC:
                raise ValueError(f"unsupported OGBN method: {method}")
            for size, graph_dir in base_inputs.items():
                if not (root / graph_dir / "schema.json").exists():
                    raise FileNotFoundError(f"missing OGBN graph for size {size}: {root / graph_dir}")
                run_name = _run_name(size, method)
                run_dir = runs_root / run_name
                config = _prepare_config(root, size, method, run_dir, bool(args.progress))
                write_config_snapshot(run_dir / "config.yaml", config)
                metadata = {
                    "size": size,
                    "method": method,
                    "candidate_mode": args.candidate_mode,
                    "device": args.device,
                    "experiment_block": "next10_ogbn_aggregation_instrumented",
                }
                run_command = [
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
                    run_command.extend(["--progress", "--progress-backend", "plain"])
                write_command_metadata(run_dir, run_name=run_name, command=run_command, status="running", **metadata)
                completed = run_subprocess_with_log(
                    run_command,
                    cwd=root,
                    log_path=run_dir / "coarsen.log",
                    stream_output=bool(args.progress),
                )
                write_command_metadata(
                    run_dir,
                    run_name=run_name,
                    command=run_command,
                    status="success" if completed.returncode == 0 else "failed",
                    returncode=completed.returncode,
                    **metadata,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"OGBN aggregation run failed for {run_name}; see {run_dir / 'coarsen.log'}")
    summarize_next9_ogbn_aggregation(
        input_summary=args.input,
        input_runs=None if args.summary_only else runs_root,
        output=output,
        command_lines=[command],
    )


if __name__ == "__main__":
    main()
