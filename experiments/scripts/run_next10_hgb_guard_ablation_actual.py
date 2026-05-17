from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import repo_root, run_subprocess_with_log, write_command_metadata, write_config_snapshot
from experiments.scripts.summarize_next9_hgb_guard_ablation import summarize_next9_hgb_guard_ablation


VARIANT_CONFIGS: Mapping[str, tuple[str, str]] = {
    "P_baseline": ("configs/paper/hgb_hesf_lvc_p.yaml", "HeSF-LVC-P"),
    "P_spectral_guard": ("configs/paper/hgb_hesf_lvc_p_spectral_guard.yaml", "HeSF-LVC-P"),
    "P_source_aware_auto": ("configs/paper/hgb_hesf_lvc_p_sourceaware_auto.yaml", "HeSF-LVC-P"),
    "P_spectral_guard_plus_source_aware_auto": (
        "configs/paper/hgb_hesf_lvc_p_spectral_guard_plus_sourceaware_auto.yaml",
        "HeSF-LVC-P",
    ),
    "S_baseline": ("configs/paper/hgb_hesf_lvc_s.yaml", "HeSF-LVC-S"),
    "S_spectral_guard": ("configs/paper/hgb_hesf_lvc_s_spectral_guard.yaml", "HeSF-LVC-S"),
    "S_source_aware_auto": ("configs/paper/hgb_hesf_lvc_s_sourceaware_auto.yaml", "HeSF-LVC-S"),
    "S_spectral_guard_plus_source_aware_auto": (
        "configs/paper/hgb_hesf_lvc_s_spectral_guard_plus_sourceaware_auto.yaml",
        "HeSF-LVC-S",
    ),
    "flatten-sum": ("configs/paper/hgb_flatten_sum.yaml", "flatten-sum"),
    "H6-no-spec": ("configs/paper/hgb_h6_no_spec.yaml", "H6-no-spec"),
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_name(dataset: str, variant: str, seed: int) -> str:
    safe_variant = variant.replace("-", "_")
    return f"next10_guard_{dataset}_{safe_variant}_seed{seed}"


def _graph_dir(graph_root: Path, dataset: str) -> Path:
    return graph_root / f"{dataset.lower()}_hesf"


def _prepare_config(root: Path, variant: str, dataset: str, seed: int, run_dir: Path, progress: bool) -> tuple[dict, str]:
    config_path, method = VARIANT_CONFIGS[variant]
    config = deepcopy(_load_yaml(root / config_path))
    config["seed"] = int(seed)
    config.setdefault("output", {})["dir"] = str(run_dir)
    config.setdefault("diagnostics", {})["spectral_relation_detail"] = True
    config.setdefault("diagnostics", {})["enable_relation_diagnostics"] = True
    config.setdefault("diagnostics", {})["enable_large_graph_envelope"] = True
    config.setdefault("progress", {})["enabled"] = bool(progress)
    config.setdefault("progress", {})["backend"] = "plain"
    config.setdefault("paper", {})["method"] = method
    config.setdefault("paper", {})["variant"] = variant
    return config, method


def run_guard_ablation(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[str],
    graph_root: Path,
    output: Path,
    python: str,
    device: str,
    progress: bool,
    task_epochs: int,
    task_refine_epochs: Sequence[int],
) -> None:
    root = repo_root()
    runs_root = output / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"unsupported guard ablation variant: {variant}")
        for dataset in datasets:
            graph_dir = _graph_dir(graph_root, dataset)
            if not (graph_dir / "schema.json").exists():
                raise FileNotFoundError(f"missing graph directory: {graph_dir}")
            for seed in seeds:
                run_name = _run_name(dataset, variant, int(seed))
                run_dir = runs_root / run_name
                config, method = _prepare_config(root, variant, dataset, int(seed), run_dir, progress)
                write_config_snapshot(run_dir / "config.yaml", config)
                metadata = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": variant,
                    "method": method,
                    "target_ratio": config.get("coarsening", {}).get("target_ratio", 0.5),
                    "experiment_block": "next10_guard_ablation_actual",
                    "size": "hgb",
                }
                command = [
                    python,
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
                if progress:
                    command.extend(["--progress", "--progress-backend", "plain"])
                write_command_metadata(run_dir, run_name=run_name, command=command, status="running", **metadata)
                completed = run_subprocess_with_log(
                    command,
                    cwd=root,
                    log_path=run_dir / "coarsen.log",
                    stream_output=progress,
                )
                write_command_metadata(
                    run_dir,
                    run_name=run_name,
                    command=command,
                    status="success" if completed.returncode == 0 else "failed",
                    returncode=completed.returncode,
                    **metadata,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"coarsening failed for {run_name}; see {run_dir / 'coarsen.log'}")

    task_output = output / "task_eval_summary.csv"
    task_command = [
        python,
        "-m",
        "experiments.scripts.run_hgb_task_eval",
        "--runs-root",
        str(runs_root),
        "--graph-root",
        str(graph_root),
        "--datasets",
        *datasets,
        "--epochs",
        str(task_epochs),
        "--refine-epochs-list",
        *[str(value) for value in task_refine_epochs],
        "--hidden-dim",
        "32",
        "--device",
        device,
        "--output",
        str(task_output),
    ]
    if progress:
        task_command.append("--progress")
    completed = run_subprocess_with_log(
        task_command,
        cwd=root,
        log_path=output / "task_eval.log",
        stream_output=progress,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"task evaluation failed; see {output / 'task_eval.log'}")

    summary_dir = output / "actual_summary"
    summary_command = [
        python,
        "-m",
        "experiments.scripts.summarize_experiments",
        str(runs_root),
        "--output",
        str(summary_dir),
    ]
    completed = run_subprocess_with_log(
        summary_command,
        cwd=root,
        log_path=output / "summarize_experiments.log",
        stream_output=progress,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"experiment summary failed; see {output / 'summarize_experiments.log'}")
    summarize_next9_hgb_guard_ablation(
        actual_summary=summary_dir,
        output=output / "summary",
        command_lines=[" ".join(summary_command)],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CONFIGS))
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-epochs", type=int, default=20)
    parser.add_argument("--task-refine-epochs", type=int, nargs="+", default=[0, 1, 3, 5])
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    run_guard_ablation(
        datasets=args.datasets,
        seeds=args.seeds,
        variants=args.variants,
        graph_root=args.graph_root,
        output=args.output,
        python=args.python,
        device=args.device,
        progress=bool(args.progress),
        task_epochs=int(args.task_epochs),
        task_refine_epochs=args.task_refine_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
