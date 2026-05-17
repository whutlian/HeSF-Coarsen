from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.run_next13_metapath_mass import _coarse_for_next13
from experiments.scripts.summarize_next13_structure_critical_tasks import summarize_next13_structure_critical_tasks
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.structure_tasks import evaluate_feature_free_label_propagation, evaluate_lowpass_signal_reconstruction
from hesf_coarsen.io.edge_list import load_graph


def run_next13_structure_critical_tasks(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    tasks: Sequence[str],
    graph_root: Path,
    resource_runs: Path,
    guard_runs: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        original = load_graph(graph_root / f"{dataset.lower()}_hesf")
        for seed in seeds:
            for method in methods:
                try:
                    coarse, assignment, source = _coarse_for_next13(method, dataset, int(seed), original, resource_runs, guard_runs)
                    if not isinstance(assignment, Assignment):
                        assignment = Assignment(assignment, coarse.node_type)
                except Exception as exc:
                    rows.append({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "failed", "reason": str(exc)})
                    continue
                for task in tasks:
                    try:
                        if task == "lowpass_signal_reconstruction":
                            row = evaluate_lowpass_signal_reconstruction(original, coarse, assignment, seed=int(seed), num_signals=8)
                        elif task == "feature_free_label_propagation":
                            row = evaluate_feature_free_label_propagation(original, coarse, assignment, seed=int(seed))
                        else:
                            raise ValueError(f"unsupported structure-critical task: {task}")
                        row.update({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "available", **{k: v for k, v in source.items() if isinstance(v, (str, int, float, bool))}})
                    except Exception as exc:
                        row = {"dataset": dataset, "seed": int(seed), "method": method, "task": task, "run_status": "failed", "reason": str(exc)}
                    rows.append(row)
                    write_csv(output / "structure_task_runs.csv", rows)
    write_csv(output / "structure_task_runs.csv", rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best", "AH-UGC-style-tuned", "GraphZoom-style", "ConvMatch-style", "random"])
    parser.add_argument("--tasks", nargs="+", default=["lowpass_signal_reconstruction", "feature_free_label_propagation"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-runs", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517/runs"))
    parser.add_argument("--guard-runs", type=Path, default=Path("outputs/exp_next10_hgb_guard_ablation_actual_20260517/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next13_structure_critical_tasks(datasets=args.datasets, seeds=args.seeds, methods=args.methods, tasks=args.tasks, graph_root=args.graph_root, resource_runs=args.resource_runs, guard_runs=args.guard_runs, output=args.output)
    summarize_next13_structure_critical_tasks(input=args.output, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
