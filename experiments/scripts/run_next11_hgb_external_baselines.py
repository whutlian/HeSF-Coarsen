from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.next11_common import read_csv
from experiments.scripts.summarize_next11_external_baselines import summarize_next11_external_baselines
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.resource_logging import ResourceMonitor, snapshot_resources
from hesf_coarsen.eval.spectral_diagnostics import _target_matched_baseline, compute_spectral_diagnostics
from hesf_coarsen.eval.task_gnn import evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph


TARGET_MATCHED = {"GraphZoom-style": "graphzoom_style", "ConvMatch-style": "convmatch_style", "random": "random"}


def _uniform_relation_weights(graph) -> dict[int, float]:
    rels = sorted(int(relation_id) for relation_id in graph.relations)
    return {relation_id: 1.0 / len(rels) for relation_id in rels} if rels else {}


def _resource_rows(path: Path, methods: set[str], datasets: set[str], seeds: set[int]) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path / "hgb_resource_logged_runs.csv"):
        if row.get("method") in methods and row.get("dataset") in datasets and int(float(row.get("seed", 0))) in seeds:
            rows.append({**row, "evidence_source": "next10_resource_logged", "run_status": "available"})
    return rows


def _task_values(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "projected": metrics.get("projected_original_macro_f1", ""),
        "refined@0": metrics.get("refined_original_macro_f1@0", ""),
        "refined@1": metrics.get("refined_original_macro_f1@1", ""),
        "refined@3": metrics.get("refined_original_macro_f1@3", ""),
        "refined@5": metrics.get("refined_original_macro_f1@5", ""),
        "best": metrics.get("best_refined_macro_f1", ""),
        "AUC": metrics.get("refine_auc_macro_f1", ""),
        "task_train_wall_clock_sec": metrics.get("train_time", ""),
        "refine_wall_clock_sec": metrics.get("refine_time", ""),
    }


def _run_baseline(method: str, dataset: str, seed: int, graph, device: str, epochs: int) -> dict[str, Any]:
    monitor = ResourceMonitor()
    start = perf_counter()
    if method == "AH-UGC-style":
        coarse, assignment, control = coarsen_type_isolated_lsh(graph, target_ratio=0.5, seed=int(seed), max_cluster_size=4)
    else:
        coarse, assignment, control = _target_matched_baseline(
            graph,
            TARGET_MATCHED[method],
            target_ratio=0.5,
            target_tolerance=0.02,
            max_levels=4,
            seed=int(seed),
            relation_weights=_uniform_relation_weights(graph),
            dim=4,
        )
    coarsening_sec = perf_counter() - start
    spectral = compute_spectral_diagnostics(
        graph,
        coarse,
        assignment,
        seed=int(seed),
        num_signals=4,
        smoothing_steps=1,
        relation_weights=_uniform_relation_weights(graph),
        baseline_methods=None,
    )
    task = evaluate_rgcn_task(
        graph,
        coarse,
        assignment.assignment,
        seed=int(seed),
        epochs=int(epochs),
        refine_epochs=5,
        refine_epochs_list=[0, 1, 3, 5],
        hidden_dim=32,
        device=device,
    ).metrics
    resources = monitor.sample()
    cuda = snapshot_resources()
    return {
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "target_hit": control.get("target_hit", ""),
        "final_ratio": coarse.num_nodes / max(graph.num_nodes, 1),
        "cumulative_dee_or_audited_dee": spectral.get("dirichlet_energy_relative_error", ""),
        "FSE": spectral.get("fused_sketch_energy_relative_error", ""),
        "REEmax": spectral.get("relation_energy_relative_error_max", ""),
        "SIPE": spectral.get("chebheat_sketch_inner_product_relative_error", ""),
        **_task_values(task),
        "coarsening_wall_clock_sec": coarsening_sec,
        "peak_rss_gb": resources.get("peak_rss_gb", ""),
        "peak_vram_allocated_gb": cuda.get("peak_vram_allocated_gb", ""),
        "run_status": "available",
        "evidence_source": "next11_actual_baseline",
    }


def run_next11_hgb_external_baselines(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    graph_root: Path,
    resource_logged: Path,
    output: Path,
    device: str,
    epochs: int,
    quick: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds[:2] if quick else seeds)
    rows = _resource_rows(resource_logged, set(methods), set(datasets), {int(seed) for seed in seeds})
    existing = {(row["method"], row["dataset"], int(float(row["seed"]))) for row in rows}
    for dataset in datasets:
        graph = load_graph(graph_root / f"{dataset.lower()}_hesf")
        for seed in seeds:
            for method in methods:
                if (method, dataset, int(seed)) in existing:
                    continue
                if method not in {"AH-UGC-style", *TARGET_MATCHED}:
                    rows.append({"method": method, "dataset": dataset, "seed": int(seed), "run_status": "not_run", "reason": "no compatible existing run or baseline implementation"})
                    continue
                rows.append(_run_baseline(method, dataset, int(seed), graph, device, epochs))
                write_csv(output / "external_baseline_runs.csv", rows)
    write_csv(output / "external_baseline_runs.csv", rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S", "AH-UGC-style", "flatten-sum", "H0-mutual-best", "GraphZoom-style", "ConvMatch-style", "random"])
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-logged", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next11_hgb_external_baselines(datasets=args.datasets, seeds=args.seeds, methods=args.methods, graph_root=args.graph_root, resource_logged=args.resource_logged, output=args.output, device=args.device, epochs=int(args.epochs), quick=bool(args.quick))
    summarize_next11_external_baselines(input=args.output, output=Path(str(args.output) + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
