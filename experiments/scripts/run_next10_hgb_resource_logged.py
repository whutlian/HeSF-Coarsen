from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_command_metadata, write_config_snapshot, write_csv
from experiments.scripts.summarize_next9_hgb_paper_final import _plot_scatter
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.eval.resource_logging import ResourceMonitor, snapshot_resources
from hesf_coarsen.eval.spectral_diagnostics import _target_matched_baseline, compute_spectral_diagnostics
from hesf_coarsen.eval.task_gnn import evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph
from hesf_coarsen.io.schema import HeteroGraph


COARSEN_CONFIGS: Mapping[str, str] = {
    "HeSF-LVC-P": "configs/paper/hgb_hesf_lvc_p.yaml",
    "HeSF-LVC-S": "configs/paper/hgb_hesf_lvc_s.yaml",
    "flatten-sum": "configs/paper/hgb_flatten_sum.yaml",
    "H0-mutual-best": "configs/paper/hgb_h0_mutual_best.yaml",
}
TARGET_MATCHED = {
    "GraphZoom-style": "graphzoom_style",
    "ConvMatch-style": "convmatch_style",
}
FULL_GRAPH_METHODS = {"full RGCN default", "full RGCN tuned"}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _graph_dir(graph_root: Path, dataset: str) -> Path:
    return graph_root / f"{dataset.lower()}_hesf"


def _edge_count(graph: HeteroGraph) -> int:
    return int(sum(rel.num_edges for rel in graph.relations.values()))


def _uniform_relation_weights(graph: HeteroGraph) -> dict[int, float]:
    relation_ids = sorted(int(relation_id) for relation_id in graph.relations)
    if not relation_ids:
        return {}
    weight = 1.0 / len(relation_ids)
    return {relation_id: weight for relation_id in relation_ids}


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _resources(monitor: ResourceMonitor) -> dict[str, Any]:
    snapshot = monitor.sample()
    cuda = snapshot_resources()
    return {
        "peak_rss_gb": snapshot.get("peak_rss_gb", ""),
        "peak_vram_allocated_gb": cuda.get("peak_vram_allocated_gb", ""),
        "peak_vram_reserved_gb": cuda.get("peak_vram_reserved_gb", ""),
    }


def _with_resource_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["coarsening_wall_clock_sec"] = out.get("coarsen_sec", "")
    out["task_train_wall_clock_sec"] = out.get("coarse_train_sec", "")
    out["refine_wall_clock_sec"] = out.get("refine_sec", "")
    return out


def _task_values(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "projected_macro_f1": metrics.get("projected_original_macro_f1", ""),
        "refined_macro_f1@0": metrics.get("refined_original_macro_f1@0", ""),
        "refined_macro_f1@1": metrics.get("refined_original_macro_f1@1", ""),
        "refined_macro_f1@3": metrics.get("refined_original_macro_f1@3", ""),
        "refined_macro_f1@5": metrics.get("refined_original_macro_f1@5", metrics.get("refined_original_macro_f1", "")),
        "best_macro_f1": metrics.get("best_refined_macro_f1", metrics.get("refined_original_macro_f1", "")),
        "refine_auc_macro_f1": metrics.get("refine_auc_macro_f1", ""),
        "coarse_train_sec": metrics.get("train_time", ""),
        "task_train_wall_clock_sec": metrics.get("train_time", ""),
        "refine_sec": metrics.get("refine_time", ""),
        "refine_wall_clock_sec": metrics.get("refine_time", ""),
    }


def _spectral_values(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "DEE": metrics.get("dirichlet_energy_relative_error", ""),
        "REEmax": metrics.get("relation_energy_relative_error_max", ""),
        "SIPE": metrics.get("chebheat_sketch_inner_product_relative_error", ""),
        "FSE": metrics.get("fused_sketch_energy_relative_error", ""),
    }


def _evaluate_task(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    *,
    seed: int,
    device: str,
    epochs: int,
    refine_epochs: Sequence[int],
) -> dict[str, Any]:
    result = evaluate_rgcn_task(
        original,
        coarse,
        assignment,
        seed=int(seed),
        epochs=int(epochs),
        refine_epochs=max(int(value) for value in refine_epochs),
        refine_epochs_list=[int(value) for value in refine_epochs],
        hidden_dim=32,
        device=device,
    ).metrics
    return dict(result)


def _coarsen_config_method(
    method: str,
    dataset: str,
    seed: int,
    graph: HeteroGraph,
    graph_root: Path,
    output: Path,
    *,
    device: str,
    epochs: int,
    refine_epochs: Sequence[int],
) -> dict[str, Any]:
    config = deepcopy(_load_yaml(Path(COARSEN_CONFIGS[method])))
    run_name = f"next10_resource_{dataset}_{method.replace(' ', '_').replace('-', '_')}_seed{seed}"
    run_dir = output / "runs" / run_name
    config["seed"] = int(seed)
    config.setdefault("output", {})["dir"] = str(run_dir)
    config.setdefault("diagnostics", {})["enable_large_graph_envelope"] = True
    config.setdefault("diagnostics", {})["spectral_relation_detail"] = True
    config.setdefault("paper", {})["method"] = method
    config.setdefault("paper", {})["variant"] = method
    write_config_snapshot(run_dir / "config.yaml", config)
    write_command_metadata(
        run_dir,
        run_name=run_name,
        status="running",
        dataset=dataset,
        seed=int(seed),
        method=method,
        variant=method,
        experiment_block="next10_hgb_resource_logged",
    )
    _reset_cuda_peak()
    monitor = ResourceMonitor()
    start = perf_counter()
    results = run_multilevel_coarsening(graph, config)
    coarsen_sec = float(perf_counter() - start)
    if not results:
        raise RuntimeError(f"no coarsening result for {run_name}")
    final = results[-1]
    level_dir = run_dir / f"level_{final.level}"
    cumulative_path = level_dir / "cumulative_assignment.npz"
    if cumulative_path.exists():
        assignment = np.load(cumulative_path)["assignment"].astype(np.int64, copy=False)
    else:
        assignment = final.assignment.assignment.astype(np.int64, copy=False)
    task = _evaluate_task(
        graph,
        final.graph,
        assignment,
        seed=seed,
        device=device,
        epochs=epochs,
        refine_epochs=refine_epochs,
    )
    resources = _resources(monitor)
    diagnostics = final.diagnostics
    spectral = diagnostics.get("cumulative_spectral", diagnostics.get("spectral", {}))
    row = {
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "target_ratio": config.get("coarsening", {}).get("target_ratio", 0.5),
        "target_hit": str(final.graph.num_nodes <= int(round(graph.num_nodes * float(config.get("coarsening", {}).get("target_ratio", 0.5))))).lower(),
        "original_nodes": graph.num_nodes,
        "coarse_nodes": final.graph.num_nodes,
        "original_edges": _edge_count(graph),
        "coarse_edges": _edge_count(final.graph),
        "edge_compression_ratio": _edge_count(final.graph) / max(_edge_count(graph), 1),
        "coarsen_sec": coarsen_sec,
        "total_wall_clock_sec": coarsen_sec + float(task.get("total_time", 0.0) or 0.0),
        **_spectral_values(spectral),
        **_task_values(task),
        **resources,
    }
    write_command_metadata(
        run_dir,
        run_name=run_name,
        status="success",
        dataset=dataset,
        seed=int(seed),
        method=method,
        variant=method,
        experiment_block="next10_hgb_resource_logged",
        peak_rss_gb=row.get("peak_rss_gb"),
        peak_vram_allocated_gb=row.get("peak_vram_allocated_gb"),
        peak_vram_reserved_gb=row.get("peak_vram_reserved_gb"),
    )
    return _with_resource_aliases(row)


def _target_matched_method(
    method: str,
    dataset: str,
    seed: int,
    graph: HeteroGraph,
    *,
    device: str,
    epochs: int,
    refine_epochs: Sequence[int],
) -> dict[str, Any]:
    baseline = TARGET_MATCHED[method]
    relation_weights = _uniform_relation_weights(graph)
    _reset_cuda_peak()
    monitor = ResourceMonitor()
    start = perf_counter()
    coarse, assignment, control = _target_matched_baseline(
        graph,
        baseline,
        target_ratio=0.5,
        target_tolerance=0.02,
        max_levels=4,
        seed=int(seed),
        relation_weights=relation_weights,
        dim=4,
    )
    coarsen_sec = float(perf_counter() - start)
    spectral = compute_spectral_diagnostics(
        graph,
        coarse,
        assignment,
        seed=int(seed),
        num_signals=4,
        smoothing_steps=1,
        relation_weights=relation_weights,
        baseline_methods=None,
    )
    task = _evaluate_task(
        graph,
        coarse,
        assignment.assignment,
        seed=seed,
        device=device,
        epochs=epochs,
        refine_epochs=refine_epochs,
    )
    return _with_resource_aliases({
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "target_ratio": 0.5,
        "target_hit": str(bool(control.get("target_hit", False))).lower(),
        "original_nodes": graph.num_nodes,
        "coarse_nodes": coarse.num_nodes,
        "original_edges": _edge_count(graph),
        "coarse_edges": _edge_count(coarse),
        "edge_compression_ratio": _edge_count(coarse) / max(_edge_count(graph), 1),
        "coarsen_sec": coarsen_sec,
        "total_wall_clock_sec": coarsen_sec + float(task.get("total_time", 0.0) or 0.0),
        **_spectral_values(spectral),
        **_task_values(task),
        **_resources(monitor),
    })


def _full_graph_method(
    method: str,
    dataset: str,
    seed: int,
    graph: HeteroGraph,
    *,
    device: str,
    default_epochs: int,
    tuned_epochs: int,
) -> dict[str, Any]:
    epochs = tuned_epochs if method == "full RGCN tuned" else default_epochs
    _reset_cuda_peak()
    monitor = ResourceMonitor()
    assignment = np.arange(graph.num_nodes, dtype=np.int64)
    task = _evaluate_task(
        graph,
        graph,
        assignment,
        seed=seed,
        device=device,
        epochs=epochs,
        refine_epochs=[0],
    )
    return _with_resource_aliases({
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "target_ratio": 1.0,
        "target_hit": "true",
        "original_nodes": graph.num_nodes,
        "coarse_nodes": graph.num_nodes,
        "original_edges": _edge_count(graph),
        "coarse_edges": _edge_count(graph),
        "edge_compression_ratio": 1.0,
        "coarsen_sec": 0.0,
        "total_wall_clock_sec": float(task.get("total_time", 0.0) or 0.0),
        "DEE": "",
        "REEmax": "",
        "SIPE": "",
        "FSE": "",
        **_task_values(task),
        **_resources(monitor),
    })


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("method", "")), []).append(row)
    out = []
    for method, group in sorted(groups.items()):
        item: dict[str, Any] = {"method": method, "run_count": len(group)}
        for metric in (
            "best_macro_f1",
            "refined_macro_f1@5",
            "DEE",
            "REEmax",
            "SIPE",
            "total_wall_clock_sec",
            "coarsen_sec",
            "coarsening_wall_clock_sec",
            "coarse_train_sec",
            "task_train_wall_clock_sec",
            "refine_sec",
            "refine_wall_clock_sec",
            "peak_rss_gb",
            "peak_vram_allocated_gb",
            "edge_compression_ratio",
        ):
            values = []
            for row in group:
                try:
                    if row.get(metric) not in {None, ""}:
                        values.append(float(row[metric]))
                except (TypeError, ValueError):
                    pass
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else (0.0 if values else "")
        out.append(item)
    return out


def run_resource_logged(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    graph_root: Path,
    output: Path,
    device: str,
    epochs: int,
    tuned_epochs: int,
    refine_epochs: Sequence[int],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        graph = load_graph(_graph_dir(graph_root, dataset))
        for seed in seeds:
            for method in methods:
                if method in COARSEN_CONFIGS:
                    row = _coarsen_config_method(
                        method,
                        dataset,
                        int(seed),
                        graph,
                        graph_root,
                        output,
                        device=device,
                        epochs=epochs,
                        refine_epochs=refine_epochs,
                    )
                elif method in TARGET_MATCHED:
                    row = _target_matched_method(
                        method,
                        dataset,
                        int(seed),
                        graph,
                        device=device,
                        epochs=epochs,
                        refine_epochs=refine_epochs,
                    )
                elif method in FULL_GRAPH_METHODS:
                    row = _full_graph_method(
                        method,
                        dataset,
                        int(seed),
                        graph,
                        device=device,
                        default_epochs=epochs,
                        tuned_epochs=tuned_epochs,
                    )
                else:
                    raise ValueError(f"unsupported resource method: {method}")
                rows.append(row)
                write_csv(output / "hgb_resource_logged_runs.csv", rows)
    aggregate = _aggregate(rows)
    write_csv(output / "hgb_resource_logged_runs.csv", rows)
    write_csv(output / "hgb_resource_logged_by_method.csv", aggregate)
    figure_dir = output / "figures"
    _plot_scatter(aggregate, "total_wall_clock_sec_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_total_wall_clock.png")
    _plot_scatter(aggregate, "peak_rss_gb_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_peak_rss.png")
    _plot_scatter(aggregate, "coarsen_sec_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_coarsen_time.png")
    _plot_scatter(aggregate, "edge_compression_ratio_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_edge_compression_ratio.png")
    report = [
        "# Next10 resource-logged HGB quality-cost rerun",
        "",
        markdown_table(
            aggregate,
            [
                "method",
                "run_count",
                "best_macro_f1_mean",
                "total_wall_clock_sec_mean",
                "peak_rss_gb_mean",
                "peak_vram_allocated_gb_mean",
                "edge_compression_ratio_mean",
                "DEE_mean",
            ],
        ),
        "",
        "Claim wording: slightly lower F1, much lower graph/operator cost, much stronger preservation.",
    ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "HeSF-LVC-P",
            "HeSF-LVC-S",
            "flatten-sum",
            "H0-mutual-best",
            "GraphZoom-style",
            "ConvMatch-style",
            "full RGCN default",
            "full RGCN tuned",
        ],
    )
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tuned-epochs", type=int, default=40)
    parser.add_argument("--refine-epochs", type=int, nargs="+", default=[0, 1, 3, 5])
    args = parser.parse_args(argv)
    run_resource_logged(
        datasets=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        graph_root=args.graph_root,
        output=args.output,
        device=args.device,
        epochs=int(args.epochs),
        tuned_epochs=int(args.tuned_epochs),
        refine_epochs=args.refine_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
