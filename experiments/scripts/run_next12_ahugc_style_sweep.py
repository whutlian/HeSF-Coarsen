from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.run_next11_hgb_external_baselines import _task_values, _uniform_relation_weights
from experiments.scripts.summarize_next12_ahugc_style_sweep import summarize_next12_ahugc_style_sweep
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.resource_logging import ResourceMonitor, snapshot_resources
from hesf_coarsen.eval.spectral_diagnostics import compute_spectral_diagnostics
from hesf_coarsen.eval.task_gnn import evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph


def _configs(path: Path) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for hash_bits, bucket_topk, assignment_source in product(cfg["hash_bits"], cfg["bucket_topk"], cfg["assignment_source"]):
        out.append(
            {
                "hash_bits": int(hash_bits),
                "bucket_topk": int(bucket_topk),
                "assignment_source": str(assignment_source),
                "same_partition_only": bool(cfg.get("same_partition_only", True)),
                "target_ratio": float(cfg.get("target_ratio", 0.5)),
            }
        )
    return out


def run_next12_ahugc_style_sweep(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    config: Path,
    graph_root: Path,
    output: Path,
    device: str,
    epochs: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        graph = load_graph(graph_root / f"{dataset.lower()}_hesf")
        for seed in seeds:
            for spec in _configs(config):
                monitor = ResourceMonitor()
                start = perf_counter()
                try:
                    coarse, assignment, control = coarsen_type_isolated_lsh(graph, seed=int(seed), **spec)
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
                    rows.append(
                        {
                            "method": "AH-UGC-style",
                            "dataset": dataset,
                            "seed": int(seed),
                            **spec,
                            "target_hit": control.get("target_hit", ""),
                            "final_ratio": control.get("final_ratio", ""),
                            "projected_macro_f1": task.get("projected_original_macro_f1", ""),
                            "refined@0": task.get("refined_original_macro_f1@0", ""),
                            "refined@1": task.get("refined_original_macro_f1@1", ""),
                            "refined@3": task.get("refined_original_macro_f1@3", ""),
                            "refined@5": task.get("refined_original_macro_f1@5", ""),
                            "best_macro_f1": task.get("best_refined_macro_f1", ""),
                            "AUC": task.get("refine_auc_macro_f1", ""),
                            "resource_logged_cumulative_dee": spectral.get("dirichlet_energy_relative_error", ""),
                            "relation_energy_error": spectral.get("relation_energy_relative_error_max", ""),
                            "coarsening_wall_clock_sec": coarsening_sec,
                            "peak_rss_gb": resources.get("peak_rss_gb", ""),
                            "peak_vram_allocated_gb": cuda.get("peak_vram_allocated_gb", ""),
                            "coarse_nodes": coarse.num_nodes,
                            "coarse_edges": sum(rel.num_edges for rel in coarse.relations.values()),
                            "run_status": "available" if not task.get("skipped") else "unsupported",
                            "reason": task.get("skip_reason", ""),
                        }
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    rows.append({"method": "AH-UGC-style", "dataset": dataset, "seed": int(seed), **spec, "run_status": "failed", "reason": str(exc)})
                write_csv(output / "ahugc_style_sweep_runs.csv", rows)
    write_csv(output / "ahugc_style_sweep_runs.csv", rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--config", type=Path, default=Path("configs/paper/hgb_ahugc_style_sweep.yaml"))
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next12_ahugc_style_sweep(datasets=args.datasets, seeds=args.seeds, config=args.config, graph_root=args.graph_root, output=args.output, device=args.device, epochs=int(args.epochs))
    summarize_next12_ahugc_style_sweep(input=args.output, output=Path(str(args.output) + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
