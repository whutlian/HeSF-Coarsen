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
from experiments.scripts.run_next12_metapath_retention import _coarse_for_method
from experiments.scripts.summarize_next12_structure_sensitive_stress import summarize_next12_structure_sensitive_stress
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.task_gnn import evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph
from hesf_coarsen.io.schema import HeteroGraph


TASK_MAP = {
    "projected_macro_f1": "projected_original_macro_f1",
    "refined@0": "refined_original_macro_f1@0",
    "refined@1": "refined_original_macro_f1@1",
    "refined@3": "refined_original_macro_f1@3",
    "refined@5": "refined_original_macro_f1@5",
    "best_macro_f1": "best_refined_macro_f1",
    "AUC": "refine_auc_macro_f1",
}


def transform_graph_features(graph: HeteroGraph, *, mode: str, value: float, seed: int) -> HeteroGraph:
    if graph.features is None:
        return graph
    rng = np.random.default_rng(int(seed))
    features: dict[int, np.ndarray] = {}
    for type_id, feature in graph.features.items():
        arr = feature.astype(np.float32, copy=True)
        if mode == "feature_mask":
            keep = rng.random(arr.shape) >= float(value)
            arr *= keep.astype(np.float32)
        elif mode == "feature_noise":
            arr += rng.normal(scale=float(value), size=arr.shape).astype(np.float32)
        elif mode == "structure_only":
            arr[:] = 0.0
        else:
            raise ValueError(f"unsupported stress feature mode: {mode}")
        features[int(type_id)] = arr
    return HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations=graph.relations,
        relation_specs=graph.relation_specs,
        features=features,
        labels=graph.labels,
        partitions=graph.partitions,
    )


def _ahugc_tuned(dataset: str, seed: int, graph: HeteroGraph, sweep_summary: Path):
    config = {"hash_bits": 12, "bucket_topk": 8, "assignment_source": "feature_plus_sketch"}
    best = read_csv(sweep_summary / "ahugc_style_best_config_by_dataset.csv")
    for row in best:
        if str(row.get("dataset", "")) == dataset:
            config = {
                "hash_bits": int(float(row.get("hash_bits", config["hash_bits"]))),
                "bucket_topk": int(float(row.get("bucket_topk", config["bucket_topk"]))),
                "assignment_source": row.get("assignment_source", config["assignment_source"]),
            }
            break
    coarse, assignment, diag = coarsen_type_isolated_lsh(graph, target_ratio=0.5, seed=int(seed), **config)
    return coarse, assignment.assignment, {"coarse_source": "ahugc_style_tuned_in_process", **diag}


def _task_row(base: dict[str, Any], metrics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    row = {
        **base,
        "run_status": "unsupported" if metrics.get("skipped") else "available",
        "reason": metrics.get("skip_reason", ""),
        "elapsed_sec": float(elapsed),
    }
    for out_key, metric_key in TASK_MAP.items():
        row[out_key] = metrics.get(metric_key, "")
    return row


def run_next12_structure_sensitive_stress(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    feature_mask_rate: Sequence[float],
    feature_noise_std: Sequence[float],
    structure_only: bool,
    graph_root: Path,
    resource_runs: Path,
    guard_runs: Path,
    ahugc_sweep_summary: Path,
    metapath_summary: Path,
    output: Path,
    device: str,
    epochs: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    meta_index = {
        row.get("method", ""): row
        for row in read_csv(metapath_summary / "metapath_retention_by_method.csv")
    }
    for dataset in datasets:
        original = load_graph(graph_root / f"{dataset.lower()}_hesf")
        for seed in seeds:
            coarse_cache: dict[str, tuple[HeteroGraph, np.ndarray, dict[str, Any]]] = {}
            for method in methods:
                method_key = "AH-UGC-style" if method == "AH-UGC-style-tuned" else method
                try:
                    if method == "AH-UGC-style-tuned":
                        coarse_cache[method] = _ahugc_tuned(dataset, int(seed), original, ahugc_sweep_summary)
                    else:
                        coarse_cache[method] = _coarse_for_method(method_key, dataset, int(seed), original, resource_runs, guard_runs)
                except Exception as exc:
                    rows.append({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "failed", "reason": str(exc)})
                    continue
                stress_specs = [(f"feature_mask_{rate:.2f}", "feature_mask", float(rate)) for rate in feature_mask_rate]
                stress_specs.extend((f"feature_noise_{std:.2f}", "feature_noise", float(std)) for std in feature_noise_std)
                if structure_only:
                    stress_specs.append(("structure_only", "structure_only", 1.0))
                for stress_name, mode, value in stress_specs:
                    coarse, assignment, source = coarse_cache[method]
                    stressed_original = transform_graph_features(original, mode=mode, value=value, seed=int(seed))
                    stressed_coarse = transform_graph_features(coarse, mode=mode, value=value, seed=int(seed))
                    start = perf_counter()
                    try:
                        metrics = evaluate_rgcn_task(
                            stressed_original,
                            stressed_coarse,
                            assignment,
                            seed=int(seed),
                            epochs=int(epochs),
                            refine_epochs=5,
                            refine_epochs_list=[0, 1, 3, 5],
                            hidden_dim=32,
                            device=device,
                        ).metrics
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise
                        metrics = {"skipped": True, "skip_reason": str(exc)}
                    meta = meta_index.get(method_key, {})
                    rows.append(
                        _task_row(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "method": method,
                                "stress_name": stress_name,
                                "stress_mode": mode,
                                "stress_value": value,
                                "relation_energy_error": "",
                                "metapath_typed_survival": meta.get("typed_exact_step_survival_rate_mean", ""),
                                **{key: value for key, value in source.items() if isinstance(value, (str, int, float, bool))},
                            },
                            metrics,
                            perf_counter() - start,
                        )
                    )
                    write_csv(output / "structure_sensitive_stress_runs.csv", rows)
    write_csv(output / "structure_sensitive_stress_runs.csv", rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best", "AH-UGC-style-tuned", "GraphZoom-style", "ConvMatch-style"])
    parser.add_argument("--feature-mask-rate", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--feature-noise-std", type=float, nargs="+", default=[0.10, 0.50, 1.00])
    parser.add_argument("--structure-only", action="store_true", default=True)
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-runs", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517/runs"))
    parser.add_argument("--guard-runs", type=Path, default=Path("outputs/exp_next10_hgb_guard_ablation_actual_20260517/runs"))
    parser.add_argument("--ahugc-sweep-summary", type=Path, default=Path("outputs/exp_next12_ahugc_style_sweep_20260517_summary"))
    parser.add_argument("--metapath-summary", type=Path, default=Path("outputs/exp_next12_metapath_retention_20260517_summary"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next12_structure_sensitive_stress(
        datasets=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        feature_mask_rate=args.feature_mask_rate,
        feature_noise_std=args.feature_noise_std,
        structure_only=bool(args.structure_only),
        graph_root=args.graph_root,
        resource_runs=args.resource_runs,
        guard_runs=args.guard_runs,
        ahugc_sweep_summary=args.ahugc_sweep_summary,
        metapath_summary=args.metapath_summary,
        output=args.output,
        device=args.device,
        epochs=int(args.epochs),
    )
    summarize_next12_structure_sensitive_stress(input=args.output, output=Path(str(args.output) + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
