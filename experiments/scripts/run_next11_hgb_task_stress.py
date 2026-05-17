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
from experiments.scripts.run_hgb_task_eval import _cumulative_assignment, _final_level_dir
from experiments.scripts.summarize_next11_hgb_task_stress import summarize_next11_hgb_task_stress
from hesf_coarsen.eval.spectral_diagnostics import _target_matched_baseline
from hesf_coarsen.eval.task_gnn import evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj


RESOURCE_RUN_METHODS = {"HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H0-mutual-best"}
TARGET_MATCHED_METHODS = {"GraphZoom-style": "graphzoom_style", "ConvMatch-style": "convmatch_style", "random": "random"}
DEFAULT_METHODS = ["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best", "GraphZoom-style", "ConvMatch-style"]
TASK_METRIC_MAP = {
    "projected_macro_f1": "projected_original_macro_f1",
    "refined_macro_f1@0": "refined_original_macro_f1@0",
    "refined_macro_f1@1": "refined_original_macro_f1@1",
    "refined_macro_f1@3": "refined_original_macro_f1@3",
    "refined_macro_f1@5": "refined_original_macro_f1@5",
    "best_macro_f1": "best_refined_macro_f1",
    "refine_auc_macro_f1": "refine_auc_macro_f1",
}


def _method_token(method: str) -> str:
    return method.replace(" ", "_").replace("-", "_")


def _uniform_relation_weights(graph: HeteroGraph) -> dict[int, float]:
    rels = sorted(int(relation_id) for relation_id in graph.relations)
    if not rels:
        return {}
    return {relation_id: 1.0 / len(rels) for relation_id in rels}


def _existing_run_dir(method: str, dataset: str, seed: int, resource_runs: Path, guard_runs: Path) -> Path | None:
    if method in RESOURCE_RUN_METHODS:
        candidate = resource_runs / f"next10_resource_{dataset}_{_method_token(method)}_seed{seed}"
        if candidate.exists():
            return candidate
    if method == "H6-no-spec":
        candidate = guard_runs / f"next10_guard_{dataset}_H6_no_spec_seed{seed}"
        if candidate.exists():
            return candidate
    return None


def _coarse_from_existing(run_dir: Path, original: HeteroGraph) -> tuple[HeteroGraph, np.ndarray, str]:
    final = _final_level_dir(run_dir)
    if final is None:
        raise ValueError(f"missing final level in {run_dir}")
    coarse = load_graph(final)
    assignment = _cumulative_assignment(run_dir, original.num_nodes, final)
    return coarse, assignment, str(final)


def _coarse_for_method(
    method: str,
    dataset: str,
    seed: int,
    original: HeteroGraph,
    resource_runs: Path,
    guard_runs: Path,
) -> tuple[HeteroGraph, np.ndarray, dict[str, Any]]:
    run_dir = _existing_run_dir(method, dataset, seed, resource_runs, guard_runs)
    if run_dir is not None:
        coarse, assignment, final_dir = _coarse_from_existing(run_dir, original)
        return coarse, assignment, {"coarse_source": "existing_run", "run_dir": str(run_dir), "final_level_dir": final_dir}
    baseline = TARGET_MATCHED_METHODS.get(method)
    if baseline is None:
        raise ValueError(f"unsupported stress method: {method}")
    coarse, assignment_obj, control = _target_matched_baseline(
        original,
        baseline,
        target_ratio=0.5,
        target_tolerance=0.02,
        max_levels=4,
        seed=int(seed),
        relation_weights=_uniform_relation_weights(original),
        dim=4,
    )
    return coarse, assignment_obj.assignment, {"coarse_source": "target_matched_in_process", **control}


def _mask_relation(graph: HeteroGraph, relation_id: int, fraction: float, seed: int) -> HeteroGraph:
    relations = {}
    rng = np.random.default_rng(int(seed) + int(relation_id) * 104729)
    for rid, rel in graph.relations.items():
        if int(rid) != int(relation_id):
            relations[int(rid)] = rel
            continue
        keep_prob = max(0.0, min(1.0, 1.0 - float(fraction)))
        keep = rng.random(rel.num_edges) < keep_prob
        if rel.num_edges and not np.any(keep):
            keep[rng.integers(0, rel.num_edges)] = True
        relations[int(rid)] = RelationAdj(
            src=rel.src[keep],
            dst=rel.dst[keep],
            weight=rel.weight[keep],
            src_type=rel.src_type,
            dst_type=rel.dst_type,
            relation_id=rel.relation_id,
        )
    return HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type,
        relations=relations,
        relation_specs=graph.relation_specs,
        features=graph.features,
        labels=graph.labels,
        partitions=graph.partitions,
    )


def _row_from_metrics(base: dict[str, Any], metrics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    skipped = bool(metrics.get("skipped", False))
    row = {
        **base,
        "run_status": "unsupported" if skipped else "available",
        "reason": metrics.get("skip_reason", ""),
        "elapsed_sec": float(elapsed),
        "label_coverage_train": metrics.get("label_coverage_train", ""),
        "num_classes_present_train": metrics.get("num_classes_present_train", ""),
        "train_labeled_nodes": metrics.get("train_labeled_nodes", ""),
        "test_labeled_nodes": metrics.get("test_labeled_nodes", ""),
        "coarse_model": metrics.get("coarse_model", ""),
    }
    for out_key, metric_key in TASK_METRIC_MAP.items():
        row[out_key] = metrics.get(metric_key, "")
    return row


def _evaluate(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    *,
    base: dict[str, Any],
    seed: int,
    device: str,
    epochs: int,
    train_fraction: float,
    coarse_model: str = "rgcn_lite",
) -> dict[str, Any]:
    start = perf_counter()
    try:
        metrics = evaluate_rgcn_task(
            original,
            coarse,
            assignment,
            seed=int(seed),
            epochs=int(epochs),
            refine_epochs=5,
            refine_epochs_list=[0, 1, 3, 5],
            hidden_dim=32,
            device=device,
            coarse_model=coarse_model,
            train_fraction=float(train_fraction),
        ).metrics
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise
        metrics = {"skipped": True, "skip_reason": str(exc), "coarse_model": coarse_model}
    return _row_from_metrics(base, metrics, perf_counter() - start)


def run_next11_hgb_task_stress(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    stress: Sequence[str],
    graph_root: Path,
    resource_runs: Path,
    guard_runs: Path,
    output: Path,
    device: str,
    epochs: int,
    quick: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds[:2] if quick else seeds)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        original = load_graph(graph_root / f"{dataset.lower()}_hesf")
        relation_ids = sorted(original.relations)
        for seed in seeds:
            cache: dict[str, tuple[HeteroGraph, np.ndarray, dict[str, Any]]] = {}
            for method in methods:
                try:
                    cache[method] = _coarse_for_method(method, dataset, int(seed), original, resource_runs, guard_runs)
                except Exception as exc:
                    cache[method] = (original, np.arange(original.num_nodes, dtype=np.int64), {"coarse_source": "unavailable", "reason": str(exc)})
                coarse, assignment, source = cache[method]
                if "early_refine" in stress:
                    rows.append(_evaluate(original, coarse, assignment, base={"stress_block": "early_refine", "stress_name": "standard", "method": method, "dataset": dataset, "seed": int(seed), **source}, seed=int(seed), device=device, epochs=epochs, train_fraction=0.6))
                if "low_label" in stress:
                    for fraction in (0.05, 0.10, 0.20, 0.60):
                        rows.append(_evaluate(original, coarse, assignment, base={"stress_block": "low_label", "stress_name": f"label_{fraction:.2f}", "train_label_fraction": fraction, "method": method, "dataset": dataset, "seed": int(seed), **source}, seed=int(seed), device=device, epochs=epochs, train_fraction=fraction))
                if "cross_model" in stress:
                    for model in ("rgcn_lite", "han_small", "hgt_lite"):
                        rows.append(_evaluate(original, coarse, assignment, base={"stress_block": "cross_model", "stress_name": model, "method": method, "dataset": dataset, "seed": int(seed), **source}, seed=int(seed), device=device, epochs=epochs, train_fraction=0.6, coarse_model=model))
                if "relation_mask" in stress:
                    for relation_id in relation_ids:
                        for fraction in (0.25, 0.50):
                            rows.append({"stress_block": "relation_mask", "stress_name": f"train_only_rel{relation_id}_{fraction:.2f}", "mask_mode": "train_only", "masked_relation_id": int(relation_id), "mask_fraction": fraction, "method": method, "dataset": dataset, "seed": int(seed), "run_status": "unsupported", "reason": "train_only relation masking requires separate train/eval graph views"})
                            masked_original = _mask_relation(original, int(relation_id), fraction, int(seed))
                            masked_coarse = _mask_relation(coarse, int(relation_id), fraction, int(seed)) if int(relation_id) in coarse.relations else coarse
                            rows.append(_evaluate(masked_original, masked_coarse, assignment, base={"stress_block": "relation_mask", "stress_name": f"eval_stress_rel{relation_id}_{fraction:.2f}", "mask_mode": "eval_stress", "masked_relation_id": int(relation_id), "mask_fraction": fraction, "method": method, "dataset": dataset, "seed": int(seed), **source}, seed=int(seed), device=device, epochs=epochs, train_fraction=0.6))
                write_csv(output / "task_stress_runs.csv", rows)
    write_csv(output / "task_stress_runs.csv", rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--stress", nargs="+", default=["low_label", "early_refine", "cross_model", "relation_mask"])
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-runs", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517/runs"))
    parser.add_argument("--guard-runs", type=Path, default=Path("outputs/exp_next10_hgb_guard_ablation_actual_20260517/runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next11_hgb_task_stress(
        datasets=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        stress=args.stress,
        graph_root=args.graph_root,
        resource_runs=args.resource_runs,
        guard_runs=args.guard_runs,
        output=args.output,
        device=args.device,
        epochs=int(args.epochs),
        quick=bool(args.quick),
    )
    summarize_next11_hgb_task_stress(input=args.output, output=Path(str(args.output) + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

