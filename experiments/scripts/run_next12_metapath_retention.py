from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.run_hgb_task_eval import _cumulative_assignment, _final_level_dir
from experiments.scripts.run_next11_hgb_task_stress import _method_token, _target_matched_baseline, _uniform_relation_weights
from experiments.scripts.summarize_next12_metapath_retention import summarize_next12_metapath_retention
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.metapath_retention import (
    evaluate_path_retention,
    infer_schema_paths,
    sample_typed_paths,
    summarize_metapath_retention,
)
from hesf_coarsen.io.edge_list import load_graph


RESOURCE_RUN_METHODS = {"HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H0-mutual-best"}
TARGET_MATCHED_METHODS = {"GraphZoom-style": "graphzoom_style", "ConvMatch-style": "convmatch_style", "random": "random"}


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


def _coarse_for_method(method: str, dataset: str, seed: int, original, resource_runs: Path, guard_runs: Path):
    run_dir = _existing_run_dir(method, dataset, seed, resource_runs, guard_runs)
    if run_dir is not None:
        final = _final_level_dir(run_dir)
        if final is None:
            raise ValueError(f"missing final level for {run_dir}")
        return load_graph(final), _cumulative_assignment(run_dir, original.num_nodes, final), {"coarse_source": "existing_run", "run_dir": str(run_dir)}
    if method == "AH-UGC-style":
        coarse, assignment, diag = coarsen_type_isolated_lsh(
            original,
            target_ratio=0.5,
            seed=int(seed),
            hash_bits=12,
            bucket_topk=8,
            assignment_source="feature_plus_sketch",
        )
        return coarse, assignment.assignment, {"coarse_source": "ahugc_style_in_process", **diag}
    baseline = TARGET_MATCHED_METHODS.get(method)
    if baseline is None:
        raise ValueError(f"unsupported method: {method}")
    coarse, assignment, control = _target_matched_baseline(
        original,
        baseline,
        target_ratio=0.5,
        target_tolerance=0.02,
        max_levels=4,
        seed=int(seed),
        relation_weights=_uniform_relation_weights(original),
        dim=4,
    )
    return coarse, assignment.assignment, {"coarse_source": "target_matched_in_process", **control}


def run_next12_metapath_retention(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    schema_path_lengths: Sequence[int],
    max_schema_paths: int,
    max_samples_per_schema: int,
    max_frontier_per_step: int,
    max_count_frontier_per_step: int,
    max_count_per_endpoint_schema: int,
    graph_root: Path,
    resource_runs: Path,
    guard_runs: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path_samples: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    run_status: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "same_sample_controls": "dataset_seed_schema_path_sample_seed shared across methods",
        "max_schema_paths": int(max_schema_paths),
        "max_samples_per_schema": int(max_samples_per_schema),
        "max_frontier_per_step": int(max_frontier_per_step),
        "max_count_frontier_per_step": int(max_count_frontier_per_step),
        "max_count_per_endpoint_schema": int(max_count_per_endpoint_schema),
    }
    for dataset in datasets:
        original = load_graph(graph_root / f"{dataset.lower()}_hesf")
        schema_paths = infer_schema_paths(
            original,
            target_node_type=None,
            lengths=[int(value) for value in schema_path_lengths],
            max_paths=int(max_schema_paths),
        )
        for seed in seeds:
            samples = sample_typed_paths(
                original,
                schema_paths,
                sample_seed=int(seed),
                max_samples_per_schema=int(max_samples_per_schema),
                max_trials_per_schema=max(int(max_samples_per_schema) * 20, 1000),
                max_frontier_per_step=int(max_frontier_per_step),
            )
            for sample in samples:
                sample["dataset"] = dataset
                sample["seed"] = int(seed)
            path_samples.extend(samples)
            for method in methods:
                try:
                    coarse, assignment, source = _coarse_for_method(method, dataset, int(seed), original, resource_runs, guard_runs)
                    method_samples = [{**sample, "method": method} for sample in samples]
                    rows = evaluate_path_retention(
                        method_samples,
                        assignment,
                        coarse,
                        original_graph=original,
                        max_count_frontier_per_step=int(max_count_frontier_per_step),
                        max_count_per_endpoint_schema=int(max_count_per_endpoint_schema),
                    )
                    for row in rows:
                        row.update({key: value for key, value in source.items() if isinstance(value, (str, int, float, bool))})
                    retention_rows.extend(rows)
                    count_rows.extend(
                        {
                            key: row.get(key, "")
                            for key in (
                                "dataset",
                                "seed",
                                "method",
                                "sample_id",
                                "schema_path",
                                "relation_sequence",
                                "original_count_bounded",
                                "coarse_count_bounded",
                                "path_count_ratio",
                                "log_path_count_error",
                                "count_capped",
                            )
                        }
                        for row in rows
                    )
                    run_status.append({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "available", "sample_count": len(samples), **source})
                except Exception as exc:
                    run_status.append({"dataset": dataset, "seed": int(seed), "method": method, "run_status": "failed", "reason": str(exc), "sample_count": len(samples)})
            write_csv(output / "path_samples.csv", path_samples)
            write_csv(output / "path_retention_per_sample.csv", retention_rows)
            write_csv(output / "path_count_preservation_per_sample.csv", count_rows)
            write_csv(output / "run_status.csv", run_status)
    schema_rows = summarize_metapath_retention(retention_rows)
    write_csv(output / "method_schema_path_retention.csv", schema_rows)
    write_csv(output / "method_dataset_retention.csv", _aggregate_method_dataset(retention_rows))
    write_json(output / "diagnostics.json", diagnostics)


def _aggregate_method_dataset(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    metrics = [
        "typed_exact_step_survival_rate",
        "untyped_step_survival_rate",
        "schema_path_survival_gap",
        "endpoint_pair_collapse_rate",
        "any_consecutive_collapse_rate",
        "unique_cluster_ratio",
        "log_path_count_error",
        "path_weight_missing_step_rate",
    ]
    out = []
    for (dataset, method), group in sorted(groups.items()):
        item = {"dataset": dataset, "method": method, "run_count": len(group)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric, "") not in {"", None} and np.isfinite(float(row[metric]))]
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals else ""
        out.append(item)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--methods", nargs="+", default=["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best", "AH-UGC-style"])
    parser.add_argument("--schema-path-lengths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--max-schema-paths", type=int, default=12)
    parser.add_argument("--max-samples-per-schema", type=int, default=2000)
    parser.add_argument("--max-frontier-per-step", type=int, default=128)
    parser.add_argument("--max-count-frontier-per-step", type=int, default=512)
    parser.add_argument("--max-count-per-endpoint-schema", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--resource-runs", type=Path, default=Path("outputs/exp_next10_hgb_resource_logged_20260517/runs"))
    parser.add_argument("--guard-runs", type=Path, default=Path("outputs/exp_next10_hgb_guard_ablation_actual_20260517/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next12_metapath_retention(
        datasets=args.datasets,
        seeds=args.seeds,
        methods=args.methods,
        schema_path_lengths=args.schema_path_lengths,
        max_schema_paths=int(args.max_schema_paths),
        max_samples_per_schema=int(args.max_samples_per_schema),
        max_frontier_per_step=int(args.max_frontier_per_step),
        max_count_frontier_per_step=int(args.max_count_frontier_per_step),
        max_count_per_endpoint_schema=int(args.max_count_per_endpoint_schema),
        graph_root=args.graph_root,
        resource_runs=args.resource_runs,
        guard_runs=args.guard_runs,
        output=args.output,
    )
    summarize_next12_metapath_retention(input=args.output, output=Path(str(args.output) + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
