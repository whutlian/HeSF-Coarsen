from __future__ import annotations

import argparse
import csv
import sys
from time import perf_counter
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from experiments.scripts._common import markdown_table, write_csv
from hesf_coarsen.eval.spectral_diagnostics import (
    _target_matched_baseline,
    compute_spectral_diagnostics,
)
from hesf_coarsen.eval.task_gnn import evaluate_rgcn_task
from hesf_coarsen.io.edge_list import load_graph


DEFAULT_BASELINES = ("random", "heavy_edge", "graphzoom_style", "convmatch_style")
REFINE_EPOCHS = (0, 1, 3, 5)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _relation_weights_from_row(graph: Any, row: Mapping[str, Any]) -> dict[int, float]:
    weights: dict[int, float] = {}
    prefix = "fusion.relation_weight_by_relation."
    for key, value in row.items():
        if str(key).startswith(prefix):
            try:
                weights[int(str(key).removeprefix(prefix))] = float(value)
            except (TypeError, ValueError):
                continue
    if weights:
        return weights
    relation_ids = sorted(int(relation_id) for relation_id in graph.relations)
    if not relation_ids:
        return {}
    uniform = 1.0 / len(relation_ids)
    return {relation_id: uniform for relation_id in relation_ids}


def _graph_dir(graph_root: Path, dataset: str) -> Path:
    candidate = graph_root / f"{dataset.lower()}_hesf"
    if candidate.exists():
        return candidate
    return graph_root / dataset.lower()


def _field_prefix(baseline: str) -> str:
    return f"baseline_{baseline.replace('-', '_')}_"


def _has_baseline_fields(source: Mapping[str, Any], baseline: str) -> bool:
    prefix = _field_prefix(baseline)
    return any(str(key).startswith(prefix) for key in source)


def _baseline_row(source: Mapping[str, Any], baseline: str, refine_epochs: Iterable[int]) -> dict[str, Any]:
    prefix = _field_prefix(baseline)
    target_hit = _first(source, (f"{prefix}target_hit",), "")
    row: dict[str, Any] = {
        "run_name": source.get("run_name", ""),
        "run_dir": source.get("run_dir", ""),
        "dataset": source.get("dataset", ""),
        "variant": source.get("variant", ""),
        "seed": source.get("seed", ""),
        "target_ratio": source.get("target_ratio", ""),
        "baseline": baseline,
        "baseline_target_hit": target_hit,
        "baseline_target_abs_error": _first(source, (f"{prefix}target_abs_error",), ""),
        "baseline_final_cumulative_ratio": _first(source, (f"{prefix}final_cumulative_ratio",), ""),
        "baseline_cumulative_dee": _first(source, (f"{prefix}cumulative_dee",), ""),
        "baseline_cumulative_fwe_weighted": _first(source, (f"{prefix}cumulative_fwe_weighted",), ""),
        "baseline_cumulative_fse_unweighted": _first(source, (f"{prefix}cumulative_fse_unweighted",), ""),
        "baseline_cumulative_ree_max": _first(source, (f"{prefix}cumulative_ree_max",), ""),
        "baseline_cumulative_sipe": _first(source, (f"{prefix}cumulative_sipe",), ""),
        "baseline_cumulative_sampled_eigen_error": _first(source, (f"{prefix}cumulative_sampled_eigen_error",), ""),
        "baseline_projected_macro_f1": _first(
            source,
            (
                f"{prefix}task_projected_macro_f1",
                f"{prefix}projected_macro_f1",
            ),
        ),
        "baseline_refined_macro_f1": _first(
            source,
            (
                f"{prefix}task_refined_macro_f1",
                f"{prefix}refined_macro_f1",
            ),
        ),
        "baseline_runtime_total": _first(source, (f"{prefix}runtime_total", f"{prefix}task_total_time"), ""),
        "baseline_task_train_time": _first(source, (f"{prefix}task_train_time",), ""),
        "baseline_task_refine_time": _first(source, (f"{prefix}task_refine_time",), ""),
        "baseline_task_total_time": _first(source, (f"{prefix}task_total_time",), ""),
        "baseline_task_best_refined_macro_f1": _first(source, (f"{prefix}task_best_refined_macro_f1",), ""),
        "baseline_task_best_refined_epoch": _first(source, (f"{prefix}task_best_refined_epoch",), ""),
        "baseline_task_refine_auc_macro_f1": _first(source, (f"{prefix}task_refine_auc_macro_f1",), ""),
        "baseline_task_skipped": _first(source, (f"{prefix}task_skipped",), ""),
        "baseline_task_skip_reason": _first(source, (f"{prefix}task_skip_reason",), ""),
    }
    for epoch in refine_epochs:
        row[f"baseline_projected_macro_f1@{epoch}"] = _first(
            source,
            (
                f"{prefix}task_projected_macro_f1@{epoch}",
                f"{prefix}projected_macro_f1@{epoch}",
                f"{prefix}task_projected_original_macro_f1@{epoch}",
            ),
        )
        row[f"baseline_refined_macro_f1@{epoch}"] = _first(
            source,
            (
                f"{prefix}task_refined_macro_f1@{epoch}",
                f"{prefix}refined_macro_f1@{epoch}",
                f"{prefix}task_refined_original_macro_f1@{epoch}",
            ),
        )
    row["comparison_status"] = "included" if _truthy(target_hit) else "failed target control"
    return row


def _baseline_row_from_metrics(
    source: Mapping[str, Any],
    baseline: str,
    metrics: Mapping[str, Any],
    refine_epochs: Iterable[int],
) -> dict[str, Any]:
    target_hit = bool(metrics.get("target_hit", False))
    row: dict[str, Any] = {
        "run_name": source.get("run_name", ""),
        "run_dir": source.get("run_dir", ""),
        "dataset": source.get("dataset", ""),
        "variant": source.get("variant", ""),
        "seed": source.get("seed", ""),
        "target_ratio": metrics.get("target_ratio", source.get("target_ratio", "")),
        "baseline": baseline,
        "baseline_target_hit": target_hit,
        "baseline_target_abs_error": metrics.get("target_abs_error", ""),
        "baseline_final_cumulative_ratio": metrics.get("final_cumulative_ratio", ""),
        "baseline_cumulative_dee": metrics.get("dirichlet_energy_relative_error", ""),
        "baseline_cumulative_fwe_weighted": metrics.get("relation_weighted_fused_energy_relative_error", ""),
        "baseline_cumulative_fse_unweighted": metrics.get("fused_sketch_energy_relative_error", ""),
        "baseline_cumulative_ree_max": metrics.get("relation_energy_relative_error_max", ""),
        "baseline_cumulative_sipe": metrics.get("chebheat_sketch_inner_product_relative_error", ""),
        "baseline_cumulative_sampled_eigen_error": (
            (metrics.get("exact_eigenvalue_sanity") or {}).get("relative_error", "")
            if isinstance(metrics.get("exact_eigenvalue_sanity"), Mapping)
            else ""
        ),
        "baseline_projected_macro_f1": metrics.get("task_projected_macro_f1", ""),
        "baseline_refined_macro_f1": metrics.get("task_refined_macro_f1", ""),
        "baseline_runtime_total": metrics.get("runtime_total", ""),
        "baseline_task_train_time": metrics.get("task_train_time", ""),
        "baseline_task_refine_time": metrics.get("task_refine_time", ""),
        "baseline_task_total_time": metrics.get("task_total_time", ""),
        "baseline_task_best_refined_macro_f1": metrics.get("task_best_refined_macro_f1", ""),
        "baseline_task_best_refined_epoch": metrics.get("task_best_refined_epoch", ""),
        "baseline_task_refine_auc_macro_f1": metrics.get("task_refine_auc_macro_f1", ""),
        "baseline_task_skipped": metrics.get("task_skipped", ""),
        "baseline_task_skip_reason": metrics.get("task_skip_reason", ""),
        "comparison_status": "included" if target_hit else "failed target control",
        "baseline_status": metrics.get("status", "computed"),
        "baseline_failure_reason": metrics.get("reason", ""),
        "baseline_stopped_by": metrics.get("stopped_by", ""),
        "baseline_levels": metrics.get("levels", ""),
    }
    for epoch in refine_epochs:
        row[f"baseline_projected_macro_f1@{epoch}"] = metrics.get(
            f"task_projected_macro_f1@{epoch}",
            metrics.get("task_projected_macro_f1", ""),
        )
        row[f"baseline_refined_macro_f1@{epoch}"] = metrics.get(
            f"task_refined_macro_f1@{epoch}",
            "",
        )
    return row


def _compute_baseline_row(
    source: Mapping[str, Any],
    baseline: str,
    *,
    graph_root: Path,
    target_tolerance: float,
    baseline_max_levels: int,
    baseline_max_nodes: int | None,
    spectral_num_signals: int,
    spectral_smoothing_steps: int,
    spectral_exact_eigenvalue_max_nodes: int | None,
    task_eval: bool,
    task_epochs: int,
    task_hidden_dim: int,
    task_device: str,
    refine_epochs: list[int],
) -> dict[str, Any]:
    dataset = str(source.get("dataset", "") or "")
    target_ratio = _as_float(source.get("target_ratio"), None)
    if target_ratio is None:
        target_ratio = _as_float(source.get("final_cumulative_ratio"), None)
    if target_ratio is None:
        target_ratio = _as_float(source.get("final_ratio"), None)
    if not dataset or target_ratio is None:
        return _baseline_row_from_metrics(
            source,
            baseline,
            {
                "status": "skipped",
                "target_hit": False,
                "target_abs_error": "",
                "reason": "missing_dataset_or_target_ratio",
            },
            refine_epochs,
        )
    original = load_graph(_graph_dir(graph_root, dataset))
    if baseline_max_nodes is not None and baseline_max_nodes > 0 and original.num_nodes > baseline_max_nodes:
        return _baseline_row_from_metrics(
            source,
            baseline,
            {
                "status": "skipped",
                "target_ratio": target_ratio,
                "target_hit": False,
                "target_abs_error": "",
                "reason": "node_count_exceeds_limit",
            },
            refine_epochs,
        )
    seed = int(float(source.get("seed", 12345) or 12345))
    relation_weights = _relation_weights_from_row(original, source)
    start = perf_counter()
    coarse, assignment, control = _target_matched_baseline(
        original,
        baseline,
        target_ratio=float(target_ratio),
        target_tolerance=float(target_tolerance),
        max_levels=int(baseline_max_levels),
        seed=seed,
        relation_weights=relation_weights,
        dim=int(spectral_num_signals),
    )
    metrics = compute_spectral_diagnostics(
        original,
        coarse,
        assignment,
        seed=seed,
        num_signals=int(spectral_num_signals),
        smoothing_steps=int(spectral_smoothing_steps),
        relation_weights=relation_weights,
        exact_eigenvalue_max_nodes=spectral_exact_eigenvalue_max_nodes,
        baseline_methods=None,
    )
    payload: dict[str, Any] = {
        "status": "computed",
        "coarse_nodes": int(coarse.num_nodes),
        "final_cumulative_ratio": float(coarse.num_nodes / max(original.num_nodes, 1)),
        "matched_pairs": int(np.sum(assignment.cluster_sizes() == 2)),
        **control,
        "dirichlet_energy_relative_error": metrics.get("dirichlet_energy_relative_error", ""),
        "relation_weighted_fused_energy_relative_error": metrics.get(
            "relation_weighted_fused_energy_relative_error",
            "",
        ),
        "fused_sketch_energy_relative_error": metrics.get("fused_sketch_energy_relative_error", ""),
        "relation_energy_relative_error_max": metrics.get("relation_energy_relative_error_max", ""),
        "chebheat_sketch_inner_product_relative_error": metrics.get(
            "chebheat_sketch_inner_product_relative_error",
            "",
        ),
        "runtime_total": float(perf_counter() - start),
        "exact_eigenvalue_sanity": metrics.get("exact_eigenvalue_sanity", {}),
    }
    if task_eval and bool(control.get("target_hit", False)):
        task_result = evaluate_rgcn_task(
            original,
            coarse,
            assignment.assignment,
            seed=seed,
            epochs=int(task_epochs),
            refine_epochs=max(refine_epochs) if refine_epochs else 0,
            refine_epochs_list=refine_epochs,
            hidden_dim=int(task_hidden_dim),
            device=task_device,
        ).metrics
        payload.update(
            {
                "task_projected_macro_f1": task_result.get("projected_original_macro_f1", ""),
                "task_refined_macro_f1": task_result.get("refined_original_macro_f1", ""),
                "task_train_time": task_result.get("train_time", ""),
                "task_refine_time": task_result.get("refine_time", ""),
                "task_total_time": task_result.get("total_time", ""),
                "task_best_refined_macro_f1": task_result.get("best_refined_macro_f1", ""),
                "task_best_refined_epoch": task_result.get("best_refined_epoch", ""),
                "task_refine_auc_macro_f1": task_result.get("refine_auc_macro_f1", ""),
                "task_skipped": task_result.get("skipped", False),
                "task_skip_reason": task_result.get("skip_reason", ""),
            }
        )
        for epoch in refine_epochs:
            payload[f"task_refined_macro_f1@{epoch}"] = task_result.get(
                f"refined_original_macro_f1@{epoch}",
                "",
            )
            payload[f"task_projected_macro_f1@{epoch}"] = task_result.get(
                f"projected_original_macro_f1@{epoch}",
                task_result.get("projected_original_macro_f1", ""),
            )
    payload["runtime_total"] = float(perf_counter() - start)
    return _baseline_row_from_metrics(source, baseline, payload, refine_epochs)


def _wide_rows(source_rows: list[dict[str, str]], baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in source_rows:
        key = (
            str(row.get("run_name", "")),
            str(row.get("dataset", "")),
            str(row.get("variant", "")),
            str(row.get("seed", "")),
        )
        by_run[key] = dict(row)
    for row in baseline_rows:
        key = (
            str(row.get("run_name", "")),
            str(row.get("dataset", "")),
            str(row.get("variant", "")),
            str(row.get("seed", "")),
        )
        wide = by_run.setdefault(key, {})
        prefix = _field_prefix(str(row.get("baseline", "")))
        for key_name, value in row.items():
            if key_name.startswith("baseline_"):
                wide[f"{prefix}{key_name.removeprefix('baseline_')}"] = value
    return list(by_run.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Next4 cumulative baseline comparisons.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--task-eval", action="store_true")
    parser.add_argument("--refine-epochs", type=int, nargs="+", default=list(REFINE_EPOCHS))
    parser.add_argument("--graph-root", type=Path, default=Path("data"))
    parser.add_argument("--target-tolerance", type=float, default=0.02)
    parser.add_argument("--baseline-max-levels", type=int, default=4)
    parser.add_argument("--baseline-max-nodes", type=int, default=0)
    parser.add_argument("--spectral-num-signals", type=int, default=4)
    parser.add_argument("--spectral-smoothing-steps", type=int, default=1)
    parser.add_argument("--spectral-exact-eigenvalue-max-nodes", type=int, default=256)
    parser.add_argument("--task-epochs", type=int, default=20)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--task-device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_rows = _read_csv(args.summary)
    wanted_variants = None if args.variants is None else {str(value) for value in args.variants}
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        if wanted_variants is not None and str(source.get("variant", "")) not in wanted_variants:
            continue
        for baseline in args.baselines:
            baseline_name = str(baseline)
            if _has_baseline_fields(source, baseline_name):
                rows.append(_baseline_row(source, baseline_name, args.refine_epochs))
                continue
            rows.append(
                _compute_baseline_row(
                    source,
                    baseline_name,
                    graph_root=args.graph_root,
                    target_tolerance=float(args.target_tolerance),
                    baseline_max_levels=int(args.baseline_max_levels),
                    baseline_max_nodes=(
                        None if int(args.baseline_max_nodes) <= 0 else int(args.baseline_max_nodes)
                    ),
                    spectral_num_signals=int(args.spectral_num_signals),
                    spectral_smoothing_steps=int(args.spectral_smoothing_steps),
                    spectral_exact_eigenvalue_max_nodes=(
                        None
                        if int(args.spectral_exact_eigenvalue_max_nodes) <= 0
                        else int(args.spectral_exact_eigenvalue_max_nodes)
                    ),
                    task_eval=bool(args.task_eval),
                    task_epochs=int(args.task_epochs),
                    task_hidden_dim=int(args.task_hidden_dim),
                    task_device=str(args.task_device),
                    refine_epochs=[int(epoch) for epoch in args.refine_epochs],
                )
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "baseline_summary.csv", rows)
    write_csv(args.output / "final_summary_with_baselines.csv", _wide_rows(source_rows, rows))
    failed_count = sum(1 for row in rows if row.get("comparison_status") == "failed target control")
    included_count = sum(1 for row in rows if row.get("comparison_status") == "included")
    report = [
        "# Next4 Baseline Summary",
        "",
        f"Rows: {len(rows)}",
        f"Included: {included_count}",
        f"Failed target control: {failed_count}",
        "",
        markdown_table(
            rows[:20],
            [
                "run_name",
                "dataset",
                "variant",
                "baseline",
                "baseline_target_hit",
                "baseline_target_abs_error",
                "baseline_final_cumulative_ratio",
                "comparison_status",
            ],
        ),
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
