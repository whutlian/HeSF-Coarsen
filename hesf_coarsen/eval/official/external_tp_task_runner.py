from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.external_baselines_tp import TP_BASELINE_METHODS, plan_external_tp_rows
from hesf_coarsen.eval.official.freehgc_env_bridge import FREEHGC_REPO_URL, freehgc_preflight


STRICT_EXTERNAL_TP_FIELDS = [
    "dataset",
    "method",
    "baseline_name",
    "method_family",
    "protocol",
    "external_baseline",
    "budget_type",
    "budget_value",
    "support_node_ratio",
    "structural_budget",
    "graph_seed",
    "training_seed",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "training_executed",
    "eligible_for_tp_main_comparison",
    "uses_synthetic_nodes",
    "uses_weighted_edges",
    "requires_loader_adapter",
    "missing_dependency",
    "missing_dependency_name",
    "failure_type",
    "failure_message",
    "success",
    "success_count",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "actual_support_node_ratio",
    "actual_support_edge_ratio",
    "actual_structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "adapter_package_ratio",
    "compress_wall_time_seconds",
    "export_wall_time_seconds",
    "preprocess_wall_time_seconds",
    "train_wall_time_seconds",
    "peak_cpu_rss_mb",
    "peak_gpu_memory_mb",
    "artifact_manifest_path",
    "artifact_estimate_ready",
    "task_result_ready",
    "freehgc_repo_url",
    "suggested_install_command",
]


def build_external_tp_task_rows(
    *,
    dataset: str,
    methods: Iterable[str],
    support_node_ratios: Iterable[float],
    structural_budgets: Iterable[float],
    graph_seeds: Iterable[int],
    training_seeds: Iterable[int],
    freehgc_root: str | Path | None = None,
    native_hgb_root: str | Path | None = Path("external/SeHGNN/data"),
    task_metrics_csv: str | Path | None = None,
    quick: bool = False,
) -> list[dict[str, Any]]:
    metric_rows = _read_metrics(task_metrics_csv)
    selected_methods = list(methods)
    if quick:
        selected_methods = [m for m in selected_methods if m in {"Random-HG-TP", "FreeHGC-TP"}]
        graph_seeds = [next(iter(graph_seeds))]
        training_seeds = [next(iter(training_seeds))]
        support_node_ratios = [next(iter(support_node_ratios))]
        structural_budgets = [next(iter(structural_budgets))]

    rows: list[dict[str, Any]] = []
    preflight = freehgc_preflight(freehgc_root=freehgc_root)
    for budget in support_node_ratios:
        estimate_rows = plan_external_tp_rows(
            dataset=str(dataset).upper(),
            methods=selected_methods,
            budgets=[float(budget)],
            graph_seeds=graph_seeds,
            training_seeds=training_seeds,
            freehgc_root=freehgc_root,
            native_hgb_root=native_hgb_root,
        )
        for row in estimate_rows:
            row["structural_budget"] = ""
            rows.append(_strict_row(row, metric_rows=metric_rows, preflight=preflight))

    for structural_budget in structural_budgets:
        for method in selected_methods:
            if method not in TP_BASELINE_METHODS:
                raise ValueError(f"unsupported TP baseline: {method}")
            for graph_seed in graph_seeds:
                for training_seed in training_seeds:
                    base = {
                        "dataset": str(dataset).upper(),
                        "baseline_name": method,
                        "method": method,
                        "method_family": "external_tp_baseline",
                        "protocol": "schema_preserving_tp",
                        "external_baseline": True,
                        "budget_type": "structural_budget",
                        "budget_value": float(structural_budget),
                        "support_node_ratio": "",
                        "structural_budget": float(structural_budget),
                        "graph_seed": int(graph_seed),
                        "training_seed": int(training_seed),
                        "uses_synthetic_nodes": False,
                        "uses_weighted_edges": False,
                        "requires_loader_adapter": method == "FreeHGC-TP",
                        "success": False,
                        "failure_type": "not_executed",
                        "failure_message": "Structural-budget TP task run is scheduled by Gate21.7 but no official task metric exists locally.",
                    }
                    rows.append(_strict_row(base, metric_rows=metric_rows, preflight=preflight))
    return rows


def summarize_external_tp_by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("method", row.get("baseline_name", ""))), str(row.get("budget_type", ""))), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method, budget_type), group in sorted(groups.items()):
        ready = [row for row in group if _bool(row.get("task_result_ready"))]
        micros = [_float(row.get("test_micro_f1")) for row in ready]
        macros = [_float(row.get("test_macro_f1")) for row in ready]
        first = group[0]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "baseline_name": method,
                "method_family": first.get("method_family", "external_tp_baseline"),
                "protocol": "schema_preserving_tp",
                "external_baseline": True,
                "budget_type": budget_type,
                "runs": len(group),
                "success_count": len(ready),
                "official_hgb_exported": bool(ready) and all(_bool(row.get("official_hgb_exported")) for row in ready),
                "training_executed": bool(ready) and all(_bool(row.get("training_executed")) for row in ready),
                "eligible_for_tp_main_comparison": bool(ready),
                "task_result_ready": bool(ready),
                "test_micro_f1": mean(micros) if micros else "",
                "test_micro_f1_mean": mean(micros) if micros else "",
                "test_micro_f1_std": _std(micros),
                "test_macro_f1": mean(macros) if macros else "",
                "test_macro_f1_mean": mean(macros) if macros else "",
                "test_macro_f1_std": _std(macros),
                "failure_count": len(group) - len(ready),
                "graph_seed_count": len({str(row.get("graph_seed", "")) for row in ready}),
                "training_seed_count": len({str(row.get("training_seed", "")) for row in ready}),
                "requested_budget": first.get("budget_value", ""),
                "actual_support_node_ratio": _mean_field(ready, "actual_support_node_ratio"),
                "actual_support_edge_ratio": _mean_field(ready, "actual_support_edge_ratio"),
                "actual_structural_storage_ratio": _mean_field(ready, "actual_structural_storage_ratio"),
                "raw_hgb_text_byte_ratio": _mean_field(ready, "raw_hgb_text_byte_ratio"),
                "adapter_package_ratio": _mean_field(ready, "adapter_package_ratio"),
                "recovery_micro_mean": _recovery_mean(micros, 0.9533802),
                "recovery_macro_mean": _recovery_mean(macros, 0.9498198),
                "compress_wall_time_seconds_mean": _mean_field(ready, "compress_wall_time_seconds"),
                "export_wall_time_seconds_mean": _mean_field(ready, "export_wall_time_seconds"),
                "preprocess_wall_time_seconds_mean": _mean_field(ready, "preprocess_wall_time_seconds"),
                "train_wall_time_seconds_mean": _mean_field(ready, "train_wall_time_seconds"),
                "peak_cpu_rss_mb_mean": _mean_field(ready, "peak_cpu_rss_mb"),
                "peak_gpu_memory_mb_mean": _mean_field(ready, "peak_gpu_memory_mb"),
                "failure_type": "" if ready else first.get("failure_type", "not_executed"),
                "failure_message": "" if ready else first.get("failure_message", ""),
            }
        )
    return out


def artifact_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", row.get("baseline_name", "")),
            "graph_seed": row.get("graph_seed", ""),
            "training_seed": row.get("training_seed", ""),
            "budget_type": row.get("budget_type", ""),
            "budget_value": row.get("budget_value", ""),
            "official_hgb_exported": row.get("official_hgb_exported", False),
            "training_executed": row.get("training_executed", False),
            "artifact_estimate_ready": row.get("artifact_estimate_ready", False),
            "task_result_ready": row.get("task_result_ready", False),
            "failure_type": row.get("failure_type", ""),
            "failure_message": row.get("failure_message", ""),
        }
        for row in rows
    ]


def failure_log_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if not _bool(row.get("task_result_ready")) or row.get("failure_type")]


def _strict_row(row: Mapping[str, Any], *, metric_rows: dict[tuple[str, str, str, str, str], Mapping[str, str]], preflight: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    method = str(out.get("method", out.get("baseline_name", "")))
    dataset = str(out.get("dataset", "")).upper()
    graph_seed = str(out.get("graph_seed", ""))
    training_seed = str(out.get("training_seed", ""))
    budget_value = str(out.get("budget_value", ""))
    out["method"] = method
    out["baseline_name"] = method
    out["method_family"] = out.get("method_family") or "external_tp_baseline"
    out["protocol"] = "schema_preserving_tp"
    out["external_baseline"] = True
    out["official_sehgnn_unmodified"] = False
    out["uses_weighted_edges"] = bool(out.get("uses_weighted_edges", False))
    out["requires_loader_adapter"] = bool(out.get("requires_loader_adapter", method == "FreeHGC-TP"))
    out["artifact_estimate_ready"] = bool(out.get("success")) and bool(out.get("export_hash", "")) and method != "FreeHGC-TP"
    out["freehgc_repo_url"] = FREEHGC_REPO_URL if method == "FreeHGC-TP" else ""
    out["suggested_install_command"] = preflight.get("suggested_install_command", "") if method == "FreeHGC-TP" else ""
    out["missing_dependency"] = False
    out["missing_dependency_name"] = ""
    if method == "FreeHGC-TP" and preflight.get("missing_dependency"):
        out["missing_dependency"] = True
        out["missing_dependency_name"] = preflight.get("missing_dependency_name", "")
        out["failure_type"] = "missing_external_dependency"
        out["failure_message"] = "FreeHGC upstream preflight failed; FreeHGC-TP is not READY."
    key = (dataset, method, graph_seed, training_seed, budget_value)
    metric = metric_rows.get(key)
    if metric:
        out.update(metric)
    out["actual_support_node_ratio"] = out.get("actual_support_node_ratio") or out.get("support_node_ratio", "")
    out["actual_support_edge_ratio"] = out.get("actual_support_edge_ratio") or out.get("support_edge_ratio", "")
    out["actual_structural_storage_ratio"] = out.get("actual_structural_storage_ratio") or out.get("structural_storage_ratio", "")
    out["raw_hgb_text_byte_ratio"] = out.get("raw_hgb_text_byte_ratio") or out.get("official_text_hgb_byte_ratio", "")
    official_hgb_exported = _bool(out.get("official_hgb_exported"))
    training_executed = _bool(out.get("training_executed"))
    micro = _float_or_nan(out.get("test_micro_f1"))
    macro = _float_or_nan(out.get("test_macro_f1"))
    ready = bool(official_hgb_exported and training_executed and not math.isnan(micro) and not math.isnan(macro))
    out["task_result_ready"] = ready
    out["eligible_for_tp_main_comparison"] = ready
    out["success"] = ready
    out["success_count"] = 1 if ready else 0
    if not ready and not out.get("failure_type"):
        out["failure_type"] = "not_executed"
        out["failure_message"] = "No official SeHGNN task metric row was available for this external TP run."
    for field in STRICT_EXTERNAL_TP_FIELDS:
        out.setdefault(field, "")
    return {field: out.get(field, "") for field in STRICT_EXTERNAL_TP_FIELDS}


def _read_metrics(path: str | Path | None) -> dict[tuple[str, str, str, str, str], Mapping[str, str]]:
    if not path or not Path(path).exists():
        return {}
    out: dict[tuple[str, str, str, str, str], Mapping[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("dataset", "")).upper(),
                str(row.get("method", row.get("baseline_name", ""))),
                str(row.get("graph_seed", "")),
                str(row.get("training_seed", "")),
                str(row.get("budget_value", "")),
            )
            out[key] = row
    return out


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    parsed = _float_or_nan(value)
    return 0.0 if math.isnan(parsed) else parsed


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float_or_nan(row.get(field)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return mean(values) if values else ""


def _std(values: Sequence[float]) -> float | str:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if len(vals) < 2:
        return 0.0 if vals else ""
    mu = mean(vals)
    return float((sum((value - mu) ** 2 for value in vals) / len(vals)) ** 0.5)


def _recovery_mean(values: Sequence[float], full_value: float) -> float | str:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if not vals or float(full_value) == 0.0:
        return ""
    return float(mean(vals) / float(full_value))


def _float_or_nan(value: Any) -> float:
    if value in {"", None}:
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def budget_match_status(row: Mapping[str, Any], *, tolerance: float = 0.02) -> dict[str, Any]:
    budget_type = str(row.get("budget_type", ""))
    requested = _float_or_nan(row.get("requested_budget", row.get("budget_value")))
    actual_field = "actual_structural_storage_ratio" if budget_type == "structural_storage_ratio" else "actual_support_node_ratio"
    actual = _float_or_nan(row.get(actual_field))
    if math.isnan(requested) or math.isnan(actual):
        return {"budget_match_pass": False, "budget_match_status": "missing_budget_or_actual"}
    if budget_type == "structural_storage_ratio":
        passed = abs(actual - requested) <= tolerance
    else:
        passed = actual <= requested + tolerance
    return {
        "budget_match_pass": bool(passed),
        "budget_match_status": "within_tolerance" if passed else "budget_infeasible",
        "budget_match_error": abs(actual - requested),
    }


def external_tp_by_method(rows: Sequence[Mapping[str, Any]], *, required_methods: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in required_methods:
        method_rows = [row for row in rows if str(row.get("method", row.get("baseline_name", ""))) == method]
        ready = [
            row
            for row in method_rows
            if _bool(row.get("training_executed"))
            and _bool(row.get("official_hgb_exported"))
            and _bool(row.get("official_sehgnn_unmodified"))
            and not math.isnan(_float_or_nan(row.get("test_micro_f1")))
            and not math.isnan(_float_or_nan(row.get("test_macro_f1")))
        ]
        graph_seeds = {str(row.get("graph_seed", "")) for row in ready if str(row.get("graph_seed", ""))}
        training_seeds = {str(row.get("training_seed", "")) for row in ready if str(row.get("training_seed", ""))}
        first = ready[0] if ready else (method_rows[0] if method_rows else {})
        out.append(
            {
                "dataset": first.get("dataset", ""),
                "method": method,
                "budget_type": first.get("budget_type", ""),
                "requested_budget": first.get("requested_budget", ""),
                "ready_row_count": len(ready),
                "expected_row_count": 25,
                "graph_seed_count": len(graph_seeds),
                "training_seed_count": len(training_seeds),
                "test_micro_f1_mean": _mean_field(ready, "test_micro_f1"),
                "test_micro_f1_std": _std([_float_or_nan(row.get("test_micro_f1")) for row in ready]),
                "test_macro_f1_mean": _mean_field(ready, "test_macro_f1"),
                "test_macro_f1_std": _std([_float_or_nan(row.get("test_macro_f1")) for row in ready]),
                "actual_structural_storage_ratio_mean": _mean_field(ready, "actual_structural_storage_ratio"),
                "actual_structural_storage_ratio_std": _std([_float_or_nan(row.get("actual_structural_storage_ratio")) for row in ready]),
                "raw_hgb_text_byte_ratio_mean": _mean_field(ready, "raw_hgb_text_byte_ratio"),
                "budget_match_pass": bool(ready) and all(budget_match_status(row)["budget_match_pass"] for row in ready),
                "all_training_executed": bool(ready) and all(_bool(row.get("training_executed")) for row in ready),
                "all_official_hgb_exported": bool(ready) and all(_bool(row.get("official_hgb_exported")) for row in ready),
                "all_official_sehgnn_unmodified": bool(ready) and all(_bool(row.get("official_sehgnn_unmodified")) for row in ready),
                "eligible_for_tp_workload_table": True,
                "ready_5x5_flag": len(graph_seeds) >= 5 and len(training_seeds) >= 5,
            }
        )
    return out
