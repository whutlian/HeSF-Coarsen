from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


GATE21_21_FRONTIER_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "cost_axis",
    "cost_value",
    "requested_budget_type",
    "requested_budget",
    "test_micro_f1_mean",
    "test_macro_f1_mean",
    "recovery_micro",
    "recovery_macro",
    "pareto_by_micro",
    "pareto_by_macro",
    "pareto_by_micro_macro_joint",
    "pareto_by_recovery",
    "dominated_by_micro",
    "dominated_by_macro",
    "dominated_by_micro_macro_joint",
    "dominated_by_recovery",
    "eligible_for_main_decision",
)

COST_AXES = (
    "semantic_structural_storage_ratio",
    "actual_support_edge_ratio",
    "raw_hgb_text_byte_ratio",
)


def build_gate21_21_frontier_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
    cost_axes: Sequence[str] = COST_AXES,
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows if _has_metrics(row)]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        dataset_rows = [row for row in source_rows if normalize_dataset(row.get("dataset")) == dataset and _eligible(row)]
        for axis in cost_axes:
            axis_rows = [row for row in dataset_rows if float_value(row.get(axis)) is not None]
            dominated_micro = _dominators(axis_rows, axis=axis, metrics=("test_micro_f1_mean",))
            dominated_macro = _dominators(axis_rows, axis=axis, metrics=("test_macro_f1_mean",))
            dominated_joint = _dominators(axis_rows, axis=axis, metrics=("test_micro_f1_mean", "test_macro_f1_mean"))
            dominated_recovery = _dominators(axis_rows, axis=axis, metrics=("recovery_vs_native_full_micro", "recovery_vs_native_full_macro"))
            for row in sorted(axis_rows, key=lambda item: (float_value(item.get(axis)) or 999.0, str(item.get("method", "")))):
                method = str(row.get("method", ""))
                out.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "method_family": row.get("method_family", ""),
                        "cost_axis": axis,
                        "cost_value": row.get(axis, ""),
                        "requested_budget_type": row.get("requested_budget_type", ""),
                        "requested_budget": row.get("requested_budget", ""),
                        "test_micro_f1_mean": _first_value(row, "test_micro_f1_mean", "test_micro_f1"),
                        "test_macro_f1_mean": _first_value(row, "test_macro_f1_mean", "test_macro_f1"),
                        "recovery_micro": _first_value(row, "recovery_vs_native_full_micro", "recovery_micro"),
                        "recovery_macro": _first_value(row, "recovery_vs_native_full_macro", "recovery_macro"),
                        "pareto_by_micro": method not in dominated_micro,
                        "pareto_by_macro": method not in dominated_macro,
                        "pareto_by_micro_macro_joint": method not in dominated_joint,
                        "pareto_by_recovery": method not in dominated_recovery,
                        "dominated_by_micro": dominated_micro.get(method, ""),
                        "dominated_by_macro": dominated_macro.get(method, ""),
                        "dominated_by_micro_macro_joint": dominated_joint.get(method, ""),
                        "dominated_by_recovery": dominated_recovery.get(method, ""),
                        "eligible_for_main_decision": bool_value(row.get("eligible_for_main_decision", row.get("eligible_for_main_table", True))),
                    }
                )
    return out


def _dominators(rows: Sequence[Mapping[str, Any]], *, axis: str, metrics: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    comparable = [row for row in rows if _metrics_available(row, metrics)]
    for row in comparable:
        method = str(row.get("method", ""))
        cost = float_value(row.get(axis))
        if cost is None:
            continue
        metric_values = [_metric(row, metric) for metric in metrics]
        for other in comparable:
            if other is row:
                continue
            other_cost = float_value(other.get(axis))
            if other_cost is None:
                continue
            other_values = [_metric(other, metric) for metric in metrics]
            if other_cost <= cost + 1.0e-12 and all((o or -1.0) >= (v or -1.0) - 1.0e-12 for o, v in zip(other_values, metric_values)):
                strict_cost = other_cost < cost - 1.0e-12
                strict_metric = any((o or -1.0) > (v or -1.0) + 1.0e-12 for o, v in zip(other_values, metric_values))
                if strict_cost or strict_metric:
                    out[method] = str(other.get("method", ""))
                    break
    return out


def _eligible(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and bool_value(row.get("schema_compatible", True))
        and bool_value(row.get("official_hgb_exported", True))
        and bool_value(row.get("official_sehgnn_unmodified", True))
        and not bool_value(row.get("constraint_safe_fallback"))
        and not bool_value(row.get("uses_weighted_superedges"))
        and not bool_value(row.get("uses_synthetic_target_nodes"))
    )


def _has_metrics(row: Mapping[str, Any]) -> bool:
    return finite_metric(_first_value(row, "test_micro_f1_mean", "test_micro_f1")) and finite_metric(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))


def _metrics_available(row: Mapping[str, Any], metrics: Sequence[str]) -> bool:
    return all(_metric(row, metric) is not None for metric in metrics)


def _metric(row: Mapping[str, Any], metric: str) -> float | None:
    if metric == "test_micro_f1_mean":
        return float_value(_first_value(row, "test_micro_f1_mean", "test_micro_f1"))
    if metric == "test_macro_f1_mean":
        return float_value(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))
    if metric == "recovery_vs_native_full_micro":
        return float_value(_first_value(row, "recovery_vs_native_full_micro", "recovery_micro"))
    if metric == "recovery_vs_native_full_macro":
        return float_value(_first_value(row, "recovery_vs_native_full_macro", "recovery_macro"))
    return float_value(row.get(metric))


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""
