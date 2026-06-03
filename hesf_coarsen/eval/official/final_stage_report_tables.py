from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, float_value, normalize_dataset


BEST_METHOD_COMPARISON_FIELDS = (
    "dataset",
    "role",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "semantic_structural_ratio",
    "support_edge_ratio",
    "support_node_ratio",
    "raw_hgb_text_byte_ratio",
    "micro_mean",
    "micro_std",
    "macro_mean",
    "macro_std",
    "validation_micro_f1",
    "validation_macro_f1",
    "recovery_micro",
    "recovery_macro",
    "eligible_for_main_decision",
    "diagnostic_only",
    "selection_source",
)

FRONTIER_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "cost_axis",
    "cost_value",
    "semantic_structural_ratio",
    "support_edge_ratio",
    "support_node_ratio",
    "raw_hgb_text_byte_ratio",
    "micro_mean",
    "macro_mean",
    "validation_micro_f1",
    "validation_macro_f1",
    "recovery_micro",
    "recovery_macro",
    "pareto_frontier_flag",
    "dominated_by",
    "eligible_for_main_decision",
)


def build_best_method_comparison(
    rows: Iterable[Mapping[str, Any]],
    *,
    rep_rows: Iterable[Mapping[str, Any]] = (),
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    representatives = [dict(row) for row in rep_rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        dataset_rows = [row for row in source_rows if normalize_dataset(row.get("dataset")) == dataset]
        out.extend(_anchor_rows(dataset_rows))
        hesf_rep = _rep_for_dataset(representatives, dataset, "HeSF-RCS-Rep-Validated")
        if hesf_rep:
            source = _find_method(dataset_rows, str(hesf_rep.get("selected_method", "")))
            if source:
                out.append(_comparison_row(dataset, "Best HeSF-RCS-Rep-Validated", source, selection_source="validation_only_rep"))
        best_validated = _best_by_validation([row for row in dataset_rows if _eligible_compressed(row)])
        if best_validated:
            out.append(_comparison_row(dataset, "Best compressed method validated", best_validated, selection_source="validation_all_compressed"))
        structural = _best_by_validation([row for row in dataset_rows if _is_structural_or_channel_baseline(row)])
        if structural:
            out.append(_comparison_row(dataset, "Best structural/channel baseline validated", structural, selection_source="validation_baseline"))
        external = _best_by_validation([row for row in dataset_rows if str(row.get("method_family", "")) == "external_tp_baseline"])
        if external:
            out.append(_comparison_row(dataset, "Best external TP local baseline validated", external, selection_source="validation_external_tp"))
        selector = _best_by_validation([row for row in dataset_rows if "FreeHGC-score-as-selector" in str(row.get("method", ""))])
        if selector:
            out.append(_comparison_row(dataset, "Best FreeHGC-score selector probe", selector, selection_source="validation_selector_probe"))
        oracle_rep = _rep_for_dataset(representatives, dataset, "TestOracle-Best")
        if oracle_rep:
            source = _find_method(dataset_rows, str(oracle_rep.get("selected_method", "")))
            if source:
                out.append(
                    _comparison_row(
                        dataset,
                        "TestOracle best compressed diagnostic",
                        source,
                        selection_source="test_oracle_diagnostic",
                        eligible_for_main_decision=False,
                        diagnostic_only=True,
                    )
                )
    return out


def build_frontier_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows if _row_has_task_metric(row)]
    dataset_names = [normalize_dataset(item) for item in datasets] if datasets else sorted({normalize_dataset(row.get("dataset")) for row in source_rows})
    out: list[dict[str, Any]] = []
    for dataset in dataset_names:
        dataset_rows = [row for row in source_rows if normalize_dataset(row.get("dataset")) == dataset]
        comparable = [row for row in dataset_rows if _eligible_frontier(row)]
        dominated_by = _dominated_by(comparable)
        for row in sorted(dataset_rows, key=lambda item: (_cost(item), -(_metric(item, "test_micro_f1_mean") or -1.0), str(item.get("method", "")))):
            method = str(row.get("method", ""))
            cost_axis, cost = _cost_axis(row)
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_family": row.get("method_family", ""),
                    "requested_budget_type": row.get("requested_budget_type", ""),
                    "requested_budget": row.get("requested_budget", ""),
                    "cost_axis": cost_axis,
                    "cost_value": cost,
                    "semantic_structural_ratio": _first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio"),
                    "support_edge_ratio": _first_value(row, "actual_support_edge_ratio", "support_edge_ratio"),
                    "support_node_ratio": _first_value(row, "actual_support_node_ratio", "support_node_ratio"),
                    "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
                    "micro_mean": _first_value(row, "test_micro_f1_mean", "test_micro_f1"),
                    "macro_mean": _first_value(row, "test_macro_f1_mean", "test_macro_f1"),
                    "validation_micro_f1": _first_value(row, "validation_micro_f1_mean", "validation_micro_f1"),
                    "validation_macro_f1": _first_value(row, "validation_macro_f1_mean", "validation_macro_f1"),
                    "recovery_micro": row.get("recovery_vs_native_full_micro", row.get("recovery_micro", "")),
                    "recovery_macro": row.get("recovery_vs_native_full_macro", row.get("recovery_macro", "")),
                    "pareto_frontier_flag": method in comparable and method not in dominated_by,
                    "dominated_by": dominated_by.get(method, ""),
                    "eligible_for_main_decision": bool(_eligible_compressed(row) or _is_full_anchor(row)),
                }
            )
    return out


def _anchor_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method, role in (
        ("Full-native-SeHGNN", "Full-native anchor"),
        ("Export-full-SeHGNN", "Export-full anchor"),
    ):
        row = _find_method(rows, method)
        if row:
            out.append(_comparison_row(normalize_dataset(row.get("dataset")), role, row, selection_source="anchor", eligible_for_main_decision=True))
    return out


def _comparison_row(
    dataset: str,
    role: str,
    row: Mapping[str, Any],
    *,
    selection_source: str,
    eligible_for_main_decision: bool | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "role": role,
        "method": row.get("method", ""),
        "method_family": row.get("method_family", ""),
        "requested_budget_type": row.get("requested_budget_type", ""),
        "requested_budget": row.get("requested_budget", ""),
        "semantic_structural_ratio": _first_value(row, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio"),
        "support_edge_ratio": _first_value(row, "actual_support_edge_ratio", "support_edge_ratio"),
        "support_node_ratio": _first_value(row, "actual_support_node_ratio", "support_node_ratio"),
        "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
        "micro_mean": _first_value(row, "test_micro_f1_mean", "test_micro_f1"),
        "micro_std": row.get("test_micro_f1_std", ""),
        "macro_mean": _first_value(row, "test_macro_f1_mean", "test_macro_f1"),
        "macro_std": row.get("test_macro_f1_std", ""),
        "validation_micro_f1": _first_value(row, "validation_micro_f1_mean", "validation_micro_f1"),
        "validation_macro_f1": _first_value(row, "validation_macro_f1_mean", "validation_macro_f1"),
        "recovery_micro": row.get("recovery_vs_native_full_micro", row.get("recovery_micro", "")),
        "recovery_macro": row.get("recovery_vs_native_full_macro", row.get("recovery_macro", "")),
        "eligible_for_main_decision": bool_value(row.get("eligible_for_main_decision", row.get("eligible_for_decision", True)))
        if eligible_for_main_decision is None
        else bool(eligible_for_main_decision),
        "diagnostic_only": bool(diagnostic_only),
        "selection_source": selection_source,
    }


def _dominated_by(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        method = str(row.get("method", ""))
        cost = _cost(row)
        micro = _metric(row, "test_micro_f1_mean")
        macro = _metric(row, "test_macro_f1_mean")
        if micro is None or macro is None:
            continue
        for other in rows:
            if other is row:
                continue
            other_cost = _cost(other)
            other_micro = _metric(other, "test_micro_f1_mean")
            other_macro = _metric(other, "test_macro_f1_mean")
            if other_micro is None or other_macro is None:
                continue
            if (
                other_cost <= cost + 1.0e-12
                and other_micro >= micro - 1.0e-12
                and other_macro >= macro - 1.0e-12
                and (other_cost < cost - 1.0e-12 or other_micro > micro + 1.0e-12 or other_macro > macro + 1.0e-12)
            ):
                out[method] = str(other.get("method", ""))
                break
    return out


def _best_by_validation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if float_value(row.get("validation_micro_f1_mean", row.get("validation_micro_f1"))) is not None
        and float_value(row.get("validation_macro_f1_mean", row.get("validation_macro_f1"))) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float_value(row.get("validation_micro_f1_mean", row.get("validation_micro_f1"))) or -1.0,
            float_value(row.get("validation_macro_f1_mean", row.get("validation_macro_f1"))) or -1.0,
            -_cost(row),
        ),
    )


def _eligible_compressed(row: Mapping[str, Any]) -> bool:
    return bool(
        not _is_full_anchor(row)
        and bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("success", True))
        and bool_value(row.get("training_executed", True))
        and not bool_value(row.get("constraint_safe_fallback"))
        and not bool_value(row.get("uses_test_for_selection"))
    )


def _eligible_frontier(row: Mapping[str, Any]) -> bool:
    return _eligible_compressed(row) and _cost(row) < 1.0 + 1.0e-12 and _row_has_task_metric(row)


def _is_full_anchor(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")) in {"Full-native-SeHGNN", "Export-full-SeHGNN"}


def _row_has_task_metric(row: Mapping[str, Any]) -> bool:
    return finite_metric(_first_value(row, "test_micro_f1_mean", "test_micro_f1")) and finite_metric(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))


def _is_structural_or_channel_baseline(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    family = str(row.get("method_family", ""))
    return bool(
        family in {"relation_structural_baseline", "channel_baseline"}
        or any(token in method for token in ("Random", "Degree", "Proportional", "ValidationGreedy", "MDfull"))
    )


def _find_method(rows: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("method", "")) == method:
            return row
    return None


def _rep_for_dataset(rows: Sequence[Mapping[str, Any]], dataset: str, rep_type: str) -> Mapping[str, Any] | None:
    for row in rows:
        if normalize_dataset(row.get("dataset")) == dataset and str(row.get("rep_type", "")) == rep_type and str(row.get("selected_method", "")):
            return row
    return None


def _cost_axis(row: Mapping[str, Any]) -> tuple[str, float | str]:
    for field in (
        "semantic_structural_storage_ratio",
        "actual_semantic_structural_ratio",
        "actual_support_edge_ratio",
        "support_edge_ratio",
        "channel_edge_ratio",
        "keyword_feature_ratio",
        "raw_hgb_text_byte_ratio",
    ):
        value = float_value(row.get(field))
        if value is not None:
            return field, value
    return "", ""


def _cost(row: Mapping[str, Any]) -> float:
    _, value = _cost_axis(row)
    return float(value) if value != "" else 999.0


def _metric(row: Mapping[str, Any], field: str) -> float | None:
    if field == "test_micro_f1_mean":
        return float_value(_first_value(row, "test_micro_f1_mean", "test_micro_f1"))
    if field == "test_macro_f1_mean":
        return float_value(_first_value(row, "test_macro_f1_mean", "test_macro_f1"))
    return float_value(row.get(field))


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return value
    return ""
