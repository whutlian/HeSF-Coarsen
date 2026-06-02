from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import DATASETS, bool_value, float_value, normalize_dataset


def select_gate21_17_representatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = DATASETS,
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        candidates = [
            row
            for row in source_rows
            if normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")).startswith("HeSF-RCS-auto")
            and bool_value(row.get("eligible_for_main_table", True))
            and bool_value(row.get("success"))
            and bool_value(row.get("training_executed"))
        ]
        main = _select_main_rep(candidates)
        oracle = _select_test_oracle(candidates)
        if main is None:
            out.append(
                {
                    "dataset": dataset,
                    "method": "HeSF-RCS-Rep",
                    "source_method": "",
                    "selected_as_rep": False,
                    "selection_source": "validation_metric_missing",
                    "rep_selection_confidence": "missing",
                    "uses_test_for_selection": False,
                    "eligible_for_main_table": False,
                    "eligible_for_decision": False,
                    "failure_type": "validation_metric_missing",
                    "failure_reason": "No successful HeSF-RCS-auto row has actual validation metrics or validation proxy.",
                }
            )
        else:
            source, selection_source = main
            out.append(_rep_row(source, method="HeSF-RCS-Rep", selection_source=selection_source, diagnostic=False))
        if oracle is not None:
            out.append(_rep_row(oracle, method="HeSF-RCS-TestOracleRep", selection_source="test_oracle_diagnostic_only", diagnostic=True))
    return out


def _select_main_rep(candidates: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str] | None:
    with_actual = [row for row in candidates if float_value(row.get("validation_micro_f1_mean")) is not None and float_value(row.get("validation_macro_f1_mean")) is not None]
    if with_actual:
        return (
            max(
                with_actual,
                key=lambda row: (
                    float_value(row.get("validation_micro_f1_mean")) or -1.0,
                    float_value(row.get("validation_macro_f1_mean")) or -1.0,
                    -(float_value(row.get("actual_structural_storage_ratio")) or 999.0),
                ),
            ),
            "actual_validation",
        )
    with_proxy = [row for row in candidates if float_value(row.get("validation_proxy_score")) is not None]
    if with_proxy:
        return (
            max(
                with_proxy,
                key=lambda row: (
                    float_value(row.get("validation_proxy_score")) or -1.0,
                    -(float_value(row.get("actual_structural_storage_ratio")) or 999.0),
                ),
            ),
            "validation_proxy",
        )
    return None


def _select_test_oracle(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    with_test = [row for row in candidates if float_value(row.get("test_micro_f1_mean")) is not None and float_value(row.get("test_macro_f1_mean")) is not None]
    if not with_test:
        return None
    return max(
        with_test,
        key=lambda row: (
            float_value(row.get("test_micro_f1_mean")) or -1.0,
            float_value(row.get("test_macro_f1_mean")) or -1.0,
            -(float_value(row.get("actual_structural_storage_ratio")) or 999.0),
        ),
    )


def _rep_row(source: Mapping[str, Any], *, method: str, selection_source: str, diagnostic: bool) -> dict[str, Any]:
    return {
        "dataset": normalize_dataset(source.get("dataset")),
        "method": method,
        "source_method": source.get("method", ""),
        "method_family": "schema_preserving_rcs_diagnostic" if diagnostic else "schema_preserving_rcs",
        "requested_budget_type": source.get("requested_budget_type", ""),
        "requested_budget": source.get("requested_budget", ""),
        "actual_structural_storage_ratio": source.get("actual_structural_storage_ratio", ""),
        "support_node_ratio": source.get("support_node_ratio", ""),
        "support_edge_ratio": source.get("support_edge_ratio", ""),
        "raw_hgb_text_byte_ratio": source.get("raw_hgb_text_byte_ratio", ""),
        "test_micro_f1_mean": source.get("test_micro_f1_mean", ""),
        "test_macro_f1_mean": source.get("test_macro_f1_mean", ""),
        "validation_micro_f1_mean": source.get("validation_micro_f1_mean", ""),
        "validation_macro_f1_mean": source.get("validation_macro_f1_mean", ""),
        "validation_proxy_score": source.get("validation_proxy_score", ""),
        "selected_edge_hash": source.get("selected_edge_hash", ""),
        "planner_config_hash": source.get("planner_config_hash", ""),
        "source_path": source.get("source_path", ""),
        "training_executed": source.get("training_executed", True),
        "success": source.get("success", True),
        "official_hgb_exported": source.get("official_hgb_exported", True),
        "official_sehgnn_unmodified": source.get("official_sehgnn_unmodified", True),
        "schema_compatible": source.get("schema_compatible", True),
        "target_preserving": source.get("target_preserving", True),
        "selection_source": selection_source,
        "rep_selection_confidence": "actual_validation" if selection_source == "actual_validation" else "proxy_only" if selection_source == "validation_proxy" else "diagnostic_only",
        "uses_test_for_selection": bool(diagnostic),
        "eligible_for_main_table": not diagnostic,
        "eligible_for_decision": not diagnostic,
        "selected_as_rep": True,
    }
