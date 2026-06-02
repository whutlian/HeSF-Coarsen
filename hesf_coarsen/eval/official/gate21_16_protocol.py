from __future__ import annotations

from typing import Any, Mapping

from hesf_coarsen.eval.official.stage_report_protocol import (
    DATASETS,
    EXTERNAL_TP_BASELINES,
    FULL_METHODS,
    INTERNAL_BASELINES,
    STRUCTURAL_BASELINES,
    STRUCTURAL_BUDGETS,
    SUPPORT_NODE_BUDGETS,
    bool_value,
    finite_metric,
    float_value,
    normalize_dataset,
)


GATE21_16_MAIN_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "actual_structural_storage_ratio",
    "support_node_ratio",
    "support_edge_ratio",
    "raw_hgb_text_byte_ratio",
    "graph_seed_count",
    "training_seed_count",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "validation_proxy_score",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "schema_compatible",
    "target_preserving",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "training_executed",
    "eligible_for_main_table",
    "success",
    "failure_type",
    "failure_reason",
    "selected_edge_hash",
    "planner_config_hash",
    "source_path",
    "repo_url",
)

GATE21_16_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "ACM_EXPORT_CONSISTENCY_PASS",
    "IMDB_EXPORT_CONSISTENCY_PASS",
    "STRUCTURAL_BASELINES_EXECUTED_BY_DATASET",
    "EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET",
    "EXTERNAL_TP_QUICK_READY_BY_DATASET",
    "FREEHGC_STANDARD_ATTEMPTED",
    "FREEHGC_SCORE_TP_EXECUTED",
    "HESF_RCS_AUTO_EXECUTED_BY_DATASET",
    "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE",
    "HESF_RCS_REP_TASK_RESULTS_READY",
    "STAGE_REPORT_SMOKE_READY",
    "STAGE_REPORT_QUICK_READY",
    "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE",
    "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS",
)


def gate21_16_main_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in GATE21_16_MAIN_FIELDS}
    out["dataset"] = normalize_dataset(out.get("dataset"))
    for field in (
        "schema_compatible",
        "target_preserving",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "training_executed",
        "eligible_for_main_table",
        "success",
    ):
        out[field] = bool_value(out.get(field))
    return out


def gate21_16_success_row(**kwargs: Any) -> dict[str, Any]:
    base = {
        "schema_compatible": True,
        "target_preserving": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
        "training_executed": True,
        "eligible_for_main_table": True,
        "success": True,
    }
    base.update(kwargs)
    return gate21_16_main_row(base)


def gate21_16_pending_row(**kwargs: Any) -> dict[str, Any]:
    base = {
        "schema_compatible": True,
        "target_preserving": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
        "training_executed": False,
        "eligible_for_main_table": True,
        "success": False,
        "failure_type": "implemented_pending_official_training",
        "failure_reason": "Local Gate21.16 implementation/export path was added; official SeHGNN task training remains pending.",
    }
    base.update(kwargs)
    return gate21_16_main_row(base)


def gate21_16_row_ready(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success"))
        and bool_value(row.get("training_executed"))
        and bool_value(row.get("official_hgb_exported"))
        and bool_value(row.get("official_sehgnn_unmodified"))
        and bool_value(row.get("schema_compatible"))
        and bool_value(row.get("target_preserving"))
        and finite_metric(row.get("test_micro_f1_mean"))
        and finite_metric(row.get("test_macro_f1_mean"))
    )


def hesf_auto_name(budget: float) -> str:
    return f"HeSF-RCS-auto structural{int(round(float(budget) * 100)):02d}"


def validation_proxy_from_cost(row: Mapping[str, Any]) -> float:
    micro = float_value(row.get("validation_micro_f1_mean"))
    if micro is not None:
        return micro
    structural = float_value(row.get("actual_structural_storage_ratio")) or 1.0
    support_edge = float_value(row.get("support_edge_ratio")) or structural
    return round(1.0 - 0.35 * structural - 0.05 * support_edge, 9)
