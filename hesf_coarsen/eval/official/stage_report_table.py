from __future__ import annotations

from typing import Any, Mapping

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, finite_metric, normalize_dataset


GATE21_17_MAIN_FIELDS = (
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
    "stdout_path",
    "stderr_path",
)


def gate21_17_main_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in GATE21_17_MAIN_FIELDS}
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
    for optional in ("export_dir", "selection_source", "rep_selection_confidence", "uses_test_for_selection", "eligible_for_decision", "source_method"):
        if optional in row:
            out[optional] = row.get(optional, "")
    if bool_value(out.get("eligible_for_main_table")) and not bool_value(out.get("success")):
        failure_type = str(out.get("failure_type", "")).strip()
        if failure_type in {"", "pending", "not_executed", "missing_task_metric"}:
            out["failure_type"] = "gate21_17_unresolved_execution_state"
            out["failure_reason"] = out.get("failure_reason") or "Gate21.17 requires a concrete export/schema/runtime failure or task metrics."
    return out


def gate21_17_success_row(**kwargs: Any) -> dict[str, Any]:
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
    return gate21_17_main_row(base)


def gate21_17_failure_row(**kwargs: Any) -> dict[str, Any]:
    base = {
        "schema_compatible": True,
        "target_preserving": True,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": True,
        "training_executed": False,
        "eligible_for_main_table": True,
        "success": False,
        "failure_type": "export_schema_failure",
    }
    base.update(kwargs)
    return gate21_17_main_row(base)


def gate21_17_row_ready(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("eligible_for_main_table", True))
        and bool_value(row.get("success"))
        and bool_value(row.get("training_executed"))
        and bool_value(row.get("schema_compatible"))
        and bool_value(row.get("target_preserving"))
        and bool_value(row.get("official_hgb_exported"))
        and bool_value(row.get("official_sehgnn_unmodified"))
        and finite_metric(row.get("test_micro_f1_mean"))
        and finite_metric(row.get("test_macro_f1_mean"))
    )
