from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


DATASETS = ("DBLP", "ACM", "IMDB")
STRUCTURAL_BUDGETS = (0.50, 0.30, 0.20, 0.16, 0.12)
SUPPORT_NODE_BUDGETS = (0.30, 0.50)

FULL_METHODS = ("Full-native-SeHGNN", "Export-full-SeHGNN")
INTERNAL_BASELINES = ("H6-node30", "flatten-node30", "TypedHash-node30")
STRUCTURAL_BASELINES = ("Random-edge-relwise", "Degree-edge-relwise", "Proportional-relation-budget")
EXTERNAL_TP_BASELINES = (
    "Random-HG-TP",
    "Herding-HG-TP",
    "KCenter-HG-TP",
    "GraphSparsify-TP",
    "Coarsening-HG-TP",
    "FreeHGC-score-TP",
)

MAIN_TABLE_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "requested_budget_type",
    "requested_budget",
    "actual_structural_storage_ratio",
    "support_node_ratio",
    "support_edge_ratio",
    "total_node_ratio",
    "total_edge_ratio",
    "raw_hgb_text_byte_ratio",
    "link_dat_bytes",
    "node_dat_bytes",
    "export_total_bytes",
    "native_full_total_bytes",
    "graph_seed_count",
    "training_seed_count",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "recovery_vs_native_full_micro",
    "recovery_vs_native_full_macro",
    "full_minus_micro",
    "full_minus_macro",
    "schema_compatible",
    "target_preserving",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "training_executed",
    "eligible_for_main_table",
    "failure_type",
    "failure_reason",
)

REP_SELECTION_FIELDS = (
    "dataset",
    "candidate_method",
    "candidate_requested_budget",
    "candidate_actual_structural_ratio",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "selected_as_rep",
    "selection_rank",
    "selection_reason",
    "uses_test_for_selection",
    "selected_edge_hash",
    "planner_config_hash",
)

REQUIRED_DECISION_FLAGS = (
    "FULL_NATIVE_READY_BY_DATASET",
    "EXPORT_FULL_FIDELITY_PASS_BY_DATASET",
    "MAIN_TABLE_HAS_DBLP_ACM_IMDB",
    "HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE",
    "HESF_RCS_REP_TASK_RESULTS_READY",
    "STRUCTURAL_BASELINES_READY",
    "EXTERNAL_TP_BASELINES_CLONED_OR_IMPLEMENTED",
    "EXTERNAL_TP_TASK_RESULTS_READY",
    "FREEHGC_STANDARD_READY_OR_HARD_FAILURE_RECORDED",
    "FREEHGC_SCORE_TP_READY",
    "BUDGET_MATCH_AUDIT_PASS",
    "NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE",
    "NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS",
    "STAGE_REPORT_TABLE_READY",
)


def normalize_dataset(value: object) -> str:
    return str(value or "").strip().upper()


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "success"}


def float_value(value: object) -> float | None:
    if value in {"", None, "NaN", "nan", "None"}:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def finite_metric(value: object) -> bool:
    return float_value(value) is not None


def main_row_success_ready(row: Mapping[str, Any]) -> bool:
    return bool(
        bool_value(row.get("success", row.get("training_executed")))
        and bool_value(row.get("schema_compatible"))
        and bool_value(row.get("target_preserving"))
        and bool_value(row.get("official_hgb_exported"))
        and bool_value(row.get("official_sehgnn_unmodified"))
        and bool_value(row.get("training_executed"))
        and bool_value(row.get("eligible_for_main_table", True))
        and finite_metric(row.get("test_micro_f1_mean"))
        and finite_metric(row.get("test_macro_f1_mean"))
    )


def normalize_main_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in MAIN_TABLE_FIELDS}
    out["dataset"] = normalize_dataset(out.get("dataset"))
    out["method"] = str(out.get("method", ""))
    out["method_family"] = str(out.get("method_family", ""))
    for field in (
        "schema_compatible",
        "target_preserving",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "training_executed",
        "eligible_for_main_table",
    ):
        out[field] = bool_value(out.get(field))
    out["success"] = bool_value(row.get("success", main_row_success_ready(out)))
    for extra in (
        "source_gate",
        "source_path",
        "expected_success_count",
        "selected_edge_hash",
        "planner_config_hash",
        "diagnostic_only",
        "repo_url",
        "traceback_path",
    ):
        if extra in row:
            out[extra] = row.get(extra, "")
    return out


def success_main_row(
    *,
    dataset: str,
    method: str,
    method_family: str,
    test_micro_f1_mean: object,
    test_macro_f1_mean: object,
    test_micro_f1_std: object = "",
    test_macro_f1_std: object = "",
    validation_micro_f1_mean: object = "",
    validation_macro_f1_mean: object = "",
    requested_budget_type: object = "",
    requested_budget: object = "",
    actual_structural_storage_ratio: object = "",
    support_node_ratio: object = "",
    support_edge_ratio: object = "",
    raw_hgb_text_byte_ratio: object = "",
    total_node_ratio: object = "",
    total_edge_ratio: object = "",
    graph_seed_count: object = 1,
    training_seed_count: object = 5,
    recovery_vs_native_full_micro: object = "",
    recovery_vs_native_full_macro: object = "",
    full_minus_micro: object = "",
    full_minus_macro: object = "",
    source_gate: str = "",
    source_path: str = "",
    selected_edge_hash: str = "",
    planner_config_hash: str = "",
) -> dict[str, Any]:
    return normalize_main_row(
        {
            "dataset": dataset,
            "method": method,
            "method_family": method_family,
            "requested_budget_type": requested_budget_type,
            "requested_budget": requested_budget,
            "actual_structural_storage_ratio": actual_structural_storage_ratio,
            "support_node_ratio": support_node_ratio,
            "support_edge_ratio": support_edge_ratio,
            "total_node_ratio": total_node_ratio,
            "total_edge_ratio": total_edge_ratio,
            "raw_hgb_text_byte_ratio": raw_hgb_text_byte_ratio,
            "graph_seed_count": graph_seed_count,
            "training_seed_count": training_seed_count,
            "test_micro_f1_mean": test_micro_f1_mean,
            "test_micro_f1_std": test_micro_f1_std,
            "test_macro_f1_mean": test_macro_f1_mean,
            "test_macro_f1_std": test_macro_f1_std,
            "validation_micro_f1_mean": validation_micro_f1_mean,
            "validation_macro_f1_mean": validation_macro_f1_mean,
            "recovery_vs_native_full_micro": recovery_vs_native_full_micro,
            "recovery_vs_native_full_macro": recovery_vs_native_full_macro,
            "full_minus_micro": full_minus_micro,
            "full_minus_macro": full_minus_macro,
            "schema_compatible": True,
            "target_preserving": True,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "training_executed": True,
            "eligible_for_main_table": True,
            "success": True,
            "source_gate": source_gate,
            "source_path": source_path,
            "selected_edge_hash": selected_edge_hash,
            "planner_config_hash": planner_config_hash,
        }
    )


def failure_main_row(
    *,
    dataset: str,
    method: str,
    method_family: str,
    failure_type: str,
    failure_reason: str,
    requested_budget_type: object = "",
    requested_budget: object = "",
    actual_structural_storage_ratio: object = "",
    support_node_ratio: object = "",
    support_edge_ratio: object = "",
    raw_hgb_text_byte_ratio: object = "",
    graph_seed_count: object = "",
    training_seed_count: object = "",
    schema_compatible: bool = True,
    target_preserving: bool = True,
    official_hgb_exported: bool = False,
    official_sehgnn_unmodified: bool = True,
    eligible_for_main_table: bool = True,
    source_gate: str = "",
    repo_url: str = "",
    traceback_path: str = "",
) -> dict[str, Any]:
    return normalize_main_row(
        {
            "dataset": dataset,
            "method": method,
            "method_family": method_family,
            "requested_budget_type": requested_budget_type,
            "requested_budget": requested_budget,
            "actual_structural_storage_ratio": actual_structural_storage_ratio,
            "support_node_ratio": support_node_ratio,
            "support_edge_ratio": support_edge_ratio,
            "raw_hgb_text_byte_ratio": raw_hgb_text_byte_ratio,
            "graph_seed_count": graph_seed_count,
            "training_seed_count": training_seed_count,
            "schema_compatible": schema_compatible,
            "target_preserving": target_preserving,
            "official_hgb_exported": official_hgb_exported,
            "official_sehgnn_unmodified": official_sehgnn_unmodified,
            "training_executed": False,
            "eligible_for_main_table": eligible_for_main_table,
            "success": False,
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "source_gate": source_gate,
            "repo_url": repo_url,
            "traceback_path": traceback_path,
        }
    )


def select_hesf_rcs_representatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = DATASETS,
) -> list[dict[str, Any]]:
    """Select HeSF-RCS-Rep candidates using validation metrics only."""

    source_rows = [dict(row) for row in rows]
    output: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        candidates = [
            row
            for row in source_rows
            if normalize_dataset(row.get("dataset")) == dataset
            and str(row.get("method", "")).startswith("HeSF-RCS-auto")
            and bool_value(row.get("eligible_for_main_table", True))
            and bool_value(row.get("training_executed", False))
            and bool_value(row.get("success", True))
        ]
        eligible = [
            row
            for row in candidates
            if finite_metric(row.get("validation_micro_f1_mean", row.get("validation_micro_f1")))
            and finite_metric(row.get("validation_macro_f1_mean", row.get("validation_macro_f1")))
        ]
        if not eligible:
            output.append(
                {
                    "dataset": dataset,
                    "candidate_method": "",
                    "candidate_requested_budget": "",
                    "candidate_actual_structural_ratio": "",
                    "validation_micro_f1": "",
                    "validation_macro_f1": "",
                    "test_micro_f1": "",
                    "test_macro_f1": "",
                    "selected_as_rep": False,
                    "selection_rank": "",
                    "selection_reason": "validation_metrics_missing",
                    "uses_test_for_selection": False,
                    "selected_edge_hash": "",
                    "planner_config_hash": "",
                    "failure_type": "validation_metrics_missing",
                    "failure_reason": "No eligible HeSF-RCS-auto row has validation_micro_f1 and validation_macro_f1.",
                }
            )
            continue

        ranked = sorted(
            eligible,
            key=lambda row: (
                -(float_value(row.get("validation_micro_f1_mean", row.get("validation_micro_f1"))) or -1.0),
                -(float_value(row.get("validation_macro_f1_mean", row.get("validation_macro_f1"))) or -1.0),
                float_value(row.get("actual_structural_storage_ratio")) or float("inf"),
                float_value(row.get("raw_hgb_text_byte_ratio")) or float("inf"),
                str(row.get("method", "")),
            ),
        )
        selected = ranked[0]
        for rank, row in enumerate(ranked, start=1):
            output.append(
                {
                    "dataset": dataset,
                    "candidate_method": row.get("method", ""),
                    "candidate_requested_budget": row.get("requested_budget", row.get("requested_structural_budget", "")),
                    "candidate_actual_structural_ratio": row.get("actual_structural_storage_ratio", ""),
                    "validation_micro_f1": row.get("validation_micro_f1_mean", row.get("validation_micro_f1", "")),
                    "validation_macro_f1": row.get("validation_macro_f1_mean", row.get("validation_macro_f1", "")),
                    "test_micro_f1": row.get("test_micro_f1_mean", row.get("test_micro_f1", "")),
                    "test_macro_f1": row.get("test_macro_f1_mean", row.get("test_macro_f1", "")),
                    "selected_as_rep": row is selected,
                    "selection_rank": rank,
                    "selection_reason": (
                        "validation_micro_f1 desc; validation_macro_f1 desc; "
                        "actual_structural_storage_ratio asc; raw_hgb_text_byte_ratio asc"
                    ),
                    "uses_test_for_selection": False,
                    "selected_edge_hash": row.get("selected_edge_hash", ""),
                    "planner_config_hash": row.get("planner_config_hash", row.get("selection_config_hash", "")),
                }
            )
    return output


def required_stage_methods() -> tuple[str, ...]:
    hesf_auto = tuple(f"HeSF-RCS-auto structural{int(round(budget * 100)):02d}" for budget in STRUCTURAL_BUDGETS)
    return (*FULL_METHODS, *INTERNAL_BASELINES, *STRUCTURAL_BASELINES, *EXTERNAL_TP_BASELINES, *hesf_auto, "HeSF-RCS-Rep")
