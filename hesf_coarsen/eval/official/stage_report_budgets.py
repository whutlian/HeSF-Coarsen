from __future__ import annotations

from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.stage_report_protocol import (
    STRUCTURAL_BUDGETS,
    SUPPORT_NODE_BUDGETS,
    bool_value,
    float_value,
)


def requested_budget_rows(
    *,
    datasets: Iterable[str],
    structural_budgets: Iterable[float] = STRUCTURAL_BUDGETS,
    support_node_budgets: Iterable[float] = SUPPORT_NODE_BUDGETS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for budget in structural_budgets:
            rows.append({"dataset": str(dataset).upper(), "requested_budget_type": "structural_storage_ratio", "requested_budget": float(budget)})
        for budget in support_node_budgets:
            rows.append({"dataset": str(dataset).upper(), "requested_budget_type": "support_node_ratio", "requested_budget": float(budget)})
    return rows


def build_budget_match_audit(rows: Iterable[Mapping[str, Any]], *, tolerance: float = 0.01) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for row in rows:
        requested_type = str(row.get("requested_budget_type", ""))
        requested = float_value(row.get("requested_budget"))
        actual = float_value(row.get("actual_structural_storage_ratio"))
        if requested_type == "structural_storage_ratio" and requested is not None and actual is not None:
            slack = requested - actual
            match_pass = slack >= -float(tolerance)
            failure_reason = "" if match_pass else f"actual structural ratio {actual:.6f} exceeds requested {requested:.6f}"
        elif requested_type == "support_node_ratio" and requested is not None:
            support = float_value(row.get("support_node_ratio"))
            slack = None if support is None else requested - support
            match_pass = support is not None and abs(slack or 0.0) <= float(tolerance)
            failure_reason = "" if match_pass else "support-node budget is missing or outside tolerance"
        else:
            slack = ""
            match_pass = True
            failure_reason = ""
        audit.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "requested_budget_type": requested_type,
                "requested_budget": row.get("requested_budget", ""),
                "actual_structural_storage_ratio": row.get("actual_structural_storage_ratio", ""),
                "support_node_ratio": row.get("support_node_ratio", ""),
                "budget_slack": "" if slack is None else slack,
                "success": bool_value(row.get("success")),
                "budget_match_pass": bool(match_pass),
                "failure_reason": failure_reason,
            }
        )
    return audit


def build_storage_audit(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "requested_budget_type": row.get("requested_budget_type", ""),
            "requested_budget": row.get("requested_budget", ""),
            "actual_structural_storage_ratio": row.get("actual_structural_storage_ratio", ""),
            "support_node_ratio": row.get("support_node_ratio", ""),
            "support_edge_ratio": row.get("support_edge_ratio", ""),
            "total_node_ratio": row.get("total_node_ratio", ""),
            "total_edge_ratio": row.get("total_edge_ratio", ""),
            "raw_hgb_text_byte_ratio": row.get("raw_hgb_text_byte_ratio", ""),
            "link_dat_bytes": row.get("link_dat_bytes", ""),
            "node_dat_bytes": row.get("node_dat_bytes", ""),
            "export_total_bytes": row.get("export_total_bytes", ""),
            "native_full_total_bytes": row.get("native_full_total_bytes", ""),
            "success": bool_value(row.get("success")),
            "failure_type": row.get("failure_type", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        for row in rows
    ]


def build_export_fidelity_audit(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "schema_compatible": bool_value(row.get("schema_compatible")),
            "target_preserving": bool_value(row.get("target_preserving")),
            "official_hgb_exported": bool_value(row.get("official_hgb_exported")),
            "official_sehgnn_unmodified": bool_value(row.get("official_sehgnn_unmodified")),
            "training_executed": bool_value(row.get("training_executed")),
            "eligible_for_main_table": bool_value(row.get("eligible_for_main_table")),
            "export_fidelity_pass": bool_value(row.get("schema_compatible"))
            and bool_value(row.get("target_preserving"))
            and bool_value(row.get("official_hgb_exported"))
            and bool_value(row.get("official_sehgnn_unmodified")),
            "failure_type": row.get("failure_type", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        for row in rows
    ]
