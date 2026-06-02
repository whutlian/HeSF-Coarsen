from __future__ import annotations

from typing import Any, Iterable

from hesf_coarsen.eval.official.stage_report_protocol import STRUCTURAL_BASELINES, STRUCTURAL_BUDGETS


def build_relation_structural_baseline_rows(
    *,
    datasets: Iterable[str],
    budgets: Iterable[float] = STRUCTURAL_BUDGETS,
    mode: str = "quick",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in STRUCTURAL_BASELINES:
            for budget in budgets:
                rows.append(
                    {
                        "dataset": str(dataset).upper(),
                        "method": method,
                        "method_family": "relation_structural_baseline",
                        "requested_budget_type": "structural_storage_ratio",
                        "requested_budget": float(budget),
                        "success": False,
                        "training_executed": False,
                        "schema_compatible": True,
                        "target_preserving": True,
                        "official_hgb_exported": False,
                        "official_sehgnn_unmodified": True,
                        "budget_infeasible": False,
                        "failure_type": "not_executed",
                        "failure_reason": f"{method} at structural budget {budget:.2f} is planned for Gate21.15 {mode}, but no official SeHGNN task metric exists locally.",
                    }
                )
    return rows
