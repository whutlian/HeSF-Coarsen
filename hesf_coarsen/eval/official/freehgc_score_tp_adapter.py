from __future__ import annotations

from typing import Any, Iterable, Mapping


FREEHGC_SCORE_FAILURE_TYPE = "edge_provenance_missing"


def build_freehgc_score_tp_rows(
    *,
    datasets: Iterable[str],
    support_node_budgets: Iterable[float],
    structural_budgets: Iterable[float],
    repo_audit_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    repo = next((dict(row) for row in repo_audit_rows if str(row.get("baseline_name")) == "FreeHGC"), {})
    repo_url = str(repo.get("repo_url", "https://github.com/GooLiang/FreeHGC"))
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for support in support_node_budgets:
            rows.append(_failure_row(dataset=dataset, budget_type="support_node_ratio", budget=support, repo_url=repo_url))
        for budget in structural_budgets:
            rows.append(_failure_row(dataset=dataset, budget_type="structural_storage_ratio", budget=budget, repo_url=repo_url))
    return rows


def _failure_row(*, dataset: str, budget_type: str, budget: float, repo_url: str) -> dict[str, Any]:
    return {
        "dataset": str(dataset).upper(),
        "method": "FreeHGC-score-TP",
        "baseline_name": "FreeHGC-score-TP",
        "method_family": "external_tp_baseline",
        "protocol": "schema_preserving_target_preserving_official_sehgnn",
        "repo_url": repo_url,
        "requested_budget_type": budget_type,
        "requested_budget": float(budget),
        "support_node_ratio": float(budget) if budget_type == "support_node_ratio" else "",
        "success": False,
        "training_executed": False,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": True,
        "schema_compatible": True,
        "target_preserving": True,
        "budget_infeasible": False,
        "failure_type": FREEHGC_SCORE_FAILURE_TYPE,
        "failure_reason": "FreeHGC scoring artifacts do not expose original official HGB edge provenance needed to construct TP link.dat endpoints without a loader adapter.",
    }
