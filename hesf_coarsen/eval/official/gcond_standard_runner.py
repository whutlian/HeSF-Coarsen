from __future__ import annotations

from typing import Any, Iterable, Mapping


def build_gcond_standard_rows(*, datasets: Iterable[str], repo_audit_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    audit = {str(row.get("baseline_name")): dict(row) for row in repo_audit_rows}
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method, baseline in (("GCond-standard", "GCond"), ("GCondenser-standard", "GCondenser")):
            repo = audit.get(baseline, {})
            rows.append(
                {
                    "dataset": str(dataset).upper(),
                    "method": method,
                    "method_family": "standard_condensation",
                    "protocol": "standard_graph_condensation",
                    "repo_url": repo.get("repo_url", ""),
                    "clone_success": repo.get("clone_success", False),
                    "success": False,
                    "training_executed": False,
                    "eligible_for_main_table": False,
                    "failure_type": "standard_condensation_not_official_tp",
                    "failure_reason": f"{method} is not a schema-preserving target-preserving official SeHGNN row and no local standard task run is ready.",
                }
            )
    return rows
