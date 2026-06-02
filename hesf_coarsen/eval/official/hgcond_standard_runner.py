from __future__ import annotations

from typing import Any, Iterable, Mapping


def build_hgcond_standard_rows(*, datasets: Iterable[str], repo_audit_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    repo = next((dict(row) for row in repo_audit_rows if str(row.get("baseline_name")) == "HGCond"), {})
    return [
        {
            "dataset": str(dataset).upper(),
            "method": "HGCond-standard",
            "method_family": "standard_condensation",
            "protocol": "hgcond_standard_condensation",
            "repo_url": repo.get("repo_url", "https://github.com/jianjianGJ/hgcond"),
            "clone_success": repo.get("clone_success", False),
            "success": False,
            "training_executed": False,
            "eligible_for_main_table": False,
            "failure_type": "standard_condensation_not_official_tp",
            "failure_reason": "HGCond-standard uses a condensation protocol separate from official-unmodified SeHGNN TP export; no local comparable standard run is ready.",
        }
        for dataset in datasets
    ]
