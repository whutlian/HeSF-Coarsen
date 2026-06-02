from __future__ import annotations

from typing import Any, Iterable

from hesf_coarsen.eval.official.stage_report_table import gate21_17_main_row


def build_condensation_score_tp_local_rows(*, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in ("HGCond-score-TP-local", "GCond-score-TP-local"):
            rows.append(
                gate21_17_main_row(
                    {
                    "dataset": dataset,
                    "method": method,
                    "method_family": "condensation_score_tp_proxy",
                    "requested_budget_type": "support_node_ratio",
                    "requested_budget": 0.50,
                    "support_node_ratio": 0.50,
                    "source_path": "local:condensation_score_tp_local",
                    "repo_url": _repo_url(method),
                    "schema_compatible": True,
                    "target_preserving": True,
                    "official_hgb_exported": True,
                    "official_sehgnn_unmodified": True,
                    "training_executed": False,
                    "eligible_for_main_table": True,
                    "success": False,
                    "failure_type": "implemented_pending_official_training",
                    "failure_reason": (
                        f"{method} proxy scoring is defined from paper-style representativeness, diversity, "
                        "feature moment matching, and relation coverage; Gate21.17 must send it through "
                        "the official SeHGNN TP training queue."
                    ),
                    }
                )
            )
    return rows


def _repo_url(method: str) -> str:
    if method.startswith("HGCond"):
        return "https://github.com/jianjianGJ/hgcond"
    return "https://github.com/ChandlerBang/GCond"
