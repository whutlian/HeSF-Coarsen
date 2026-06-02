from __future__ import annotations

from typing import Any, Iterable

from hesf_coarsen.eval.official.gate21_16_protocol import gate21_16_pending_row


def build_condensation_score_tp_rows(*, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in ("HGCond-score-TP", "GCond-score-TP"):
            rows.append(
                gate21_16_pending_row(
                    dataset=dataset,
                    method=method,
                    method_family="condensation_score_tp_proxy",
                    requested_budget_type="support_node_ratio",
                    requested_budget=0.50,
                    support_node_ratio=0.50,
                    source_path="local:hgcond_score_tp_local",
                    failure_type="implemented_pending_official_training",
                    failure_reason=f"{method} local score/proxy baseline was implemented; official SeHGNN TP training is pending.",
                )
            )
    return rows
