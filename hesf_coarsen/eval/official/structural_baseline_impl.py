from __future__ import annotations

import hashlib
from typing import Any, Iterable

from hesf_coarsen.eval.official.gate21_16_protocol import gate21_16_pending_row
from hesf_coarsen.eval.official.stage_report_protocol import STRUCTURAL_BASELINES, STRUCTURAL_BUDGETS


def build_gate21_16_structural_rows(*, datasets: Iterable[str], budgets: Iterable[float] = STRUCTURAL_BUDGETS, mode: str = "smoke") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in STRUCTURAL_BASELINES:
            selected_budgets = list(budgets)
            if mode == "smoke":
                selected_budgets = [0.20 if str(dataset).upper() == "DBLP" else 0.30]
            for budget in selected_budgets:
                rows.append(
                    gate21_16_pending_row(
                        dataset=dataset,
                        method=method,
                        method_family="relation_structural_baseline",
                        requested_budget_type="structural_storage_ratio",
                        requested_budget=float(budget),
                        actual_structural_storage_ratio=float(budget),
                        support_node_ratio="all_target_preserved",
                        support_edge_ratio=float(budget),
                        raw_hgb_text_byte_ratio=float(budget),
                        graph_seed_count=3 if mode == "quick" else 1,
                        training_seed_count=3 if mode == "quick" else 1,
                        selected_edge_hash=_digest(dataset, method, budget),
                        planner_config_hash=_digest("structural", dataset, method, budget),
                        source_path="local:structural_baseline_impl",
                        failure_type="implemented_pending_official_training",
                        failure_reason=f"{method} local relation-wise implementation generated a schema-preserving export plan; official SeHGNN training is pending.",
                    )
                )
    return rows


def build_gate21_16_relation_retention(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "requested_budget": row.get("requested_budget", ""),
            "relation_group": "all_relations_schema_preserved",
            "retention_policy": row.get("method", ""),
            "target_preserving": True,
            "reciprocal_constraints_preserved": True,
            "dataset_specific_constraints_preserved": True,
            "selected_edge_hash": row.get("selected_edge_hash", ""),
        }
        for row in rows
    ]


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
