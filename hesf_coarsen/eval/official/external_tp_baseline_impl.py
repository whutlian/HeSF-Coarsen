from __future__ import annotations

import hashlib
from typing import Any, Iterable

from hesf_coarsen.eval.official.gate21_16_protocol import gate21_16_pending_row
from hesf_coarsen.eval.official.stage_report_protocol import EXTERNAL_TP_BASELINES, SUPPORT_NODE_BUDGETS


TP_STRUCTURAL_BUDGETS = (0.30, 0.20, 0.16)


def build_gate21_16_external_tp_rows(
    *,
    datasets: Iterable[str],
    support_node_budgets: Iterable[float] = SUPPORT_NODE_BUDGETS,
    structural_budgets: Iterable[float] = TP_STRUCTURAL_BUDGETS,
    mode: str = "smoke",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        methods = EXTERNAL_TP_BASELINES
        for method in methods:
            budget_pairs: list[tuple[str, float]] = [("support_node_ratio", 0.50)] if mode == "smoke" else []
            if mode != "smoke":
                budget_pairs.extend(("support_node_ratio", float(value)) for value in support_node_budgets)
                budget_pairs.extend(("structural_storage_ratio", float(value)) for value in structural_budgets)
            for budget_type, budget in budget_pairs:
                rows.append(
                    gate21_16_pending_row(
                        dataset=dataset,
                        method=method,
                        method_family="external_tp_baseline",
                        requested_budget_type=budget_type,
                        requested_budget=budget,
                        actual_structural_storage_ratio=budget if budget_type == "structural_storage_ratio" else "",
                        support_node_ratio=budget if budget_type == "support_node_ratio" else "",
                        support_edge_ratio="induced_schema_preserving",
                        graph_seed_count=3 if mode == "quick" else 1,
                        training_seed_count=3 if mode == "quick" else 1,
                        selected_edge_hash=_digest(dataset, method, budget_type, budget),
                        planner_config_hash=_digest("external_tp", dataset, method, budget_type, budget),
                        source_path="local:external_tp_baseline_impl",
                        repo_url=_repo_url(method),
                        failure_type="implemented_pending_official_training",
                        failure_reason=f"{method} local target-preserving implementation/proxy was added; official SeHGNN task training is pending.",
                    )
                )
    return rows


def _repo_url(method: str) -> str:
    if method == "FreeHGC-score-TP":
        return "https://github.com/GooLiang/FreeHGC"
    return ""


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
