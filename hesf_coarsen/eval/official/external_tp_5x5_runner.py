from __future__ import annotations

from typing import Any

from hesf_coarsen.eval.official.gate21_9_decision import REQUIRED_EXTERNAL_TP_5X5


EXTERNAL_TP_BUDGETS = (
    ("support_node_ratio", 0.30),
    ("support_node_ratio", 0.50),
    ("structural_storage_ratio", 0.12),
    ("structural_storage_ratio", 0.16),
    ("structural_storage_ratio", 0.20),
    ("structural_storage_ratio", 0.30),
)


def build_external_tp_5x5_grid(graph_seeds: list[int], training_seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "protocol": "schema_preserving_tp",
            "budget_type": budget_type,
            "requested_budget": budget,
            "graph_seed": graph_seed,
            "training_seed": training_seed,
        }
        for method in REQUIRED_EXTERNAL_TP_5X5
        for budget_type, budget in EXTERNAL_TP_BUDGETS
        for graph_seed in graph_seeds
        for training_seed in training_seeds
    ]
