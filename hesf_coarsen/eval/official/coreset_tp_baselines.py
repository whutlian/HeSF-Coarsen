from __future__ import annotations

from typing import Any

import numpy as np


def select_random_support_nodes(nodes: list[int], *, ratio: float, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    count = max(0, min(len(nodes), int(round(len(nodes) * float(ratio)))))
    if count == 0:
        return []
    values = np.array([int(node) for node in nodes], dtype=np.int64)
    selected = rng.choice(values, size=count, replace=False)
    return sorted(int(node) for node in selected.tolist())


def coreset_plan_row(method: str, *, support_node_ratio: float, selection_signal: str) -> dict[str, Any]:
    return {
        "baseline_name": str(method),
        "method_family": "external_tp_baseline",
        "budget_type": "support_node_ratio",
        "budget_value": float(support_node_ratio),
        "selection_signal": str(selection_signal),
        "keeps_all_target_nodes": True,
        "schema_compatible": True,
        "uses_synthetic_nodes": False,
    }
