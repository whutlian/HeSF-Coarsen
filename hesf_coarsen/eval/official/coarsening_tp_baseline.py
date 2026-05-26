from __future__ import annotations

from typing import Any


def coarsening_tp_plan_row(*, support_node_ratio: float, creates_synthetic_nodes: bool = False) -> dict[str, Any]:
    return {
        "baseline_name": "Coarsening-HG-TP",
        "method_family": "external_tp_baseline",
        "budget_type": "support_node_ratio",
        "budget_value": float(support_node_ratio),
        "keeps_all_target_nodes": True,
        "schema_compatible": True,
        "uses_synthetic_nodes": bool(creates_synthetic_nodes),
        "official_sehgnn_unmodified": not bool(creates_synthetic_nodes),
    }
