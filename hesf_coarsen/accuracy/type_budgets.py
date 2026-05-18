from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph


def compute_type_budget_report(
    original: HeteroGraph,
    coarse: HeteroGraph,
    *,
    target_node_type: int,
) -> dict[str, object]:
    per_type: dict[str, dict[str, float | int | bool]] = {}
    type_ids = sorted(set(int(v) for v in np.unique(original.node_type)).union(int(v) for v in np.unique(coarse.node_type)))
    for type_id in type_ids:
        original_count = int(np.sum(original.node_type == type_id))
        coarse_count = int(np.sum(coarse.node_type == type_id))
        per_type[str(type_id)] = {
            "original_nodes": original_count,
            "coarse_nodes": coarse_count,
            "ratio": float(coarse_count / max(original_count, 1)),
            "is_target": bool(type_id == int(target_node_type)),
        }
    return {
        "target_type": int(target_node_type),
        "original_nodes": int(original.num_nodes),
        "coarse_nodes": int(coarse.num_nodes),
        "global_ratio": float(coarse.num_nodes / max(original.num_nodes, 1)),
        "per_type": per_type,
    }
