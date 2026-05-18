from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


def target_anchor_budget(
    graph: HeteroGraph,
    *,
    target_node_type: int,
    train_nodes: np.ndarray,
    global_target_ratio: float,
    mode: str = "accuracy_first",
    min_extra_fraction: float = 0.1,
) -> dict[str, int | float | str]:
    target_nodes = nodes_of_type(graph, int(target_node_type))
    train_count = int(len(np.unique(np.asarray(train_nodes, dtype=np.int64))))
    global_budget = max(1, int(round(graph.num_nodes * float(global_target_ratio))))
    if str(mode) == "benchmark_matching":
        target_budget = min(len(target_nodes), max(train_count, int(round(global_budget * 0.5))))
    else:
        target_budget = min(
            len(target_nodes),
            max(train_count, train_count + int(round(len(target_nodes) * float(min_extra_fraction)))),
        )
    return {
        "mode": str(mode),
        "target_node_type": int(target_node_type),
        "target_nodes": int(len(target_nodes)),
        "train_target_nodes": int(train_count),
        "global_target_ratio": float(global_target_ratio),
        "global_node_budget": int(global_budget),
        "target_anchor_budget": int(target_budget),
    }
