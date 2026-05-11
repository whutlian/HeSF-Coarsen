from __future__ import annotations

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph


def run_greedy_matching(
    graph: HeteroGraph,
    scored_pairs: np.ndarray,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> Assignment:
    coarsen_cfg = config.get("coarsening", {})
    same_type_only = bool(coarsen_cfg.get("same_type_only", True))
    same_partition_only = bool(coarsen_cfg.get("same_partition_only", True))
    max_matched_pairs = coarsen_cfg.get("max_matched_pairs")
    if max_matched_pairs is not None:
        max_matched_pairs = int(max_matched_pairs)

    used = np.zeros(graph.num_nodes, dtype=bool)
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types: list[int] = []

    if scored_pairs.size:
        order = sorted(
            scored_pairs.tolist(),
            key=lambda row: (float(row[2]), int(row[0]), int(row[1])),
        )
    else:
        order = []

    matched = 0
    for raw_i, raw_j, _raw_cost in order:
        if max_matched_pairs is not None and matched >= max_matched_pairs:
            break
        i = int(raw_i)
        j = int(raw_j)
        if i == j or used[i] or used[j]:
            continue
        if same_type_only and graph.node_type[i] != graph.node_type[j]:
            continue
        if (
            same_partition_only
            and partition_id is not None
            and partition_id[i] != partition_id[j]
        ):
            continue
        super_id = len(super_types)
        assignment[i] = super_id
        assignment[j] = super_id
        used[i] = True
        used[j] = True
        super_types.append(int(graph.node_type[i]))
        matched += 1

    for node in range(graph.num_nodes):
        if assignment[node] >= 0:
            continue
        super_id = len(super_types)
        assignment[node] = super_id
        super_types.append(int(graph.node_type[node]))

    return Assignment(
        assignment=assignment,
        supernode_type=np.asarray(super_types, dtype=np.int32),
    )
