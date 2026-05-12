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
        pairs = np.asarray(scored_pairs)
        left = pairs[:, 0].astype(np.int64, copy=False)
        right = pairs[:, 1].astype(np.int64, copy=False)
        costs = pairs[:, 2].astype(np.float64, copy=False)
        order = np.lexsort((right, left, costs))
    else:
        pairs = np.empty((0, 3), dtype=np.float64)
        order = np.empty(0, dtype=np.int64)

    matched = 0
    for row_index in order:
        if max_matched_pairs is not None and matched >= max_matched_pairs:
            break
        i = int(pairs[row_index, 0])
        j = int(pairs[row_index, 1])
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


def _singleton_assignment(graph: HeteroGraph) -> Assignment:
    assignment = np.arange(graph.num_nodes, dtype=np.int64)
    return Assignment(assignment=assignment, supernode_type=graph.node_type.astype(np.int32, copy=True))


def _assignment_from_pair_arrays(
    graph: HeteroGraph,
    left: np.ndarray,
    right: np.ndarray,
) -> Assignment:
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types: list[int] = []
    for raw_i, raw_j in zip(left, right):
        i = int(raw_i)
        j = int(raw_j)
        if assignment[i] >= 0 or assignment[j] >= 0:
            continue
        super_id = len(super_types)
        assignment[i] = super_id
        assignment[j] = super_id
        super_types.append(int(graph.node_type[i]))

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


def run_mutual_best_matching(
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
        max_matched_pairs = max(0, int(max_matched_pairs))
        if max_matched_pairs == 0:
            return _singleton_assignment(graph)

    if scored_pairs.size == 0:
        return _singleton_assignment(graph)

    pairs = np.asarray(scored_pairs)
    left = pairs[:, 0].astype(np.int64, copy=False)
    right = pairs[:, 1].astype(np.int64, copy=False)
    costs = pairs[:, 2].astype(np.float64, copy=False)

    valid = (left != right) & (left >= 0) & (right >= 0) & (left < graph.num_nodes) & (right < graph.num_nodes)
    if same_type_only:
        typed = np.zeros_like(valid, dtype=bool)
        typed[valid] = graph.node_type[left[valid]] == graph.node_type[right[valid]]
        valid &= typed
    if same_partition_only and partition_id is not None:
        same_partition = np.zeros_like(valid, dtype=bool)
        same_partition[valid] = partition_id[left[valid]] == partition_id[right[valid]]
        valid &= same_partition
    if not np.any(valid):
        return _singleton_assignment(graph)

    left = left[valid]
    right = right[valid]
    costs = costs[valid]

    best_cost = np.full(graph.num_nodes, np.inf, dtype=np.float64)
    np.minimum.at(best_cost, left, costs)
    np.minimum.at(best_cost, right, costs)

    missing_neighbor = np.iinfo(np.int64).max
    best_neighbor = np.full(graph.num_nodes, missing_neighbor, dtype=np.int64)
    left_best = costs == best_cost[left]
    right_best = costs == best_cost[right]
    if np.any(left_best):
        np.minimum.at(best_neighbor, left[left_best], right[left_best])
    if np.any(right_best):
        np.minimum.at(best_neighbor, right[right_best], left[right_best])

    has_neighbor = best_neighbor != missing_neighbor
    candidate_nodes = np.flatnonzero(has_neighbor).astype(np.int64)
    candidate_partners = best_neighbor[candidate_nodes]
    mutual_mask = (candidate_nodes < candidate_partners) & (
        best_neighbor[candidate_partners] == candidate_nodes
    )
    mutual_left = candidate_nodes[mutual_mask]
    if len(mutual_left) == 0:
        return _singleton_assignment(graph)
    mutual_right = best_neighbor[mutual_left]
    mutual_cost = best_cost[mutual_left]

    if max_matched_pairs is not None and len(mutual_left) > max_matched_pairs:
        order = np.lexsort((mutual_right, mutual_left, mutual_cost))
        keep = order[:max_matched_pairs]
        mutual_left = mutual_left[keep]
        mutual_right = mutual_right[keep]

    return _assignment_from_pair_arrays(graph, mutual_left, mutual_right)


def run_matching(
    graph: HeteroGraph,
    scored_pairs: np.ndarray,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> Assignment:
    method = str(config.get("coarsening", {}).get("matching_method", "mutual_best")).lower()
    if method in {"mutual_best", "mutual-best"}:
        return run_mutual_best_matching(graph, scored_pairs, config, partition_id=partition_id)
    if method in {"greedy", "global_greedy", "sorted_greedy"}:
        return run_greedy_matching(graph, scored_pairs, config, partition_id=partition_id)
    raise ValueError(f"unsupported matching_method: {method}")
