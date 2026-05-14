from __future__ import annotations

from dataclasses import dataclass

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


def run_greedy_cluster_matching(
    graph: HeteroGraph,
    scored_pairs: np.ndarray,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> Assignment:
    coarsen_cfg = config.get("coarsening", {})
    same_type_only = bool(coarsen_cfg.get("same_type_only", True))
    same_partition_only = bool(coarsen_cfg.get("same_partition_only", True))
    max_cluster_size = max(2, int(coarsen_cfg.get("max_cluster_size", 4)))
    max_matched_pairs = coarsen_cfg.get("max_matched_pairs")
    max_merges = None if max_matched_pairs is None else max(0, int(max_matched_pairs))
    if max_merges == 0:
        return _singleton_assignment(graph)

    parent = np.arange(graph.num_nodes, dtype=np.int64)
    size = np.ones(graph.num_nodes, dtype=np.int32)

    def find(node: int) -> int:
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[int(node)] != int(node):
            next_node = int(parent[int(node)])
            parent[int(node)] = root
            node = next_node
        return root

    if scored_pairs.size:
        pairs = np.asarray(scored_pairs)
        left = pairs[:, 0].astype(np.int64, copy=False)
        right = pairs[:, 1].astype(np.int64, copy=False)
        costs = pairs[:, 2].astype(np.float64, copy=False)
        order = np.lexsort((right, left, costs))
    else:
        pairs = np.empty((0, 3), dtype=np.float64)
        order = np.empty(0, dtype=np.int64)

    merges = 0
    for row_index in order:
        if max_merges is not None and merges >= max_merges:
            break
        i = int(pairs[row_index, 0])
        j = int(pairs[row_index, 1])
        if i == j:
            continue
        if same_type_only and graph.node_type[i] != graph.node_type[j]:
            continue
        if (
            same_partition_only
            and partition_id is not None
            and partition_id[i] != partition_id[j]
        ):
            continue
        root_i = find(i)
        root_j = find(j)
        if root_i == root_j:
            continue
        if same_type_only and graph.node_type[root_i] != graph.node_type[root_j]:
            continue
        if int(size[root_i]) + int(size[root_j]) > max_cluster_size:
            continue
        if size[root_i] < size[root_j] or (size[root_i] == size[root_j] and root_j < root_i):
            root_i, root_j = root_j, root_i
        parent[root_j] = root_i
        size[root_i] += size[root_j]
        merges += 1

    root_to_super: dict[int, int] = {}
    assignment = np.empty(graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    for node in range(graph.num_nodes):
        root = find(node)
        super_id = root_to_super.get(root)
        if super_id is None:
            super_id = len(super_types)
            root_to_super[root] = super_id
            super_types.append(int(graph.node_type[node]))
        assignment[node] = super_id

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


@dataclass
class MutualBestState:
    best_cost: np.ndarray
    best_neighbor: np.ndarray
    best_source: np.ndarray | None = None


def initialize_mutual_best_state(graph: HeteroGraph) -> MutualBestState:
    missing_neighbor = np.iinfo(np.int64).max
    return MutualBestState(
        best_cost=np.full(graph.num_nodes, np.inf, dtype=np.float64),
        best_neighbor=np.full(graph.num_nodes, missing_neighbor, dtype=np.int64),
        best_source=None,
    )


def _update_directed_best(
    state: MutualBestState,
    nodes: np.ndarray,
    neighbors: np.ndarray,
    costs: np.ndarray,
) -> None:
    if len(nodes) == 0:
        return
    unique_nodes, inverse = np.unique(nodes, return_inverse=True)
    block_best_cost = np.full(len(unique_nodes), np.inf, dtype=np.float64)
    np.minimum.at(block_best_cost, inverse, costs)

    improved = block_best_cost < state.best_cost[unique_nodes]
    if np.any(improved):
        improved_nodes = unique_nodes[improved]
        state.best_cost[improved_nodes] = block_best_cost[improved]
        state.best_neighbor[improved_nodes] = np.iinfo(np.int64).max

    eligible = costs == state.best_cost[nodes]
    if np.any(eligible):
        np.minimum.at(state.best_neighbor, nodes[eligible], neighbors[eligible])


def mutual_best_update_block(
    graph: HeteroGraph,
    state: MutualBestState,
    scored_pairs: np.ndarray,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> None:
    if scored_pairs.size == 0:
        return

    coarsen_cfg = config.get("coarsening", {})
    same_type_only = bool(coarsen_cfg.get("same_type_only", True))
    same_partition_only = bool(coarsen_cfg.get("same_partition_only", True))
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
        return

    left = left[valid]
    right = right[valid]
    costs = costs[valid]
    _update_directed_best(state, left, right, costs)
    _update_directed_best(state, right, left, costs)


def selected_pair_sources(
    assignment: Assignment,
    source_lookup,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for supernode in range(assignment.num_supernodes):
        nodes = np.flatnonzero(assignment.assignment == supernode)
        if len(nodes) != 2:
            continue
        source = source_lookup(int(nodes[0]), int(nodes[1]))
        name = "unknown" if source is None else str(source)
        counts[name] = counts.get(name, 0) + 1
    return counts


def finalize_mutual_best(
    graph: HeteroGraph,
    state: MutualBestState,
    config: dict,
) -> Assignment:
    coarsen_cfg = config.get("coarsening", {})
    max_matched_pairs = coarsen_cfg.get("max_matched_pairs")
    if max_matched_pairs is not None:
        max_matched_pairs = max(0, int(max_matched_pairs))
        if max_matched_pairs == 0:
            return _singleton_assignment(graph)

    missing_neighbor = np.iinfo(np.int64).max
    has_neighbor = state.best_neighbor != missing_neighbor
    candidate_nodes = np.flatnonzero(has_neighbor).astype(np.int64)
    if len(candidate_nodes) == 0:
        return _singleton_assignment(graph)
    candidate_partners = state.best_neighbor[candidate_nodes]
    mutual_mask = (candidate_nodes < candidate_partners) & (
        state.best_neighbor[candidate_partners] == candidate_nodes
    )
    mutual_left = candidate_nodes[mutual_mask]
    if len(mutual_left) == 0:
        return _singleton_assignment(graph)
    mutual_right = state.best_neighbor[mutual_left]
    mutual_cost = state.best_cost[mutual_left]

    if max_matched_pairs is not None and len(mutual_left) > max_matched_pairs:
        order = np.lexsort((mutual_right, mutual_left, mutual_cost))
        keep = order[:max_matched_pairs]
        mutual_left = mutual_left[keep]
        mutual_right = mutual_right[keep]

    return _assignment_from_pair_arrays(graph, mutual_left, mutual_right)


def run_mutual_best_matching(
    graph: HeteroGraph,
    scored_pairs: np.ndarray,
    config: dict,
    partition_id: np.ndarray | None = None,
) -> Assignment:
    state = initialize_mutual_best_state(graph)
    mutual_best_update_block(graph, state, scored_pairs, config, partition_id=partition_id)
    return finalize_mutual_best(graph, state, config)


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
    if method in {"greedy_cluster", "cluster_greedy", "block_local_approximate_greedy"}:
        return run_greedy_cluster_matching(graph, scored_pairs, config, partition_id=partition_id)
    raise ValueError(f"unsupported matching_method: {method}")
