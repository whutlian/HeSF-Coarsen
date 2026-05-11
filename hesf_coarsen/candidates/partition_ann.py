from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.progress import progress_iter


def _partition_groups(
    graph: HeteroGraph,
    partition_id: np.ndarray,
) -> Iterable[np.ndarray]:
    for type_id in sorted(np.unique(graph.node_type)):
        type_nodes = nodes_of_type(graph, int(type_id))
        partitions = partition_id[type_nodes]
        for partition in sorted(np.unique(partitions)):
            group = type_nodes[partitions == partition]
            if len(group) >= 2:
                yield group.astype(np.int64, copy=False)


def generate_partition_ann_candidates(
    graph: HeteroGraph,
    Z: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
) -> dict[str, int]:
    """Generate deterministic projection-window candidates per type partition.

    This is an ANN-style candidate source without external dependencies: each
    same-type, same-partition group is sorted by several seeded random
    projections, and nodes only compare against a small sliding window.
    """

    candidate_cfg = config.get("candidates", {})
    num_projections = int(candidate_cfg.get("ann_num_projections", 4))
    window_size = int(candidate_cfg.get("ann_window_size", 8))
    per_node_budget = int(candidate_cfg.get("ann_budget_K", candidate_cfg.get("total_budget_K", 16)))
    seed = int(config.get("seed", 12345))
    if num_projections <= 0:
        raise ValueError("ann_num_projections must be positive")
    if window_size <= 0:
        raise ValueError("ann_window_size must be positive")
    if per_node_budget <= 0:
        raise ValueError("ann_budget_K must be positive")

    Z = np.asarray(Z, dtype=np.float32)
    proposal_counts = np.zeros(graph.num_nodes, dtype=np.int32)
    pairs_considered = 0
    groups_considered = 0

    groups = list(_partition_groups(graph, partition_id))
    for group_id, group in enumerate(
        progress_iter(
            groups,
            total=len(groups),
            desc="partition ANN groups",
            config=config,
            unit="group",
        )
    ):
        groups_considered += 1
        group_Z = Z[group]
        for projection_id in range(num_projections):
            rng = np.random.default_rng(seed + 1_000_003 * projection_id + 9_176 * group_id)
            direction = rng.normal(size=Z.shape[1]).astype(np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                continue
            direction /= norm
            projected = group_Z @ direction
            ordered = group[np.lexsort((group, projected))]
            for pos, i in enumerate(ordered):
                i = int(i)
                if proposal_counts[i] >= per_node_budget:
                    continue
                stop = min(len(ordered), pos + window_size + 1)
                for j in ordered[pos + 1 : stop]:
                    j = int(j)
                    if proposal_counts[i] >= per_node_budget:
                        break
                    if proposal_counts[j] >= per_node_budget:
                        continue
                    diff = Z[i] - Z[j]
                    store.add(i, j, float(np.dot(diff, diff)), "partition_ann")
                    proposal_counts[i] += 1
                    proposal_counts[j] += 1
                    pairs_considered += 1

    return {
        "groups_considered": groups_considered,
        "pairs_considered": pairs_considered,
        "max_pairs_considered_per_node": int(proposal_counts.max(initial=0)),
    }
