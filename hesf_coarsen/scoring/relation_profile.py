from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph


def compute_relation_profiles(graph: HeteroGraph) -> np.ndarray:
    relation_ids = sorted(graph.relations)
    profile = np.zeros((graph.num_nodes, 2 * len(relation_ids)), dtype=np.float32)
    for idx, relation_id in enumerate(relation_ids):
        rel = graph.relations[relation_id]
        np.add.at(profile[:, 2 * idx], rel.src, rel.weight)
        np.add.at(profile[:, 2 * idx + 1], rel.dst, rel.weight)
    denom = profile.sum(axis=1, keepdims=True)
    return profile / np.maximum(denom, 1e-6)
