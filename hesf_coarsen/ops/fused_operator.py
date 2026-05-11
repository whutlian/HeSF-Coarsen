from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.relation_ops import apply_relation


def apply_fused_smoothing(
    graph: HeteroGraph,
    X: np.ndarray,
    relation_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Apply a relation-weighted smoothing operator relation by relation."""

    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] != graph.num_nodes:
        raise ValueError("X must have global shape [num_nodes, q]")
    if relation_weights is None:
        relation_weights = {relation_id: 1.0 for relation_id in graph.relations}

    accum = np.zeros_like(X, dtype=np.float32)
    total_abs = 0.0
    for relation_id in sorted(graph.relations):
        weight = float(relation_weights.get(relation_id, 0.0))
        if weight == 0.0:
            continue
        accum += weight * apply_relation(graph, relation_id, X, normalize=True)
        total_abs += abs(weight)

    if total_abs > 0:
        accum /= np.float32(total_abs)
    # A lazy self term keeps isolated nodes stable and makes repeated smoothing
    # a low-pass filter rather than a pure neighbor replacement.
    return (0.5 * X + 0.5 * accum).astype(np.float32)
