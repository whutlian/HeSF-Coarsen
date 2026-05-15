from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.relation_ops import apply_relation


def _apply_single_relation_sum_forward(graph: HeteroGraph, X: np.ndarray) -> np.ndarray:
    src_degree = np.zeros(graph.num_nodes, dtype=np.float32)
    dst_degree = np.zeros(graph.num_nodes, dtype=np.float32)
    for rel in graph.relations.values():
        np.add.at(src_degree, rel.src, rel.weight)
        np.add.at(dst_degree, rel.dst, rel.weight)

    out = np.zeros_like(X, dtype=np.float32)
    for rel in graph.relations.values():
        denom = np.sqrt(src_degree[rel.src] * dst_degree[rel.dst])
        weights = rel.weight / np.maximum(denom, 1e-12)
        np.add.at(out, rel.dst, X[rel.src] * weights[:, None])
    return out.astype(np.float32, copy=False)


def apply_fused_smoothing(
    graph: HeteroGraph,
    X: np.ndarray,
    relation_weights: dict[int, float] | None = None,
    relation_operator_mode: str = "relationwise",
) -> np.ndarray:
    """Apply a relation-weighted smoothing operator relation by relation."""

    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] != graph.num_nodes:
        raise ValueError("X must have global shape [num_nodes, q]")
    if relation_weights is None:
        relation_weights = {relation_id: 1.0 for relation_id in graph.relations}

    mode = str(relation_operator_mode or "relationwise").lower().replace("-", "_")
    if mode in {"single", "single_relation", "single_relation_sum", "flatten", "flatten_sum"}:
        smoothed = _apply_single_relation_sum_forward(graph, X)
        return (0.5 * X + 0.5 * smoothed).astype(np.float32)
    if mode not in {"relationwise", "relation_aware", "per_relation"}:
        raise ValueError(
            "fusion.relation_operator_mode must be one of: relationwise, single_relation_sum"
        )

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
