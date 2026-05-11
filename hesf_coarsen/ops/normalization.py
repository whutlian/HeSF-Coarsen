from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj


def relation_degrees(
    graph: HeteroGraph,
    rel: RelationAdj,
) -> tuple[np.ndarray, np.ndarray]:
    src_degree = np.zeros(graph.num_nodes, dtype=np.float32)
    dst_degree = np.zeros(graph.num_nodes, dtype=np.float32)
    np.add.at(src_degree, rel.src, rel.weight)
    np.add.at(dst_degree, rel.dst, rel.weight)
    return src_degree, dst_degree


def normalized_edge_weights(graph: HeteroGraph, rel: RelationAdj) -> np.ndarray:
    src_degree, dst_degree = relation_degrees(graph, rel)
    denom = np.sqrt(src_degree[rel.src] * dst_degree[rel.dst])
    denom = np.maximum(denom, 1e-12)
    return (rel.weight / denom).astype(np.float32)
