from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type, type_local_index
from hesf_coarsen.ops.normalization import normalized_edge_weights


def _as_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        return matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("input matrix must be 1D or 2D")
    return matrix


def apply_relation(
    graph: HeteroGraph,
    relation_id: int,
    X_src: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Apply one relation without constructing an adjacency matrix.

    If ``X_src`` has one row per graph node, the output also has global shape
    ``[num_nodes, q]`` with non-zero rows only at destination nodes. If it has
    one row per source-type node, the output has one row per destination-type
    node in global node order for that type.
    """

    rel = graph.relations[int(relation_id)]
    X_src = _as_2d(X_src).astype(np.float32, copy=False)
    weights = normalized_edge_weights(graph, rel) if normalize else rel.weight

    if X_src.shape[0] == graph.num_nodes:
        out = np.zeros((graph.num_nodes, X_src.shape[1]), dtype=np.float32)
        values = X_src[rel.src] * weights[:, None]
        np.add.at(out, rel.dst, values)
        return out

    src_nodes = nodes_of_type(graph, rel.src_type)
    dst_nodes = nodes_of_type(graph, rel.dst_type)
    if X_src.shape[0] != len(src_nodes):
        raise ValueError(
            f"X_src has {X_src.shape[0]} rows; expected {graph.num_nodes} "
            f"or {len(src_nodes)} source-type rows"
        )
    src_index = type_local_index(graph, rel.src_type)
    dst_index = type_local_index(graph, rel.dst_type)
    out = np.zeros((len(dst_nodes), X_src.shape[1]), dtype=np.float32)
    local_src = np.array([src_index[int(node)] for node in rel.src], dtype=np.int64)
    local_dst = np.array([dst_index[int(node)] for node in rel.dst], dtype=np.int64)
    values = X_src[local_src] * weights[:, None]
    np.add.at(out, local_dst, values)
    return out


def apply_relation_transpose(
    graph: HeteroGraph,
    relation_id: int,
    X_dst: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    rel = graph.relations[int(relation_id)]
    X_dst = _as_2d(X_dst).astype(np.float32, copy=False)
    weights = normalized_edge_weights(graph, rel) if normalize else rel.weight

    if X_dst.shape[0] == graph.num_nodes:
        out = np.zeros((graph.num_nodes, X_dst.shape[1]), dtype=np.float32)
        values = X_dst[rel.dst] * weights[:, None]
        np.add.at(out, rel.src, values)
        return out

    dst_nodes = nodes_of_type(graph, rel.dst_type)
    src_nodes = nodes_of_type(graph, rel.src_type)
    if X_dst.shape[0] != len(dst_nodes):
        raise ValueError(
            f"X_dst has {X_dst.shape[0]} rows; expected {graph.num_nodes} "
            f"or {len(dst_nodes)} destination-type rows"
        )
    dst_index = type_local_index(graph, rel.dst_type)
    src_index = type_local_index(graph, rel.src_type)
    out = np.zeros((len(src_nodes), X_dst.shape[1]), dtype=np.float32)
    local_dst = np.array([dst_index[int(node)] for node in rel.dst], dtype=np.int64)
    local_src = np.array([src_index[int(node)] for node in rel.src], dtype=np.int64)
    values = X_dst[local_dst] * weights[:, None]
    np.add.at(out, local_src, values)
    return out
