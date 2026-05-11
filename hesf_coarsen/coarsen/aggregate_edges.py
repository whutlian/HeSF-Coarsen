from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, nodes_of_type, validate_schema


def _aggregate_features(graph: HeteroGraph, assignment: Assignment) -> dict[int, np.ndarray] | None:
    if graph.features is None:
        return None
    result: dict[int, np.ndarray] = {}
    for type_id, feature in graph.features.items():
        old_nodes = nodes_of_type(graph, type_id)
        supernodes = np.flatnonzero(assignment.supernode_type == int(type_id)).astype(np.int64)
        if len(supernodes) == 0:
            result[type_id] = np.empty((0, feature.shape[1]), dtype=np.float32)
            continue
        rows = np.zeros((len(supernodes), feature.shape[1]), dtype=np.float32)
        positions = np.searchsorted(supernodes, assignment.assignment[old_nodes])
        np.add.at(rows, positions, feature.astype(np.float32, copy=False))
        counts = np.bincount(positions, minlength=len(supernodes)).astype(np.float32)
        rows /= np.maximum(counts[:, None], 1.0)
        result[type_id] = rows
    return result


def _aggregate_labels(graph: HeteroGraph, assignment: Assignment) -> np.ndarray | None:
    if graph.labels is None:
        return None
    labels = np.full(assignment.num_supernodes, -1, dtype=graph.labels.dtype)
    flat_labels = np.asarray(graph.labels).reshape(-1)
    valid = flat_labels >= 0
    if not np.any(valid):
        return labels
    supernodes = assignment.assignment[valid].astype(np.int64, copy=False)
    values = flat_labels[valid].astype(np.int64, copy=False)
    label_base = int(values.max(initial=0)) + 1
    pair_keys = supernodes * np.int64(label_base) + values
    order = np.argsort(pair_keys, kind="mergesort")
    sorted_keys = pair_keys[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    reduced_keys = sorted_keys[boundaries]
    counts = np.diff(np.r_[boundaries, len(sorted_keys)]).astype(np.int64)
    reduced_supernodes = reduced_keys // np.int64(label_base)
    reduced_labels = reduced_keys % np.int64(label_base)

    # Sort by supernode, descending count, ascending label. The first row for
    # each supernode is therefore the deterministic majority label.
    best_order = np.lexsort((reduced_labels, -counts, reduced_supernodes))
    ordered_supernodes = reduced_supernodes[best_order]
    first = np.r_[0, np.flatnonzero(ordered_supernodes[1:] != ordered_supernodes[:-1]) + 1]
    labels[ordered_supernodes[first]] = reduced_labels[best_order[first]].astype(labels.dtype)
    return labels


def coarsen_graph(graph: HeteroGraph, assignment: Assignment) -> HeteroGraph:
    if assignment.assignment.shape != (graph.num_nodes,):
        raise ValueError("assignment length must equal graph.num_nodes")
    for node, supernode in enumerate(assignment.assignment):
        expected = graph.node_type[node]
        actual = assignment.supernode_type[supernode]
        if expected != actual:
            raise ValueError("coarse node type must match every cluster member")

    relations: dict[int, RelationAdj] = {}
    for relation_id, rel in graph.relations.items():
        weights: defaultdict[tuple[int, int], float] = defaultdict(float)
        for src, dst, weight in zip(rel.src, rel.dst, rel.weight):
            coarse_src = int(assignment.assignment[src])
            coarse_dst = int(assignment.assignment[dst])
            weights[(coarse_src, coarse_dst)] += float(weight)
        items = sorted(weights.items())
        src = np.asarray([key[0] for key, _value in items], dtype=np.int64)
        dst = np.asarray([key[1] for key, _value in items], dtype=np.int64)
        weight = np.asarray([value for _key, value in items], dtype=np.float32)
        relations[relation_id] = RelationAdj(
            src=src,
            dst=dst,
            weight=weight,
            src_type=rel.src_type,
            dst_type=rel.dst_type,
            relation_id=relation_id,
        )

    specs = {
        relation_id: RelationSpec(
            relation_id=spec.relation_id,
            name=spec.name,
            src_type=spec.src_type,
            dst_type=spec.dst_type,
        )
        for relation_id, spec in graph.relation_specs.items()
    }
    coarse = HeteroGraph(
        num_nodes=assignment.num_supernodes,
        node_type=assignment.supernode_type.copy(),
        relations=relations,
        relation_specs=specs,
        features=_aggregate_features(graph, assignment),
        labels=_aggregate_labels(graph, assignment),
    )
    validate_schema(coarse)
    return coarse


def coarsen_graph_chunked(
    graph: HeteroGraph,
    assignment: Assignment,
    chunk_size: int = 1_000_000,
    output_dir: str | Path | None = None,
    reducer: str = "sort",
) -> HeteroGraph:
    """Coarsen relation edges in chunks.

    This explicit large-graph path avoids building a full per-edge coarse table.
    The default ``sort`` reducer does vectorized per-chunk sort-reduce and then
    merges reduced chunk tables. ``hash`` keeps the older Python dictionary path
    for debugging on small graphs.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if reducer not in {"sort", "hash"}:
        raise ValueError("reducer must be either 'sort' or 'hash'")
    if assignment.assignment.shape != (graph.num_nodes,):
        raise ValueError("assignment length must equal graph.num_nodes")
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for node, supernode in enumerate(assignment.assignment):
        expected = graph.node_type[node]
        actual = assignment.supernode_type[supernode]
        if expected != actual:
            raise ValueError("coarse node type must match every cluster member")

    relations: dict[int, RelationAdj] = {}
    for relation_id, rel in graph.relations.items():
        if reducer == "hash":
            src, dst, weight = _aggregate_relation_hash(rel, assignment, chunk_size)
        else:
            src, dst, weight = _aggregate_relation_sort(rel, assignment, chunk_size)
        relations[relation_id] = RelationAdj(
            src=src,
            dst=dst,
            weight=weight,
            src_type=rel.src_type,
            dst_type=rel.dst_type,
            relation_id=relation_id,
        )

    specs = {
        relation_id: RelationSpec(
            relation_id=spec.relation_id,
            name=spec.name,
            src_type=spec.src_type,
            dst_type=spec.dst_type,
        )
        for relation_id, spec in graph.relation_specs.items()
    }
    coarse = HeteroGraph(
        num_nodes=assignment.num_supernodes,
        node_type=assignment.supernode_type.copy(),
        relations=relations,
        relation_specs=specs,
        features=_aggregate_features(graph, assignment),
        labels=_aggregate_labels(graph, assignment),
    )
    validate_schema(coarse)
    return coarse


def _aggregate_relation_hash(
    rel: RelationAdj,
    assignment: Assignment,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced: defaultdict[tuple[int, int], float] = defaultdict(float)
    for start in range(0, rel.num_edges, chunk_size):
        stop = min(start + chunk_size, rel.num_edges)
        coarse_src = assignment.assignment[rel.src[start:stop]]
        coarse_dst = assignment.assignment[rel.dst[start:stop]]
        weights = rel.weight[start:stop]
        for src, dst, weight in zip(coarse_src, coarse_dst, weights):
            reduced[(int(src), int(dst))] += float(weight)
    items = sorted(reduced.items())
    return (
        np.asarray([key[0] for key, _value in items], dtype=np.int64),
        np.asarray([key[1] for key, _value in items], dtype=np.int64),
        np.asarray([value for _key, value in items], dtype=np.float32),
    )


def _reduce_sorted_keys(
    keys: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return keys.astype(np.int64), weights.astype(np.float32)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_weights = weights[order].astype(np.float32, copy=False)
    boundaries = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    reduced_keys = sorted_keys[boundaries]
    reduced_weights = np.add.reduceat(sorted_weights, boundaries).astype(np.float32)
    return reduced_keys.astype(np.int64, copy=False), reduced_weights


def _aggregate_relation_sort(
    rel: RelationAdj,
    assignment: Assignment,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_supernodes = assignment.num_supernodes
    chunk_keys: list[np.ndarray] = []
    chunk_weights: list[np.ndarray] = []
    for start in range(0, rel.num_edges, chunk_size):
        stop = min(start + chunk_size, rel.num_edges)
        coarse_src = assignment.assignment[rel.src[start:stop]].astype(np.int64, copy=False)
        coarse_dst = assignment.assignment[rel.dst[start:stop]].astype(np.int64, copy=False)
        keys = coarse_src * np.int64(num_supernodes) + coarse_dst
        reduced_keys, reduced_weights = _reduce_sorted_keys(keys, rel.weight[start:stop])
        chunk_keys.append(reduced_keys)
        chunk_weights.append(reduced_weights)

    if not chunk_keys:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    all_keys = np.concatenate(chunk_keys)
    all_weights = np.concatenate(chunk_weights)
    final_keys, final_weights = _reduce_sorted_keys(all_keys, all_weights)
    src = (final_keys // np.int64(num_supernodes)).astype(np.int64)
    dst = (final_keys % np.int64(num_supernodes)).astype(np.int64)
    return src, dst, final_weights
