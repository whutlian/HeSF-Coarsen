from __future__ import annotations

import heapq
import shutil
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec, nodes_of_type, validate_schema


_FEATURE_AGGREGATION_METHODS = {
    "mean",
    "degree_weighted",
    "pagerank_weighted",
    "custom_weight",
}


def _incident_weight_mass(graph: HeteroGraph) -> np.ndarray:
    weights = np.zeros(graph.num_nodes, dtype=np.float32)
    for rel in graph.relations.values():
        np.add.at(weights, rel.src, rel.weight.astype(np.float32, copy=False))
        np.add.at(weights, rel.dst, rel.weight.astype(np.float32, copy=False))
    return weights


def _pagerank_weights(
    graph: HeteroGraph,
    *,
    iterations: int = 20,
    damping: float = 0.85,
) -> np.ndarray:
    if graph.num_nodes == 0:
        return np.empty(0, dtype=np.float32)
    iterations = max(int(iterations), 1)
    damping = min(max(float(damping), 0.0), 1.0)
    degree = _incident_weight_mass(graph).astype(np.float64, copy=False)
    rank = np.full(graph.num_nodes, 1.0 / graph.num_nodes, dtype=np.float64)
    teleport = (1.0 - damping) / graph.num_nodes
    for _ in range(iterations):
        next_rank = np.full(graph.num_nodes, teleport, dtype=np.float64)
        dangling_mass = float(rank[degree <= 0.0].sum())
        if dangling_mass:
            next_rank += damping * dangling_mass / graph.num_nodes
        for rel in graph.relations.values():
            weight = rel.weight.astype(np.float64, copy=False)
            src_denom = np.maximum(degree[rel.src], 1.0e-12)
            dst_denom = np.maximum(degree[rel.dst], 1.0e-12)
            np.add.at(next_rank, rel.dst, damping * rank[rel.src] * weight / src_denom)
            np.add.at(next_rank, rel.src, damping * rank[rel.dst] * weight / dst_denom)
        total = float(next_rank.sum())
        rank = next_rank / max(total, 1.0e-12)
    return rank.astype(np.float32)


def _custom_feature_weights(
    graph: HeteroGraph,
    feature_weights: np.ndarray | dict[int, np.ndarray] | None,
) -> np.ndarray:
    if feature_weights is None:
        raise ValueError("coarsening.feature_aggregation=custom_weight requires feature weights")
    if isinstance(feature_weights, dict):
        weights = np.zeros(graph.num_nodes, dtype=np.float32)
        for type_id, values in feature_weights.items():
            nodes = nodes_of_type(graph, int(type_id))
            typed = np.asarray(values, dtype=np.float32)
            if typed.shape != (len(nodes),):
                raise ValueError(
                    f"custom feature weights for type {type_id} must have shape {(len(nodes),)}"
                )
            weights[nodes] = typed
    else:
        weights = np.asarray(feature_weights, dtype=np.float32)
        if weights.shape != (graph.num_nodes,):
            raise ValueError("custom feature weights must have shape [num_nodes]")
    if not np.all(np.isfinite(weights)):
        raise ValueError("custom feature weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("custom feature weights must be non-negative")
    return weights.astype(np.float32, copy=False)


def _feature_weight_vector(
    graph: HeteroGraph,
    method: str,
    *,
    feature_weights: np.ndarray | dict[int, np.ndarray] | None = None,
    pagerank_iterations: int = 20,
    pagerank_damping: float = 0.85,
) -> np.ndarray:
    method = str(method).lower()
    if method not in _FEATURE_AGGREGATION_METHODS:
        raise ValueError(f"unsupported coarsening.feature_aggregation: {method}")
    if method == "mean":
        return np.ones(graph.num_nodes, dtype=np.float32)
    if method == "degree_weighted":
        return _incident_weight_mass(graph)
    if method == "pagerank_weighted":
        return _pagerank_weights(
            graph,
            iterations=pagerank_iterations,
            damping=pagerank_damping,
        )
    return _custom_feature_weights(graph, feature_weights)


def _aggregate_features(
    graph: HeteroGraph,
    assignment: Assignment,
    *,
    feature_aggregation: str = "mean",
    feature_weights: np.ndarray | dict[int, np.ndarray] | None = None,
    pagerank_iterations: int = 20,
    pagerank_damping: float = 0.85,
) -> dict[int, np.ndarray] | None:
    if graph.features is None:
        return None
    node_weights = _feature_weight_vector(
        graph,
        feature_aggregation,
        feature_weights=feature_weights,
        pagerank_iterations=pagerank_iterations,
        pagerank_damping=pagerank_damping,
    )
    result: dict[int, np.ndarray] = {}
    for type_id, feature in graph.features.items():
        old_nodes = nodes_of_type(graph, type_id)
        supernodes = np.flatnonzero(assignment.supernode_type == int(type_id)).astype(np.int64)
        if len(supernodes) == 0:
            result[type_id] = np.empty((0, feature.shape[1]), dtype=np.float32)
            continue
        rows = np.zeros((len(supernodes), feature.shape[1]), dtype=np.float32)
        fallback_rows = np.zeros_like(rows)
        positions = np.searchsorted(supernodes, assignment.assignment[old_nodes])
        typed_features = feature.astype(np.float32, copy=False)
        typed_weights = node_weights[old_nodes].astype(np.float32, copy=False)
        np.add.at(rows, positions, typed_features * typed_weights[:, None])
        np.add.at(fallback_rows, positions, typed_features)
        counts = np.bincount(positions, minlength=len(supernodes)).astype(np.float32)
        weight_sums = np.bincount(
            positions,
            weights=typed_weights,
            minlength=len(supernodes),
        ).astype(np.float32)
        weighted = weight_sums > 0.0
        rows[weighted] /= weight_sums[weighted][:, None]
        rows[~weighted] = fallback_rows[~weighted] / np.maximum(counts[~weighted][:, None], 1.0)
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


def coarsen_graph(
    graph: HeteroGraph,
    assignment: Assignment,
    *,
    feature_aggregation: str = "mean",
    feature_weights: np.ndarray | dict[int, np.ndarray] | None = None,
    pagerank_iterations: int = 20,
    pagerank_damping: float = 0.85,
) -> HeteroGraph:
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
        features=_aggregate_features(
            graph,
            assignment,
            feature_aggregation=feature_aggregation,
            feature_weights=feature_weights,
            pagerank_iterations=pagerank_iterations,
            pagerank_damping=pagerank_damping,
        ),
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
    feature_aggregation: str = "mean",
    feature_weights: np.ndarray | dict[int, np.ndarray] | None = None,
    pagerank_iterations: int = 20,
    pagerank_damping: float = 0.85,
) -> HeteroGraph:
    """Coarsen relation edges in chunks.

    This explicit large-graph path avoids building a full per-edge coarse table.
    The default ``sort`` reducer does vectorized per-chunk sort-reduce and then
    merges reduced chunk shards with a k-way merge. When ``output_dir`` is set,
    the sort reducer spills chunk shards and final relation arrays under
    ``output_dir/_aggregation_shards``. ``hash`` keeps the older Python
    dictionary path for debugging on small graphs.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if reducer not in {"sort", "hash"}:
        raise ValueError("reducer must be either 'sort' or 'hash'")
    if assignment.assignment.shape != (graph.num_nodes,):
        raise ValueError("assignment length must equal graph.num_nodes")
    spill_root = None
    if output_dir is not None:
        spill_root = Path(output_dir) / "_aggregation_shards"
        spill_root.mkdir(parents=True, exist_ok=True)

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
            relation_spill_dir = None if spill_root is None else spill_root / f"relation_{relation_id}"
            src, dst, weight = _aggregate_relation_sort(
                rel,
                assignment,
                chunk_size,
                output_dir=relation_spill_dir,
            )
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
        features=_aggregate_features(
            graph,
            assignment,
            feature_aggregation=feature_aggregation,
            feature_weights=feature_weights,
            pagerank_iterations=pagerank_iterations,
            pagerank_damping=pagerank_damping,
        ),
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


def _encode_coarse_keys(
    coarse_src: np.ndarray,
    coarse_dst: np.ndarray,
    num_supernodes: int,
) -> np.ndarray:
    if len(coarse_src) == 0:
        return np.empty(0, dtype=np.int64)
    max_src = int(coarse_src.max(initial=0))
    max_dst = int(coarse_dst.max(initial=0))
    max_int64 = np.iinfo(np.int64).max
    if num_supernodes > 0 and max_src > (max_int64 - max_dst) // int(num_supernodes):
        raise OverflowError("coarse edge key encoding would overflow int64")
    return coarse_src * np.int64(num_supernodes) + coarse_dst


def _iter_merged_sorted_chunks(
    chunks: list[tuple[np.ndarray, np.ndarray]],
) -> Iterator[tuple[int, float]]:
    positions = [0] * len(chunks)
    heap: list[tuple[int, int]] = []
    for chunk_id, (keys, _weights) in enumerate(chunks):
        if len(keys) > 0:
            heapq.heappush(heap, (int(keys[0]), chunk_id))

    current_key: int | None = None
    current_weight = 0.0
    while heap:
        key, chunk_id = heapq.heappop(heap)
        pos = positions[chunk_id]
        weight = float(chunks[chunk_id][1][pos])
        if current_key is None:
            current_key = key
            current_weight = weight
        elif key == current_key:
            current_weight += weight
        else:
            yield current_key, current_weight
            current_key = key
            current_weight = weight

        pos += 1
        positions[chunk_id] = pos
        keys = chunks[chunk_id][0]
        if pos < len(keys):
            heapq.heappush(heap, (int(keys[pos]), chunk_id))

    if current_key is not None:
        yield current_key, current_weight


def _merge_sorted_chunks(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    num_supernodes: int,
    output_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = sum(1 for _key, _weight in _iter_merged_sorted_chunks(chunks))
    if output_dir is None:
        src = np.empty(count, dtype=np.int64)
        dst = np.empty(count, dtype=np.int64)
        weight = np.empty(count, dtype=np.float32)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        src = np.lib.format.open_memmap(
            output_dir / "final_src.npy",
            mode="w+",
            dtype=np.int64,
            shape=(count,),
        )
        dst = np.lib.format.open_memmap(
            output_dir / "final_dst.npy",
            mode="w+",
            dtype=np.int64,
            shape=(count,),
        )
        weight = np.lib.format.open_memmap(
            output_dir / "final_weight.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count,),
        )

    for pos, (key, merged_weight) in enumerate(_iter_merged_sorted_chunks(chunks)):
        src[pos] = key // int(num_supernodes)
        dst[pos] = key % int(num_supernodes)
        weight[pos] = np.float32(merged_weight)

    for array in (src, dst, weight):
        flush = getattr(array, "flush", None)
        if callable(flush):
            flush()
    return src, dst, weight


def _write_sorted_chunk_shard(
    chunk_dir: Path,
    chunk_id: int,
    keys: np.ndarray,
    weights: np.ndarray,
) -> tuple[Path, Path]:
    key_path = chunk_dir / f"chunk_{chunk_id:06d}_keys.npy"
    weight_path = chunk_dir / f"chunk_{chunk_id:06d}_weights.npy"
    np.save(key_path, keys.astype(np.int64, copy=False))
    np.save(weight_path, weights.astype(np.float32, copy=False))
    return key_path, weight_path


def _aggregate_relation_sort(
    rel: RelationAdj,
    assignment: Assignment,
    chunk_size: int,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_supernodes = assignment.num_supernodes
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    chunk_paths: list[tuple[Path, Path]] = []
    chunk_dir = None
    if output_dir is not None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        chunk_dir = output_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, rel.num_edges, chunk_size):
        stop = min(start + chunk_size, rel.num_edges)
        coarse_src = assignment.assignment[rel.src[start:stop]].astype(np.int64, copy=False)
        coarse_dst = assignment.assignment[rel.dst[start:stop]].astype(np.int64, copy=False)
        keys = _encode_coarse_keys(coarse_src, coarse_dst, num_supernodes)
        reduced_keys, reduced_weights = _reduce_sorted_keys(keys, rel.weight[start:stop])
        if chunk_dir is None:
            chunks.append((reduced_keys, reduced_weights))
        else:
            chunk_id = len(chunk_paths)
            chunk_paths.append(
                _write_sorted_chunk_shard(chunk_dir, chunk_id, reduced_keys, reduced_weights)
            )

    if chunk_paths:
        chunks = [
            (
                np.load(key_path, mmap_mode="r"),
                np.load(weight_path, mmap_mode="r"),
            )
            for key_path, weight_path in chunk_paths
        ]
    return _merge_sorted_chunks(chunks, num_supernodes, output_dir)
