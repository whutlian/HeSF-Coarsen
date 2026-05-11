from __future__ import annotations

import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.io.schema import HeteroGraph


def generate_onehop_candidates(
    graph: HeteroGraph,
    Z: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
) -> None:
    same_partition = bool(config.get("coarsening", {}).get("same_partition_only", True))
    for rel in graph.relations.values():
        if rel.src_type != rel.dst_type:
            continue
        for src, dst in zip(rel.src, rel.dst):
            if same_partition and partition_id[src] != partition_id[dst]:
                continue
            diff = Z[src].astype(np.float32) - Z[dst].astype(np.float32)
            store.add(int(src), int(dst), float(np.dot(diff, diff)), "onehop")


def generate_onehop_candidates_chunked(
    graph: HeteroGraph,
    Z: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
    edge_chunk_size: int = 1_000_000,
) -> dict[str, int]:
    if edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")
    same_partition = bool(config.get("coarsening", {}).get("same_partition_only", True))
    emitted = 0
    for rel in graph.relations.values():
        if rel.src_type != rel.dst_type:
            continue
        for start in range(0, rel.num_edges, edge_chunk_size):
            stop = min(start + edge_chunk_size, rel.num_edges)
            src = rel.src[start:stop]
            dst = rel.dst[start:stop]
            if same_partition:
                mask = partition_id[src] == partition_id[dst]
                src = src[mask]
                dst = dst[mask]
            if len(src) == 0:
                continue
            diff = Z[src].astype(np.float32) - Z[dst].astype(np.float32)
            scores = np.einsum("ij,ij->i", diff, diff)
            store.add_many(src, dst, scores, "onehop") if hasattr(store, "add_many") else [
                store.add(int(i), int(j), float(score), "onehop")
                for i, j, score in zip(src, dst, scores)
            ]
            emitted += int(len(src))
    return {"pairs_considered": emitted}
