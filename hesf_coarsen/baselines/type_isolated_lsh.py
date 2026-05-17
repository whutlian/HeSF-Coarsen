from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


def _node_embedding(graph: HeteroGraph, seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    embedding = np.zeros((graph.num_nodes, max(int(dim), 1)), dtype=np.float32)
    if graph.features:
        for type_id, feature in graph.features.items():
            nodes = nodes_of_type(graph, int(type_id))
            local = feature.astype(np.float32, copy=False)
            if local.shape[1] >= embedding.shape[1]:
                embedding[nodes] = local[:, : embedding.shape[1]]
            else:
                embedding[nodes, : local.shape[1]] = local
    degree = np.zeros(graph.num_nodes, dtype=np.float32)
    for rel in graph.relations.values():
        np.add.at(degree, rel.src, rel.weight.astype(np.float32, copy=False))
        np.add.at(degree, rel.dst, rel.weight.astype(np.float32, copy=False))
    embedding[:, 0] += degree
    embedding += rng.normal(scale=1.0e-4, size=embedding.shape).astype(np.float32)
    return embedding


def _signatures(embedding: np.ndarray, seed: int, bits: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 7919)
    planes = rng.standard_normal((embedding.shape[1], max(int(bits), 1))).astype(np.float32)
    signs = embedding @ planes >= 0.0
    sig = np.zeros(embedding.shape[0], dtype=np.int64)
    for bit in range(signs.shape[1]):
        sig |= signs[:, bit].astype(np.int64) << np.int64(bit)
    return sig


def coarsen_type_isolated_lsh(
    graph: HeteroGraph,
    *,
    target_ratio: float = 0.5,
    seed: int = 12345,
    max_cluster_size: int = 4,
    hash_bits: int = 8,
    same_partition_only: bool = True,
) -> tuple[HeteroGraph, Assignment, dict[str, Any]]:
    target_nodes = max(1, int(ceil(graph.num_nodes * float(target_ratio) - 1.0e-12)))
    merges_remaining = max(0, graph.num_nodes - target_nodes)
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types: list[int] = []
    embedding = _node_embedding(graph, int(seed), dim=8)
    signatures = _signatures(embedding, int(seed), int(hash_bits))
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    partitions = graph.partitions if graph.partitions is not None and same_partition_only else np.zeros(graph.num_nodes, dtype=np.int32)
    for node in range(graph.num_nodes):
        buckets[(int(graph.node_type[node]), int(partitions[node]), int(signatures[node]))].append(int(node))
    cluster_hist: dict[int, int] = {}
    for key in sorted(buckets):
        nodes = sorted(buckets[key])
        index = 0
        while index < len(nodes):
            if merges_remaining <= 0:
                break
            size = min(int(max_cluster_size), len(nodes) - index, merges_remaining + 1)
            if size < 2:
                index += 1
                continue
            super_id = len(super_types)
            for node in nodes[index : index + size]:
                assignment[node] = super_id
            super_types.append(int(graph.node_type[nodes[index]]))
            cluster_hist[size] = cluster_hist.get(size, 0) + 1
            merges_remaining -= size - 1
            index += size
    for node in range(graph.num_nodes):
        if assignment[node] >= 0:
            continue
        super_id = len(super_types)
        assignment[node] = super_id
        super_types.append(int(graph.node_type[node]))
        cluster_hist[1] = cluster_hist.get(1, 0) + 1
    assign = Assignment(assignment=assignment, supernode_type=np.asarray(super_types, dtype=np.int32))
    coarse = coarsen_graph(graph, assign)
    diagnostics = {
        "baseline_name": "AH-UGC-style protocol-matched baseline",
        "target_ratio": float(target_ratio),
        "final_ratio": float(coarse.num_nodes / max(graph.num_nodes, 1)),
        "target_hit": bool(abs(float(coarse.num_nodes / max(graph.num_nodes, 1)) - float(target_ratio)) <= 0.02),
        "hash_bits": int(hash_bits),
        "max_cluster_size": int(max_cluster_size),
        "same_partition_only": bool(same_partition_only),
        "cluster_size_hist": {str(k): int(v) for k, v in sorted(cluster_hist.items())},
    }
    return coarse, assign, diagnostics

