from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore


def _bounded_pairs(nodes: np.ndarray, cap: int, seed: int) -> list[tuple[int, int]]:
    total = len(nodes) * (len(nodes) - 1) // 2
    if total <= cap:
        return [(int(i), int(j)) for i, j in combinations(nodes.tolist(), 2)]
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < cap and attempts < max(cap * 12, 32):
        a, b = rng.choice(nodes, size=2, replace=False)
        i, j = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        pairs.add((i, j))
        attempts += 1
    return sorted(pairs)


def generate_bucket_candidates(
    buckets: np.ndarray,
    node_type: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
) -> None:
    candidate_cfg = config.get("candidates", {})
    cap = int(candidate_cfg.get("bucket_pair_cap", 64))
    seed = int(config.get("seed", 12345))
    groups: dict[int, list[int]] = defaultdict(list)
    for node, bucket in enumerate(np.asarray(buckets, dtype=np.int64)):
        groups[int(bucket)].append(node)

    for bucket, nodes in sorted(groups.items()):
        if len(nodes) < 2:
            continue
        arr = np.asarray(sorted(nodes), dtype=np.int64)
        for i, j in _bounded_pairs(arr, cap, seed + bucket % 1_000_003):
            if node_type[i] != node_type[j]:
                continue
            if partition_id[i] != partition_id[j]:
                continue
            store.add(i, j, 0.0, "bucket")


def generate_bucket_candidates_chunked(
    buckets: np.ndarray,
    node_type: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
    node_chunk_size: int = 1_000_000,
) -> dict[str, int]:
    if node_chunk_size <= 0:
        raise ValueError("node_chunk_size must be positive")
    candidate_cfg = config.get("candidates", {})
    cap = int(candidate_cfg.get("bucket_pair_cap", 64))
    seed = int(config.get("seed", 12345))
    emitted = 0
    buckets = np.asarray(buckets, dtype=np.int64)
    for start in range(0, len(buckets), node_chunk_size):
        stop = min(start + node_chunk_size, len(buckets))
        groups: dict[int, list[int]] = defaultdict(list)
        for local_node, bucket in enumerate(buckets[start:stop], start=start):
            groups[int(bucket)].append(local_node)
        for bucket, nodes in sorted(groups.items()):
            if len(nodes) < 2:
                continue
            arr = np.asarray(sorted(nodes), dtype=np.int64)
            for i, j in _bounded_pairs(arr, cap, seed + bucket % 1_000_003):
                if node_type[i] != node_type[j]:
                    continue
                if partition_id[i] != partition_id[j]:
                    continue
                store.add(i, j, 0.0, "bucket")
                emitted += 1
    return {"pairs_considered": emitted}
