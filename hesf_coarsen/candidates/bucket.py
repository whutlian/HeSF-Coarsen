from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product

import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.progress import progress_iter


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


def _bounded_cross_pairs(
    left: np.ndarray,
    right: np.ndarray,
    cap: int,
    seed: int,
) -> list[tuple[int, int]]:
    total = len(left) * len(right)
    if total <= cap:
        return [
            (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            for i, j in product(left.tolist(), right.tolist())
            if int(i) != int(j)
        ]
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < cap and attempts < max(cap * 12, 32):
        i = int(left[int(rng.integers(0, len(left)))])
        j = int(right[int(rng.integers(0, len(right)))])
        if i != j:
            pairs.add((i, j) if i < j else (j, i))
        attempts += 1
    return sorted(pairs)


def _quota_config(config: dict) -> dict:
    quotas = config.get("candidates", {}).get("quotas", {}) or {}
    return quotas if isinstance(quotas, dict) else {}


def _bucket_score(config: dict) -> float:
    # A positive bucket_min_fraction asks bucket candidates to survive local top-K
    # competition; store retention treats lower scores as better.
    bucket_min = float(_quota_config(config).get("bucket_min_fraction", 0.0) or 0.0)
    return -max(bucket_min, 0.0)


def _effective_hamming_radius(nodes: list[int], config: dict) -> int:
    candidate_cfg = config.get("candidates", {})
    if not bool(candidate_cfg.get("multi_probe", False)):
        return 0
    configured = candidate_cfg.get("hamming_radius", None)
    radius = 1 if configured is None else int(configured)
    if bool(candidate_cfg.get("adaptive_hamming_radius", False)) and len(nodes) < 2:
        radius = max(radius, 1)
    return max(radius, 0)


def _neighbor_bucket_keys(bucket: int, bits: int, radius: int) -> list[int]:
    if radius <= 0:
        return []
    radius = min(int(radius), int(bits))
    low_mask = (1 << int(bits)) - 1
    high = int(bucket) & ~low_mask
    hash_bits = int(bucket) & low_mask
    keys: list[int] = []
    for distance in range(1, radius + 1):
        for flip_bits in combinations(range(int(bits)), distance):
            mask = 0
            for bit in flip_bits:
                mask |= 1 << bit
            keys.append(high | ((hash_bits ^ mask) & low_mask))
    return keys


def _emit_bucket_candidates(
    groups: dict[int, list[int]],
    node_type: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
    *,
    table_seed: int,
) -> int:
    candidate_cfg = config.get("candidates", {})
    cap = int(candidate_cfg.get("bucket_pair_cap", 64))
    bits = int(candidate_cfg.get("active_hash_bits", candidate_cfg.get("simhash_bits", 16)))
    score = _bucket_score(config)
    emitted = 0
    seen: set[tuple[int, int]] = set()
    for bucket, nodes in sorted(groups.items()):
        if len(nodes) < 2 and _effective_hamming_radius(nodes, config) <= 0:
            continue
        arr = np.asarray(sorted(nodes), dtype=np.int64)
        emitted_for_bucket = 0
        for i, j in _bounded_pairs(arr, cap, table_seed + bucket % 1_000_003):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            if node_type[i] != node_type[j] or partition_id[i] != partition_id[j]:
                continue
            store.add(i, j, score, "bucket")
            emitted += 1
            emitted_for_bucket += 1

        radius = _effective_hamming_radius(nodes, config)
        if radius <= 0:
            continue
        remaining = max(cap - emitted_for_bucket, 1)
        neighbors = [
            key for key in _neighbor_bucket_keys(bucket, bits=bits, radius=radius) if key in groups
        ]
        per_neighbor_cap = max(1, remaining // max(len(neighbors), 1))
        for neighbor in neighbors:
            if int(neighbor) <= int(bucket):
                continue
            other = np.asarray(sorted(groups[neighbor]), dtype=np.int64)
            for i, j in _bounded_cross_pairs(
                arr,
                other,
                per_neighbor_cap,
                table_seed + (bucket ^ int(neighbor)) % 1_000_003,
            ):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if node_type[i] != node_type[j] or partition_id[i] != partition_id[j]:
                    continue
                store.add(i, j, score, "bucket")
                emitted += 1
    return emitted


def generate_bucket_candidates(
    buckets: np.ndarray,
    node_type: np.ndarray,
    partition_id: np.ndarray,
    config: dict,
    store: BoundedCandidateStore,
) -> None:
    seed = int(config.get("seed", 12345))
    groups: dict[int, list[int]] = defaultdict(list)
    for node, bucket in enumerate(np.asarray(buckets, dtype=np.int64)):
        groups[int(bucket)].append(node)

    _emit_bucket_candidates(
        groups,
        node_type,
        partition_id,
        config,
        store,
        table_seed=seed,
    )


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
    seed = int(config.get("seed", 12345))
    emitted = 0
    buckets = np.asarray(buckets, dtype=np.int64)
    ranges = range(0, len(buckets), node_chunk_size)
    total = (len(buckets) + node_chunk_size - 1) // node_chunk_size
    for start in progress_iter(
        ranges,
        total=total,
        desc="bucket candidates",
        config=config,
        unit="chunk",
    ):
        stop = min(start + node_chunk_size, len(buckets))
        groups: dict[int, list[int]] = defaultdict(list)
        for local_node, bucket in enumerate(buckets[start:stop], start=start):
            groups[int(bucket)].append(local_node)
        emitted += _emit_bucket_candidates(
            groups,
            node_type,
            partition_id,
            config,
            store,
            table_seed=seed,
        )
    return {"pairs_considered": emitted}
