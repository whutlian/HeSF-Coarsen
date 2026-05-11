import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.candidates.bucket import generate_bucket_candidates
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.sketch.simhash import compute_simhash_buckets


def test_candidate_store_bounds_counts_and_deduplicates_pairs():
    node_type = np.array([0, 0, 0, 0, 1], dtype=np.int32)
    store = BoundedCandidateStore(node_type, K=2)

    store.add(0, 1, 0.5, "onehop")
    store.add(1, 0, 0.4, "onehop")
    store.add(0, 2, 0.3, "bucket")
    store.add(0, 3, 0.2, "bucket")
    store.add(0, 4, 0.1, "bucket")
    store.add(0, 0, 0.0, "bucket")

    pairs = store.to_pairs()

    assert store.counts()[0] <= 2
    assert len({tuple(pair[:2].astype(int)) for pair in pairs}) == len(pairs)
    assert all(node_type[int(i)] == node_type[int(j)] for i, j, *_ in pairs)


def test_bucket_candidates_respect_type_and_partition():
    z = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [1.1, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.2],
        ],
        dtype=np.float32,
    )
    node_type = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    partition_id = np.array([0, 0, 1, 0, 0, 0], dtype=np.int32)
    buckets = compute_simhash_buckets(z, node_type, partition_id, bits=2, seed=13)
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=8,
        bucket_pair_cap=20,
    )
    store = BoundedCandidateStore(node_type, K=8)

    generate_bucket_candidates(buckets, node_type, partition_id, config, store)

    for i, j, *_ in store.to_pairs():
        i = int(i)
        j = int(j)
        assert node_type[i] == node_type[j]
        assert partition_id[i] == partition_id[j]
