import numpy as np

from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.matching.greedy import (
    finalize_mutual_best,
    initialize_mutual_best_state,
    mutual_best_update_block,
    selected_pair_sources,
)
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


def test_candidate_store_iter_pair_blocks_matches_to_pairs():
    node_type = np.array([0, 0, 0, 0], dtype=np.int32)
    store = BoundedCandidateStore(node_type, K=3)

    store.add(0, 1, 0.5, "onehop")
    store.add(0, 2, 0.3, "bucket")
    store.add(2, 3, 0.2, "bucket")

    blocks = list(store.iter_pair_blocks(block_size=2))
    streamed = np.vstack(blocks)

    assert len(blocks) == 2
    assert np.allclose(streamed[np.lexsort((streamed[:, 1], streamed[:, 0]))], store.to_pairs())
    assert store.pair_count() == store.to_pairs().shape[0]


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


def test_bucket_multi_probe_emits_adjacent_hash_bucket_pairs():
    node_type = np.array([0, 0], dtype=np.int32)
    partition_id = np.array([0, 0], dtype=np.int32)
    buckets = np.array([0b00, 0b01], dtype=np.int64)
    base_config = dict(DEFAULT_CONFIG)
    base_config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=4,
        bucket_pair_cap=4,
        active_hash_bits=2,
        multi_probe=False,
    )
    exact_store = BoundedCandidateStore(node_type, K=4)
    generate_bucket_candidates(buckets, node_type, partition_id, base_config, exact_store)

    probe_config = dict(base_config)
    probe_config["candidates"] = dict(
        base_config["candidates"],
        multi_probe=True,
        hamming_radius=1,
        quotas={"bucket_min_fraction": 0.25},
    )
    probe_store = BoundedCandidateStore(node_type, K=4)
    generate_bucket_candidates(buckets, node_type, partition_id, probe_config, probe_store)

    assert exact_store.pair_count() == 0
    assert probe_store.pair_count() == 1
    assert probe_store.source_counts()["bucket"] == 1


def test_mutual_best_can_enforce_selected_match_source_quota():
    graph = HeteroGraph(
        num_nodes=8,
        node_type=np.zeros(8, dtype=np.int32),
        relations={},
    )
    scored_pairs = np.array(
        [
            [0, 1, 0.01],
            [2, 3, 0.02],
            [4, 5, 0.50],
            [6, 7, 0.51],
        ],
        dtype=np.float64,
    )
    sources = {
        (0, 1): "capped_twohop",
        (2, 3): "capped_twohop",
        (4, 5): "bucket",
        (6, 7): "bucket",
    }

    def source_for_pair(i, j):
        key = (i, j) if i < j else (j, i)
        return sources.get(key)

    config = {
        "coarsening": {"matching_method": "mutual_best", "max_matched_pairs": 2},
        "candidates": {
            "quotas": {
                "enforce_on": "selected_matches",
                "bucket_min_fraction": 0.5,
                "twohop_max_fraction": 0.5,
            }
        },
    }
    state = initialize_mutual_best_state(graph)
    mutual_best_update_block(graph, state, scored_pairs, config)

    assignment = finalize_mutual_best(
        graph,
        state,
        config,
        source_lookup=source_for_pair,
    )

    selected = selected_pair_sources(assignment, source_for_pair)
    assert selected == {"capped_twohop": 1, "bucket": 1}
    quota = state.selected_quota_diagnostics
    assert quota["enforced"] is True
    assert quota["selected_match_source_distribution_before_quota"]["capped_twohop"] == 2
    assert quota["selected_match_source_distribution_after_quota"]["bucket"] == 1
    assert quota["quota_violation"]["bucket"] == 0.0


def test_multitable_bucket_config_can_increase_candidate_coverage(tmp_path):
    from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
    from hesf_coarsen.io.edge_list import generate_synthetic_graph

    graph = generate_synthetic_graph(num_users=10, num_items=6, num_tags=4, seed=61)
    config = dict(DEFAULT_CONFIG)
    config["output"] = {"dir": str(tmp_path)}
    config["diagnostics"] = dict(config["diagnostics"], enable_spectral=False)
    config["coarsening"] = dict(
        config["coarsening"],
        target_ratio=0.8,
        max_levels=1,
        per_level_ratio=0.8,
    )
    config["sketch"] = dict(config["sketch"], dim=32, order=2, method="lazy", dtype="float32")
    config["candidates"] = dict(
        config["candidates"],
        enable_onehop=False,
        enable_capped_twohop=False,
        enable_bucket=True,
        enable_fallback=False,
        total_budget_K=8,
        bucket_pair_cap=32,
        hash_tables=[4, 8],
        multi_probe=True,
        adaptive_hamming_radius=True,
    )

    result = run_multilevel_coarsening(graph, config)[0]

    assert result.diagnostics["config"]["candidates"]["hash_tables"] == [4, 8]
    assert result.diagnostics["config"]["candidates"]["multi_probe"] is True
    assert result.diagnostics["candidate_source_counts"].get("bucket", 0) > 0
