import numpy as np

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.candidates.bounded_heap import BoundedCandidateStore
from hesf_coarsen.candidates.bucket import generate_bucket_candidates_chunked
from hesf_coarsen.candidates import capped_twohop as capped_twohop_module
from hesf_coarsen.candidates.capped_twohop import generate_capped_twohop_candidates_chunked
from hesf_coarsen.candidates.onehop import generate_onehop_candidates_chunked
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.sketch.simhash import compute_simhash_buckets


def _stack_pair_blocks(blocks):
    blocks = list(blocks)
    if not blocks:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(blocks)


def _sort_pairs(pairs):
    if pairs.size == 0:
        return pairs
    order = np.lexsort((pairs[:, 2], pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _source_policy():
    return {
        "bucket": {"priority": "high", "topk_per_node": 8},
        "onehop": {"priority": "medium", "topk_per_node": 1},
    }


def test_array_candidate_store_bounds_dedupes_and_can_use_memmap(tmp_path):
    node_type = np.array([0, 0, 0, 0, 1], dtype=np.int32)
    store = ArrayCandidateStore(node_type, K=2, mmap_dir=tmp_path)

    store.add(0, 1, 0.5, "onehop")
    store.add(1, 0, 0.4, "onehop")
    store.add(0, 2, 0.3, "bucket")
    store.add(0, 3, 0.2, "bucket")
    store.add(0, 4, 0.1, "bucket")
    store.add(0, 0, 0.0, "bucket")
    store.flush()

    pairs = store.to_pairs()

    assert isinstance(store.candidate_ids, np.memmap)
    assert store.counts()[0] <= 2
    assert len({tuple(pair[:2].astype(int)) for pair in pairs}) == len(pairs)
    assert all(node_type[int(i)] == node_type[int(j)] for i, j, *_ in pairs)
    assert store.source_counts()


def test_array_candidate_store_iter_pair_blocks_matches_to_pairs():
    node_type = np.array([0, 0, 0, 0, 1], dtype=np.int32)
    store = ArrayCandidateStore(node_type, K=3)

    store.add(0, 1, 0.5, "onehop")
    store.add(1, 0, 0.4, "bucket")
    store.add(0, 2, 0.3, "bucket")
    store.add(2, 3, 0.2, "onehop")
    store.add(0, 4, 0.1, "bucket")
    store.add(3, 3, 0.0, "bucket")

    streamed = _stack_pair_blocks(store.iter_pair_blocks(block_size=1))

    assert np.allclose(_sort_pairs(streamed), _sort_pairs(store.to_pairs()))
    assert store.pair_count() == store.to_pairs().shape[0]


def test_array_candidate_store_keeps_best_duplicate_score():
    node_type = np.array([0, 0, 0], dtype=np.int32)
    store = ArrayCandidateStore(node_type, K=2)

    store.add(0, 1, 0.25, "onehop")
    store.add(0, 1, 0.75, "bucket")

    pairs = store.to_pairs()

    assert pairs.shape == (1, 3)
    assert tuple(pairs[0, :2].astype(int)) == (0, 1)
    assert np.isclose(pairs[0, 2], 0.25)


def test_array_candidate_store_removes_reciprocal_slot_on_eviction():
    node_type = np.array([0, 0, 0, 0], dtype=np.int32)
    store = ArrayCandidateStore(node_type, K=2)

    store.add(0, 1, 10.0, "onehop")
    store.add(0, 2, 5.0, "onehop")
    store.add(0, 3, 1.0, "onehop")

    pairs = {tuple(row[:2].astype(int)) for row in store.to_pairs()}

    assert (0, 1) not in pairs
    assert store.counts()[1] == 0


def test_candidate_stores_apply_source_topk_and_priority():
    for store_cls in (ArrayCandidateStore, BoundedCandidateStore):
        node_type = np.zeros(4, dtype=np.int32)
        store = store_cls(node_type, K=1, source_policies=_source_policy())

        store.add(0, 1, 0.1, "onehop")
        store.add(0, 2, 0.2, "onehop")
        store.add(0, 3, 100.0, "bucket")

        assert store.source_for_pair(0, 3) == "bucket"
        assert store.source_for_pair(0, 1) is None
        assert store.source_counts() == {"bucket": 1}


def test_chunked_onehop_candidates_use_array_store():
    graph = generate_synthetic_graph(num_users=8, num_items=5, num_tags=3, seed=606)
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(DEFAULT_CONFIG["candidates"], total_budget_K=3)
    store = ArrayCandidateStore(graph.node_type, K=3)
    z = np.zeros((graph.num_nodes, 4), dtype=np.float32)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)

    generate_onehop_candidates_chunked(
        graph,
        z,
        partition_id,
        config,
        store,
        edge_chunk_size=2,
    )

    assert store.counts().max(initial=0) <= 3
    assert store.source_counts().get("onehop", 0) > 0


def test_chunked_capped_twohop_keeps_per_node_and_per_middle_caps():
    graph = generate_synthetic_graph(num_users=20, num_items=4, num_tags=3, seed=707)
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=4,
        middle_degree_cap_policy="none",
        per_middle_pair_cap=5,
    )
    store = ArrayCandidateStore(graph.node_type, K=4)
    z = np.zeros((graph.num_nodes, 4), dtype=np.float32)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)

    stats = generate_capped_twohop_candidates_chunked(
        graph,
        z,
        partition_id,
        config,
        store,
        middle_chunk_size=3,
        edge_chunk_size=5,
    )

    assert store.counts().max(initial=0) <= 4
    assert stats["max_pairs_emitted_per_middle"] <= 5
    assert store.source_counts().get("capped_twohop", 0) > 0


def test_chunked_capped_twohop_limited_mode_caps_endpoint_participation():
    graph = generate_synthetic_graph(num_users=20, num_items=1, num_tags=2, seed=717)
    z = np.zeros((graph.num_nodes, 4), dtype=np.float32)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)
    base_config = dict(DEFAULT_CONFIG)
    base_config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=20,
        middle_degree_cap_policy="none",
        per_middle_pair_cap=20,
    )
    full_store = ArrayCandidateStore(graph.node_type, K=20)
    full_stats = generate_capped_twohop_candidates_chunked(
        graph,
        z,
        partition_id,
        base_config,
        full_store,
        middle_chunk_size=3,
        edge_chunk_size=5,
    )
    limited_config = dict(base_config)
    limited_config["candidates"] = dict(
        base_config["candidates"],
        twohop_mode="capped_sampled",
        twohop_budget_per_node=1,
    )
    limited_store = ArrayCandidateStore(graph.node_type, K=20)

    limited_stats = generate_capped_twohop_candidates_chunked(
        graph,
        z,
        partition_id,
        limited_config,
        limited_store,
        middle_chunk_size=3,
        edge_chunk_size=5,
    )

    assert limited_stats["twohop_mode"] == "capped_sampled"
    assert limited_stats["twohop_budget_per_node"] == 1
    assert limited_stats["pairs_skipped_by_node_budget"] > 0
    assert limited_stats["pairs_considered"] < full_stats["pairs_considered"]


def _canonical_incident(incident):
    return {
        (int(middle), int(endpoint_type)): sorted({int(node) for node in nodes})
        for middle, by_type in incident.items()
        for endpoint_type, nodes in by_type.items()
    }


def test_capped_twohop_incident_index_matches_range_scan():
    graph = generate_synthetic_graph(num_users=12, num_items=5, num_tags=4, seed=818)

    index = capped_twohop_module.CappedTwoHopIncidentIndex.from_graph(
        graph,
        edge_chunk_size=4,
    )
    indexed = index.collect_middle_range(3, 14)
    scanned = capped_twohop_module._collect_incident_for_middle_range(
        graph,
        3,
        14,
        edge_chunk_size=4,
    )

    assert _canonical_incident(indexed) == _canonical_incident(scanned)


def test_capped_twohop_incident_index_can_use_memmap(tmp_path):
    graph = generate_synthetic_graph(num_users=12, num_items=5, num_tags=4, seed=819)

    index = capped_twohop_module.CappedTwoHopIncidentIndex.from_graph(
        graph,
        edge_chunk_size=4,
        mmap_dir=tmp_path / "incident_index",
    )
    indexed = index.collect_middle_range(0, graph.num_nodes)
    scanned = capped_twohop_module._collect_incident_for_middle_range(
        graph,
        0,
        graph.num_nodes,
        edge_chunk_size=4,
    )

    assert isinstance(index.endpoints, np.memmap)
    assert (tmp_path / "incident_index" / "incident_endpoints.npy").exists()
    assert (tmp_path / "incident_index" / "incident_indptr.npy").exists()
    assert _canonical_incident(indexed) == _canonical_incident(scanned)


def test_capped_twohop_memmap_builder_avoids_global_concatenate(tmp_path, monkeypatch):
    graph = generate_synthetic_graph(num_users=12, num_items=5, num_tags=4, seed=820)

    def fail_concatenate(*_args, **_kwargs):
        raise AssertionError("memmap incident builder should not concatenate all triples")

    monkeypatch.setattr(capped_twohop_module.np, "concatenate", fail_concatenate)

    index = capped_twohop_module.CappedTwoHopIncidentIndex.from_graph(
        graph,
        edge_chunk_size=4,
        mmap_dir=tmp_path / "incident_index",
    )

    assert isinstance(index.endpoints, np.memmap)
    assert len(index.indptr) > 1


def test_chunked_capped_twohop_uses_incident_index(monkeypatch):
    graph = generate_synthetic_graph(num_users=16, num_items=5, num_tags=4, seed=828)
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=4,
        middle_degree_cap_policy=8,
        per_middle_pair_cap=6,
    )
    store = ArrayCandidateStore(graph.node_type, K=4)
    z = np.zeros((graph.num_nodes, 4), dtype=np.float32)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)

    def fail_range_scan(*_args, **_kwargs):
        raise AssertionError("range scan should not be used by indexed chunked two-hop")

    monkeypatch.setattr(
        capped_twohop_module,
        "_collect_incident_for_middle_range",
        fail_range_scan,
    )

    stats = capped_twohop_module.generate_capped_twohop_candidates_chunked(
        graph,
        z,
        partition_id,
        config,
        store,
        middle_chunk_size=4,
        edge_chunk_size=5,
    )

    assert stats["middle_nodes_considered"] > 0
    assert stats["max_pairs_emitted_per_middle"] <= 6
    assert store.source_counts().get("capped_twohop", 0) > 0


def test_chunked_bucket_candidates_respect_type_partition_and_caps():
    graph = generate_synthetic_graph(num_users=10, num_items=6, num_tags=3, seed=808)
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=3,
        bucket_pair_cap=4,
        simhash_bits=2,
    )
    z = np.ones((graph.num_nodes, 4), dtype=np.float32)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)
    partition_id[1::2] = 1
    buckets = compute_simhash_buckets(z, graph.node_type, partition_id, bits=2, seed=808)
    store = ArrayCandidateStore(graph.node_type, K=3)

    generate_bucket_candidates_chunked(
        buckets,
        graph.node_type,
        partition_id,
        config,
        store,
        node_chunk_size=4,
    )

    for i, j, *_ in store.to_pairs():
        i = int(i)
        j = int(j)
        assert graph.node_type[i] == graph.node_type[j]
        assert partition_id[i] == partition_id[j]
    assert store.counts().max(initial=0) <= 3
