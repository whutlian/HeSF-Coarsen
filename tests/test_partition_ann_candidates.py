import numpy as np

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.candidates.partition_ann import generate_partition_ann_candidates
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_partition_ann_candidates_respect_type_partition_and_budget():
    graph = generate_synthetic_graph(num_users=12, num_items=8, num_tags=4, seed=909)
    z = np.arange(graph.num_nodes * 4, dtype=np.float32).reshape(graph.num_nodes, 4) / 10.0
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)
    partition_id[1::2] = 1
    config = dict(DEFAULT_CONFIG)
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=3,
        ann_num_projections=3,
        ann_window_size=2,
        ann_budget_K=2,
    )
    store = ArrayCandidateStore(graph.node_type, K=3)

    stats = generate_partition_ann_candidates(graph, z, partition_id, config, store)

    assert stats["pairs_considered"] > 0
    assert store.source_counts().get("partition_ann", 0) > 0
    assert store.counts().max(initial=0) <= 3
    for i, j, *_ in store.to_pairs():
        i = int(i)
        j = int(j)
        assert graph.node_type[i] == graph.node_type[j]
        assert partition_id[i] == partition_id[j]


def test_partition_ann_candidates_are_seed_deterministic():
    graph = generate_synthetic_graph(num_users=10, num_items=7, num_tags=4, seed=919)
    z = np.random.default_rng(919).normal(size=(graph.num_nodes, 6)).astype(np.float32)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)
    config = dict(DEFAULT_CONFIG)
    config["seed"] = 919
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=4,
        ann_num_projections=4,
        ann_window_size=3,
        ann_budget_K=2,
    )

    first = ArrayCandidateStore(graph.node_type, K=4)
    second = ArrayCandidateStore(graph.node_type, K=4)
    generate_partition_ann_candidates(graph, z, partition_id, config, first)
    generate_partition_ann_candidates(graph, z, partition_id, config, second)

    assert np.array_equal(first.counts(), second.counts())
    assert np.allclose(first.to_pairs(), second.to_pairs())
