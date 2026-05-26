from __future__ import annotations

import numpy as np

from tests.gate21_3_helpers import tiny_dblp_graph


def test_random_edge_sampler_reproducible_given_graph_seed() -> None:
    from hesf_coarsen.eval.official.coverage_sampler import sample_random_edge_indices

    first = sample_random_edge_indices(edge_count=20, budget=6, graph_seed=3, relation_id=2)
    second = sample_random_edge_indices(edge_count=20, budget=6, graph_seed=3, relation_id=2)

    assert first.tolist() == second.tolist()


def test_random_edge_sampler_changes_across_graph_seed() -> None:
    from hesf_coarsen.eval.official.coverage_sampler import sample_random_edge_indices

    first = sample_random_edge_indices(edge_count=30, budget=8, graph_seed=3, relation_id=2)
    second = sample_random_edge_indices(edge_count=30, budget=8, graph_seed=4, relation_id=2)

    assert first.tolist() != second.tolist()


def test_pathaware_v2_does_not_use_test_labels() -> None:
    from hesf_coarsen.eval.official.path_aware_edge_scorer_v2 import PathAwareV2Scorer

    graph = tiny_dblp_graph()
    scores, diag = PathAwareV2Scorer().score_relation(
        dataset="DBLP",
        method="H6-struct40-relgrid-best-pathaware-v2-stratified",
        graph_seed=1,
        relation_id=0,
        relation_name="AP",
        graph=graph,
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
        labels=graph.labels,
    )

    assert scores.shape == (3,)
    assert diag["trainval_label_used"] is True
    assert diag["test_label_used"] is False
    assert diag["no_test_label_usage"] is True
    assert "score_component_hub_penalty_mean" in diag


def test_coverage_sampler_preserves_min_relation_edges() -> None:
    from hesf_coarsen.eval.official.coverage_sampler import CoverageSampler

    graph = tiny_dblp_graph()
    rel = graph.relations[2]
    selected, diag = CoverageSampler(hub_cap=3).select(
        src=rel.src,
        dst=rel.dst,
        scores=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        budget=1,
        graph_seed=1,
        relation_id=2,
        min_edges=1,
    )

    assert selected.shape == (1,)
    assert diag["retained_source_node_count"] >= 1
    assert diag["orphan_rescue_count"] >= 0


def test_hub_cap_reduces_max_degree() -> None:
    from hesf_coarsen.eval.official.coverage_sampler import CoverageSampler, max_endpoint_degree

    src = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)
    dst = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    scores = np.arange(6, dtype=np.float64)[::-1]
    uncapped, _ = CoverageSampler(hub_cap=None).select(src=src, dst=dst, scores=scores, budget=4, graph_seed=1, relation_id=0)
    capped, _ = CoverageSampler(hub_cap=2).select(src=src, dst=dst, scores=scores, budget=4, graph_seed=1, relation_id=0)

    assert max_endpoint_degree(src[uncapped], dst[uncapped]) > max_endpoint_degree(src[capped], dst[capped])
