import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.scoring.merge_cost import (
    prepare_pair_scoring_context,
    score_pair_block_with_terms,
    score_candidate_pairs,
    score_pair_block,
)


def _same_type_graph_with_degrees() -> HeteroGraph:
    relations = {
        0: RelationAdj(
            src=np.array([0, 0, 1, 2], dtype=np.int64),
            dst=np.array([1, 2, 2, 3], dtype=np.int64),
            weight=np.ones(4, dtype=np.float32),
            src_type=0,
            dst_type=0,
            relation_id=0,
        )
    }
    return HeteroGraph(
        num_nodes=4,
        node_type=np.zeros(4, dtype=np.int32),
        relations=relations,
        relation_specs={0: RelationSpec(0, "node_to_node", 0, 0)},
    )


def _score_single_pair(graph: HeteroGraph, pair: tuple[int, int], config: dict) -> float:
    z = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    profiles = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    conv = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    pairs = np.array([[pair[0], pair[1], 0.0]], dtype=np.float64)
    scored = score_candidate_pairs(graph, pairs, z, profiles, conv, None, config)
    return float(scored[0, 2])


def test_score_pair_block_matches_full_scoring_and_filters_cross_type():
    graph = HeteroGraph(
        num_nodes=3,
        node_type=np.array([0, 0, 1], dtype=np.int32),
        relations={},
    )
    pairs = np.array(
        [
            [0, 1, 100.0],
            [0, 2, 0.0],
            [1, 0, 5.0],
        ],
        dtype=np.float64,
    )
    z = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    profiles = np.zeros((3, 2), dtype=np.float32)
    conv = np.zeros((3, 2), dtype=np.float32)
    config = {
        "scoring": {
            "lambda_spec": 1.0,
            "lambda_rel": 0.0,
            "lambda_feat": 0.0,
            "lambda_conv": 0.0,
            "lambda_boundary": 0.0,
            "spec_volume_weighting": False,
        },
        "acceleration": {"dense_backend": "numpy", "scoring_batch_size": 1},
    }
    context = prepare_pair_scoring_context(graph, z, profiles, conv, None, config)

    streamed = np.vstack(
        [
            score_pair_block(context, pairs[:1]),
            score_pair_block(context, pairs[1:]),
        ]
    )
    full = score_candidate_pairs(graph, pairs, z, profiles, conv, None, config)

    assert np.allclose(streamed, full)
    assert streamed.shape == (2, 3)


def test_score_pair_block_with_terms_reports_unweighted_components():
    graph = _same_type_graph_with_degrees()
    pairs = np.array([[0, 3, 0.0]], dtype=np.float64)
    z = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    profiles = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    conv = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    config = {
        "scoring": {
            "lambda_spec": 2.0,
            "lambda_rel": 3.0,
            "lambda_feat": 0.0,
            "lambda_conv": 5.0,
            "lambda_boundary": 0.0,
            "spec_volume_weighting": True,
            "relation_profile_distance": "jsd",
        },
        "acceleration": {"dense_backend": "numpy"},
    }
    context = prepare_pair_scoring_context(graph, z, profiles, conv, None, config)

    scored, terms = score_pair_block_with_terms(context, pairs)

    assert set(terms) == {"spec", "rel", "feat", "conv", "boundary"}
    assert np.isclose(terms["spec"][0], (2.0 * 1.0 / 3.0) * 25.0)
    assert np.isclose(terms["rel"][0], np.log(2.0), atol=1e-6)
    assert np.isclose(terms["conv"][0], 1.0)
    expected = 2.0 * terms["spec"][0] + 3.0 * terms["rel"][0] + 5.0 * terms["conv"][0]
    assert np.isclose(scored[0, 2], expected)


def test_score_pair_block_can_normalize_terms_before_weighting():
    graph = HeteroGraph(
        num_nodes=4,
        node_type=np.zeros(4, dtype=np.int32),
        relations={},
    )
    pairs = np.array([[0, 1, 0.0], [2, 3, 0.0]], dtype=np.float64)
    z = np.array(
        [
            [0.0],
            [1.0],
            [0.0],
            [10.0],
        ],
        dtype=np.float32,
    )
    profiles = np.zeros((4, 1), dtype=np.float32)
    conv = np.zeros((4, 1), dtype=np.float32)
    config = {
        "scoring": {
            "lambda_spec": 1.0,
            "lambda_rel": 0.0,
            "lambda_feat": 0.0,
            "lambda_conv": 0.0,
            "lambda_boundary": 0.0,
            "spec_volume_weighting": False,
            "normalization": "p95",
        },
        "acceleration": {"dense_backend": "numpy"},
    }
    context = prepare_pair_scoring_context(graph, z, profiles, conv, None, config)

    scored, terms = score_pair_block_with_terms(context, pairs)

    assert np.allclose(terms["spec"], np.array([1.0, 100.0], dtype=np.float32))
    assert scored[1, 2] <= 1.1
    assert scored[0, 2] < 0.02


def test_spec_term_uses_local_variation_volume_factor():
    graph = _same_type_graph_with_degrees()
    pairs = np.array([[0, 3, 0.0]], dtype=np.float64)
    z = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    profiles = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    conv = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    config = {
        "scoring": {
            "lambda_spec": 1.0,
            "lambda_rel": 0.0,
            "lambda_feat": 0.0,
            "lambda_conv": 0.0,
            "lambda_boundary": 0.0,
            "spec_volume_weighting": True,
        },
        "acceleration": {"dense_backend": "numpy"},
    }

    scored = score_candidate_pairs(graph, pairs, z, profiles, conv, None, config)

    # vol(0)=2, vol(3)=1, squared distance=25.
    assert np.isclose(scored[0, 2], (2.0 * 1.0 / 3.0) * 25.0)


def test_relation_distribution_term_can_use_jsd():
    graph = HeteroGraph(
        num_nodes=2,
        node_type=np.zeros(2, dtype=np.int32),
        relations={},
    )
    pairs = np.array([[0, 1, 0.0]], dtype=np.float64)
    z = np.zeros((2, 1), dtype=np.float32)
    profiles = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    conv = np.zeros((2, 1), dtype=np.float32)
    config = {
        "scoring": {
            "lambda_spec": 0.0,
            "lambda_rel": 1.0,
            "lambda_feat": 0.0,
            "lambda_conv": 0.0,
            "lambda_boundary": 0.0,
            "relation_profile_distance": "jsd",
        },
        "acceleration": {"dense_backend": "numpy"},
    }

    scored = score_candidate_pairs(graph, pairs, z, profiles, conv, None, config)

    assert np.isclose(scored[0, 2], np.log(2.0), atol=1e-6)


def test_boundary_penalty_uses_node_level_terminal_risk_within_partition():
    graph = HeteroGraph(
        num_nodes=3,
        node_type=np.zeros(3, dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0], dtype=np.int64),
                dst=np.array([2], dtype=np.int64),
                weight=np.ones(1, dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "node_to_node", 0, 0)},
    )
    config = {
        "scoring": {
            "lambda_spec": 0.0,
            "lambda_rel": 0.0,
            "lambda_feat": 0.0,
            "lambda_conv": 0.0,
            "lambda_boundary": 1.0,
            "boundary_mode": "node_risk",
            "boundary_hub_gamma": 0.0,
            "boundary_terminal_gamma": 2.0,
            "boundary_terminal_degree": 1.0,
        },
        "acceleration": {"dense_backend": "numpy"},
    }

    cost = _score_single_pair(
        graph,
        (0, 1),
        config,
    )

    assert np.isclose(cost, 2.0)
