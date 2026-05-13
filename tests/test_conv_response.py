import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.scoring.conv_response import compute_conv_response_sketch
from hesf_coarsen.sketch.operators import apply_fused_operator


def _directed_chain_graph() -> HeteroGraph:
    node_type = np.zeros(3, dtype=np.int32)
    relations = {
        0: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([1, 2], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=0,
            dst_type=0,
            relation_id=0,
        )
    }
    specs = {0: RelationSpec(0, "node__points_to__node", 0, 0)}
    return HeteroGraph(3, node_type, relations, specs)


def test_conv_response_defaults_to_explicit_fused_operator():
    graph = _directed_chain_graph()
    H = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]], dtype=np.float32)
    relation_weights = {0: 2.0}

    actual = compute_conv_response_sketch(graph, H, relation_weights)
    expected = apply_fused_operator(graph, H, relation_weights)

    assert np.allclose(actual, expected)


def test_conv_response_supports_lazy_smoothing_baseline():
    graph = _directed_chain_graph()
    H = np.array([[1.0], [2.0], [4.0]], dtype=np.float32)
    relation_weights = {0: 1.0}

    actual = compute_conv_response_sketch(
        graph,
        H,
        relation_weights,
        operator="lazy_smoothing",
    )
    expected = apply_fused_smoothing(graph, H, relation_weights)
    fused = compute_conv_response_sketch(
        graph,
        H,
        relation_weights,
        operator="fused_operator",
    )

    assert np.allclose(actual, expected)
    assert not np.allclose(actual, fused)
