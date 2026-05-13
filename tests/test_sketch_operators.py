import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.sketch.operators import (
    apply_fused_laplacian,
    apply_fused_operator,
    apply_metapath_operator,
    apply_relation_operator,
)


def _reverse_pair_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1], dtype=np.int32)
    relations = {
        0: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([2, 3], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([2, 3], dtype=np.int64),
            dst=np.array([0, 1], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
        2: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([1, 0], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=0,
            dst_type=0,
            relation_id=2,
        ),
    }
    specs = {
        0: RelationSpec(0, "author__writes__paper", 0, 1),
        1: RelationSpec(1, "paper__written_by__author", 1, 0),
        2: RelationSpec(2, "author__coauthor__author", 0, 0),
    }
    return HeteroGraph(4, node_type, relations, specs)


def test_drop_detected_reverse_relation_policy_drops_reverse_and_renormalizes():
    graph = _reverse_pair_graph()
    H = np.arange(graph.num_nodes * 2, dtype=np.float32).reshape(graph.num_nodes, 2) / 10.0
    relation_weights = {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}

    actual = apply_fused_operator(
        graph,
        H,
        relation_weights,
        reverse_relation_policy="drop_detected_reverse_for_spectral_operator",
    )
    expected = 0.5 * apply_relation_operator(graph, H, 0, direction="symmetric")
    expected += 0.5 * apply_relation_operator(graph, H, 2, direction="symmetric")
    include_all = apply_fused_operator(
        graph,
        H,
        relation_weights,
        reverse_relation_policy="include_all",
    )

    assert np.allclose(actual, expected)
    assert not np.allclose(actual, include_all)


def test_fused_operator_includes_weighted_metapath_operator_without_materializing_product():
    graph = _reverse_pair_graph()
    H = np.arange(graph.num_nodes * 2, dtype=np.float32).reshape(graph.num_nodes, 2) / 10.0
    path = {
        "name": "author_paper_author",
        "start_type": 0,
        "end_type": 0,
        "steps": [
            {"relation_id": 0, "direction": "forward"},
            {"relation_id": 0, "direction": "backward"},
        ],
    }

    metapath = apply_metapath_operator(graph, H, path)
    expected = 0.4 * apply_relation_operator(graph, H, 2, direction="symmetric")
    expected += 0.6 * metapath
    actual = apply_fused_operator(
        graph,
        H,
        relation_weights={2: 0.4},
        metapath_weights=[(path, 0.6)],
    )
    laplacian = apply_fused_laplacian(
        graph,
        H,
        relation_weights={2: 0.4},
        metapath_weights=[(path, 0.6)],
    )

    assert np.allclose(actual, expected)
    assert np.allclose(laplacian, H - expected)
    assert np.allclose(metapath[graph.node_type == 1], 0.0)
