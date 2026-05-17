import numpy as np
import pytest

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph, coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _relation_table(graph, relation_id):
    rel = graph.relations[relation_id]
    return sorted((int(s), int(d), float(w)) for s, d, w in zip(rel.src, rel.dst, rel.weight))


def test_packed_key_backend_matches_sort_on_toy_graph(tmp_path):
    graph = HeteroGraph(
        num_nodes=6,
        node_type=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 2, 0, 1], dtype=np.int64),
                dst=np.array([3, 3, 4, 4, 5], dtype=np.int64),
                weight=np.array([1.0, 2.0, 3.0, 0.5, 0.25], dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "u_to_v", 0, 1)},
    )
    assignment = Assignment(
        np.array([0, 0, 1, 2, 2, 3], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int32),
    )

    expected = coarsen_graph(graph, assignment)
    actual = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=2,
        output_dir=tmp_path / "packed",
        reducer="packed_key_sort",
    )

    assert _relation_table(actual, 0) == _relation_table(expected, 0)


def test_packed_key_backend_handles_rectangular_type_cluster_spaces(tmp_path):
    graph = HeteroGraph(
        num_nodes=8,
        node_type=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
                dst=np.array([4, 4, 5, 5, 6, 6, 7, 7], dtype=np.int64),
                weight=np.ones(8, dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "left_to_right", 0, 1)},
    )
    assignment = Assignment(
        np.array([0, 1, 1, 2, 3, 3, 4, 4], dtype=np.int64),
        np.array([0, 0, 0, 1, 1], dtype=np.int32),
    )

    actual = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=3,
        output_dir=tmp_path / "rect",
        reducer="packed_key_sort",
    )

    rel = actual.relations[0]
    assert set(rel.src.tolist()) <= {0, 1, 2}
    assert set(rel.dst.tolist()) <= {3, 4}
    assert np.isclose(float(rel.weight.sum()), 8.0)


def test_packed_key_backend_rejects_int64_overflow():
    from hesf_coarsen.coarsen.aggregate_edges import _encode_relation_local_packed_keys

    max_int64 = np.iinfo(np.int64).max
    with pytest.raises(OverflowError, match="overflow"):
        _encode_relation_local_packed_keys(
            np.array([max_int64 // 2 + 1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.int64(max_int64 // 2 + 1),
        )


def test_packed_key_backend_emits_backend_specific_timing(tmp_path):
    graph = HeteroGraph(
        num_nodes=3,
        node_type=np.array([0, 0, 0], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 1], dtype=np.int64),
                dst=np.array([1, 2, 2], dtype=np.int64),
                weight=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "r", 0, 0)},
    )
    assignment = Assignment(np.array([0, 0, 1], dtype=np.int64), np.array([0, 0], dtype=np.int32))
    diagnostics = {}

    coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=1,
        output_dir=tmp_path / "diag",
        reducer="packed_key_sort",
        aggregation_diagnostics=diagnostics,
    )

    assert diagnostics["aggregation_reducer"] == "packed_key_sort"
    assert diagnostics["aggregation_packed_key_backend"] is True
    assert diagnostics["aggregation_by_relation"][0]["edge_weight_abs_error"] == 0.0
