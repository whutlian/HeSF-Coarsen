import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _graph() -> HeteroGraph:
    node_type = np.zeros(6, dtype=np.int32)
    rel = RelationAdj(
        src=np.array([0, 1, 2, 3, 0, 1], dtype=np.int64),
        dst=np.array([2, 3, 4, 5, 2, 3], dtype=np.int64),
        weight=np.ones(6, dtype=np.float32),
        src_type=0,
        dst_type=0,
        relation_id=0,
    )
    return HeteroGraph(num_nodes=6, node_type=node_type, relations={0: rel}, relation_specs={0: RelationSpec(0, "r0", 0, 0)})


def test_exclusive_timing_keys_are_present_and_non_negative(tmp_path):
    diagnostics = {}
    graph = _graph()
    assignment = Assignment(np.array([0, 0, 1, 1, 2, 2]), np.zeros(3, dtype=np.int32))
    coarsen_graph_chunked(graph, assignment, chunk_size=2, output_dir=tmp_path, reducer="local_prededup_sort", aggregation_diagnostics=diagnostics)

    required = [
        "exclusive_relation_loop_compute_sec",
        "exclusive_assignment_map_sec",
        "exclusive_key_build_sec",
        "exclusive_sort_sec",
        "exclusive_reduce_sec",
        "exclusive_shard_write_sec",
        "exclusive_kway_merge_sec",
        "exclusive_output_write_sec",
        "total_aggregation_sec",
        "timing_inclusive_fields_present",
        "exclusive_timing_sum_sec",
        "exclusive_timing_residual_sec",
    ]
    for key in required:
        assert key in diagnostics
    assert diagnostics["timing_inclusive_fields_present"] is True
    assert diagnostics["exclusive_timing_sum_sec"] <= diagnostics["total_aggregation_sec"] + 1e-6
    assert diagnostics["exclusive_timing_residual_sec"] >= 0.0
    assert diagnostics["aggregation_local_prededup_backend"] is True


def test_a4_local_prededup_matches_sort_reducer(tmp_path):
    graph = _graph()
    assignment = Assignment(np.array([0, 0, 1, 1, 2, 2]), np.zeros(3, dtype=np.int32))
    sort_graph = coarsen_graph_chunked(graph, assignment, chunk_size=2, reducer="sort")
    a4_graph = coarsen_graph_chunked(graph, assignment, chunk_size=2, output_dir=tmp_path, reducer="local_prededup_sort")

    sort_rel = sort_graph.relations[0]
    a4_rel = a4_graph.relations[0]
    assert np.array_equal(sort_rel.src, a4_rel.src)
    assert np.array_equal(sort_rel.dst, a4_rel.dst)
    assert np.allclose(sort_rel.weight, a4_rel.weight)
