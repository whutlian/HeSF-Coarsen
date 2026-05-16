import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def test_chunked_aggregation_emits_stage_timing_diagnostics(tmp_path):
    graph = HeteroGraph(
        num_nodes=4,
        node_type=np.array([0, 0, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 1, 0], dtype=np.int64),
                dst=np.array([2, 2, 3], dtype=np.int64),
                weight=np.ones(3, dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "r0", 0, 1)},
    )
    assignment = Assignment(
        assignment=np.array([0, 0, 1, 2], dtype=np.int64),
        supernode_type=np.array([0, 1, 1], dtype=np.int32),
    )
    diagnostics: dict = {}

    coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=2,
        output_dir=tmp_path,
        reducer="sort",
        aggregation_diagnostics=diagnostics,
    )

    for key in (
        "aggregation_total_sec",
        "aggregation_relation_loop_sec",
        "aggregation_assignment_map_sec",
        "aggregation_key_build_sec",
        "aggregation_sort_sec",
        "aggregation_reduce_sec",
        "aggregation_dedup_sec",
        "aggregation_shard_write_sec",
        "aggregation_kway_merge_sec",
        "aggregation_output_write_sec",
    ):
        assert key in diagnostics
        assert diagnostics[key] >= 0.0
    assert diagnostics["aggregation_reducer"] == "sort"
    assert diagnostics["aggregation_python_dict_used"] is False
    assert diagnostics["aggregation_by_relation"][0]["coarse_edges_after_dedup"] == 2
