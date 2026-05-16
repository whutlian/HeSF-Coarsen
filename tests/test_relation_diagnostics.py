import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.relation_diagnostics import (
    coarse_edge_collapse_by_relation,
    relation_distribution_drift,
)
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1], dtype=np.int32)
    return HeteroGraph(
        num_nodes=4,
        node_type=node_type,
        relations={
            0: RelationAdj(
                src=np.array([0, 1], dtype=np.int64),
                dst=np.array([2, 2], dtype=np.int64),
                weight=np.array([1.0, 3.0], dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            ),
            1: RelationAdj(
                src=np.array([2, 3], dtype=np.int64),
                dst=np.array([0, 0], dtype=np.int64),
                weight=np.array([2.0, 4.0], dtype=np.float32),
                src_type=1,
                dst_type=0,
                relation_id=1,
            ),
        },
        relation_specs={
            0: RelationSpec(0, "u_to_v", 0, 1),
            1: RelationSpec(1, "v_to_u", 1, 0),
        },
    )


def test_relation_mass_drift_zero_for_identity_assignment():
    graph = _graph()
    assignment = Assignment(
        assignment=np.arange(graph.num_nodes, dtype=np.int64),
        supernode_type=graph.node_type.copy(),
    )
    coarse = coarsen_graph_chunked(graph, assignment)

    drift = relation_distribution_drift(graph, coarse)

    assert drift["relation_mass_l1_drift"] == 0.0
    assert drift["relation_mass_js_drift"] == 0.0


def test_relation_mass_drift_positive_for_relation_collapsing_counts():
    graph = _graph()
    coarse = HeteroGraph(
        num_nodes=4,
        node_type=graph.node_type.copy(),
        relations={
            0: graph.relations[0],
            1: RelationAdj(
                src=np.empty(0, dtype=np.int64),
                dst=np.empty(0, dtype=np.int64),
                weight=np.empty(0, dtype=np.float32),
                src_type=1,
                dst_type=0,
                relation_id=1,
            ),
        },
        relation_specs=graph.relation_specs,
    )

    drift = relation_distribution_drift(graph, coarse)

    assert drift["relation_mass_l1_drift"] > 0.0
    assert drift["relation_mass_js_drift"] > 0.0


def test_relation_edge_collapse_reports_weight_preservation():
    graph = _graph()
    assignment = Assignment(
        assignment=np.array([0, 0, 1, 2], dtype=np.int64),
        supernode_type=np.array([0, 1, 1], dtype=np.int32),
    )
    coarse = coarsen_graph_chunked(graph, assignment)

    rows = coarse_edge_collapse_by_relation(graph, coarse, assignment)
    rel0 = next(row for row in rows if row["relation_id"] == 0)

    assert rel0["original_edges"] == 2
    assert rel0["coarse_edges_before_dedup"] == 2
    assert rel0["coarse_edges_after_dedup"] == 1
    assert rel0["coarse_edge_uniqueness_ratio"] == 0.5
    assert rel0["duplicate_collapse_ratio"] == 0.5
    assert rel0["edge_weight_abs_error"] == 0.0
