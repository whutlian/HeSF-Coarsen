import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.target_preserve import (
    build_target_preserving_assignment,
    target_preservation_report,
)
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import nodes_of_type, validate_schema


def test_target_nodes_are_identity_singletons_after_support_coarsening():
    graph = generate_synthetic_graph(num_users=6, num_items=4, num_tags=2, seed=17)
    base = Assignment(
        assignment=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64),
        supernode_type=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
    )

    hybrid = build_target_preserving_assignment(graph, base, target_node_type=0)
    report = target_preservation_report(graph, hybrid, target_node_type=0)

    assert report["target_identity"] is True
    assert report["target_cluster_size_max"] == 1
    assert report["target_original_nodes"] == 6
    assert report["target_coarse_nodes"] == 6
    assert report["support_coarsened"] is True

    target_nodes = nodes_of_type(graph, 0)
    assert len(np.unique(hybrid.assignment[target_nodes])) == len(target_nodes)
    assert np.all(hybrid.supernode_type[hybrid.assignment[target_nodes]] == 0)

    coarse = coarsen_graph(graph, hybrid)
    validate_schema(coarse)
    assert coarse.num_nodes == hybrid.num_supernodes
    assert len(nodes_of_type(coarse, 0)) == len(target_nodes)
