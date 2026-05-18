import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.target_preserve import build_target_preserving_assignment
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import nodes_of_type


def test_support_only_assignment_reuses_base_support_clusters_without_target_merges():
    graph = generate_synthetic_graph(num_users=4, num_items=6, num_tags=4, seed=18)
    base = Assignment(
        assignment=np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5], dtype=np.int64),
        supernode_type=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
    )

    hybrid = build_target_preserving_assignment(graph, base, target_node_type=0)

    item_nodes = nodes_of_type(graph, 1)
    tag_nodes = nodes_of_type(graph, 2)
    assert len(np.unique(hybrid.assignment[item_nodes])) < len(item_nodes)
    assert len(np.unique(hybrid.assignment[tag_nodes])) < len(tag_nodes)
    assert hybrid.num_supernodes < graph.num_nodes
    assert set(hybrid.supernode_type.tolist()) == {0, 1, 2}
