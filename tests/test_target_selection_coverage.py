import numpy as np

from hesf_coarsen.accuracy.target_selection import support_coverage_sets
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import nodes_of_type


def test_support_coverage_sets_include_typed_onehop_support():
    graph = generate_synthetic_graph(num_users=5, num_items=4, num_tags=3, seed=21)
    target_nodes = nodes_of_type(graph, 0)

    coverage = support_coverage_sets(graph, target_node_type=0, target_nodes=target_nodes)

    assert set(coverage) == set(int(node) for node in target_nodes)
    assert any(len(values) > 0 for values in coverage.values())
    for values in coverage.values():
        assert all(isinstance(item, tuple) and len(item) == 2 for item in values)
