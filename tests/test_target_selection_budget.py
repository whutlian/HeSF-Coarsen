import numpy as np

from hesf_coarsen.accuracy.target_selection import select_target_anchors
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import nodes_of_type


def test_target_selection_keeps_train_nodes_and_is_reproducible():
    graph = generate_synthetic_graph(num_users=12, num_items=8, num_tags=3, seed=19)
    target_nodes = nodes_of_type(graph, 0)
    train_nodes = target_nodes[:4]

    first = select_target_anchors(
        graph,
        target_node_type=0,
        train_nodes=train_nodes,
        budget=7,
        seed=99,
    )
    second = select_target_anchors(
        graph,
        target_node_type=0,
        train_nodes=train_nodes,
        budget=7,
        seed=99,
    )

    assert set(train_nodes.tolist()).issubset(set(first.selected_nodes.tolist()))
    assert first.selected_nodes.tolist() == second.selected_nodes.tolist()
    assert len(first.selected_nodes) == 7
    assert first.diagnostics["mandatory_train_count"] == 4
    assert "score_terms" in first.diagnostics


def test_target_selection_never_drops_mandatory_nodes_when_budget_is_small():
    graph = generate_synthetic_graph(num_users=8, num_items=5, num_tags=2, seed=20)
    train_nodes = nodes_of_type(graph, 0)[:5]

    result = select_target_anchors(graph, target_node_type=0, train_nodes=train_nodes, budget=2, seed=7)

    assert result.selected_nodes.tolist() == train_nodes.tolist()
    assert result.diagnostics["budget_requested"] == 2
    assert result.diagnostics["budget_effective"] == 5
