from hesf_coarsen.accuracy.type_budgets import compute_type_budget_report
from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.target_preserve import build_target_preserving_assignment
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_type_budget_report_separates_target_and_support_ratios():
    graph = generate_synthetic_graph(num_users=6, num_items=4, num_tags=2, seed=23)
    base = Assignment(
        assignment=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        supernode_type=[0, 0, 0, 1, 1, 2],
    )
    assignment = build_target_preserving_assignment(graph, base, target_node_type=0)
    coarse = coarsen_graph(graph, assignment)

    report = compute_type_budget_report(graph, coarse, target_node_type=0)

    assert report["target_type"] == 0
    assert report["per_type"]["0"]["ratio"] == 1.0
    assert report["global_ratio"] == coarse.num_nodes / graph.num_nodes
    assert any(value["ratio"] < 1.0 for key, value in report["per_type"].items() if key != "0")
