import numpy as np

from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.matching.greedy import run_greedy_matching


def test_greedy_matching_never_merges_cross_type_pairs_by_default():
    graph = generate_synthetic_graph(
        num_users=4,
        num_items=3,
        num_tags=2,
        seed=5,
    )
    first_item = 4
    scored_pairs = np.array(
        [
            [0, first_item, -100.0],
            [0, 1, 0.1],
            [first_item, first_item + 1, 0.2],
        ],
        dtype=np.float64,
    )
    assignment = run_greedy_matching(graph, scored_pairs, DEFAULT_CONFIG)

    assert assignment.assignment[0] != assignment.assignment[first_item]
    assert assignment.assignment[0] == assignment.assignment[1]
    assert assignment.supernode_type[assignment.assignment[0]] == 0


def test_same_partition_is_enforced_by_default():
    graph = generate_synthetic_graph(
        num_users=4,
        num_items=3,
        num_tags=2,
        seed=5,
    )
    scored_pairs = np.array([[0, 1, 0.0]], dtype=np.float64)
    partition_id = np.zeros(graph.num_nodes, dtype=np.int32)
    partition_id[1] = 1

    assignment = run_greedy_matching(
        graph,
        scored_pairs,
        DEFAULT_CONFIG,
        partition_id=partition_id,
    )

    assert assignment.assignment[0] != assignment.assignment[1]
