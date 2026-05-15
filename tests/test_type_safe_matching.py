import inspect

import numpy as np

from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.matching.greedy import (
    finalize_mutual_best,
    initialize_mutual_best_state,
    run_greedy_cluster_matching,
    run_greedy_matching,
    run_matching,
    run_mutual_best_matching,
    mutual_best_update_block,
)


def same_type_graph(num_nodes):
    return HeteroGraph(
        num_nodes=num_nodes,
        node_type=np.zeros(num_nodes, dtype=np.int32),
        relations={},
    )


def paired(assignment, i, j):
    return int(assignment.assignment[i]) == int(assignment.assignment[j])


def test_default_matching_method_is_greedy_cluster():
    assert DEFAULT_CONFIG["coarsening"]["matching_method"] == "greedy_cluster"


def test_matching_module_avoids_python_tolist_conversion():
    assert ".tolist(" not in inspect.getsource(run_greedy_matching)
    assert ".tolist()" not in inspect.getsource(run_greedy_matching)


def test_mutual_best_matching_only_merges_reciprocal_best_pairs():
    graph = same_type_graph(4)
    scored_pairs = np.array(
        [
            [0, 1, 0.5],
            [1, 2, 0.1],
            [2, 3, 0.2],
        ],
        dtype=np.float64,
    )

    assignment = run_mutual_best_matching(
        graph,
        scored_pairs,
        {"coarsening": {"same_type_only": True, "same_partition_only": False}},
    )

    assert paired(assignment, 1, 2)
    assert not paired(assignment, 0, 1)
    assert not paired(assignment, 2, 3)


def test_run_matching_uses_greedy_cluster_by_default():
    graph = same_type_graph(4)
    scored_pairs = np.array(
        [
            [0, 1, 0.5],
            [1, 2, 0.1],
            [2, 3, 0.2],
        ],
        dtype=np.float64,
    )

    assignment = run_matching(graph, scored_pairs, DEFAULT_CONFIG)

    assert max(assignment.cluster_sizes()) <= DEFAULT_CONFIG["coarsening"]["max_cluster_size"]
    assert paired(assignment, 1, 2)
    assert paired(assignment, 2, 3)


def test_mutual_best_matching_caps_matches_by_lowest_cost():
    graph = same_type_graph(6)
    scored_pairs = np.array(
        [
            [0, 1, 5.0],
            [2, 3, 1.0],
            [4, 5, 3.0],
        ],
        dtype=np.float64,
    )

    assignment = run_mutual_best_matching(
        graph,
        scored_pairs,
        {
            "coarsening": {
                "same_type_only": True,
                "same_partition_only": False,
                "max_matched_pairs": 2,
            }
        },
    )

    assert not paired(assignment, 0, 1)
    assert paired(assignment, 2, 3)
    assert paired(assignment, 4, 5)


def test_mutual_best_matching_enforces_same_partition_by_default():
    graph = same_type_graph(2)
    scored_pairs = np.array([[0, 1, 0.0]], dtype=np.float64)
    partition_id = np.array([0, 1], dtype=np.int32)

    assignment = run_mutual_best_matching(
        graph,
        scored_pairs,
        DEFAULT_CONFIG,
        partition_id=partition_id,
    )

    assert not paired(assignment, 0, 1)


def test_greedy_cluster_matching_respects_max_cluster_size():
    graph = same_type_graph(5)
    scored_pairs = np.array(
        [
            [0, 1, 0.0],
            [1, 2, 0.1],
            [2, 3, 0.2],
            [3, 4, 0.3],
        ],
        dtype=np.float64,
    )

    assignment = run_greedy_cluster_matching(
        graph,
        scored_pairs,
        {
            "coarsening": {
                "same_type_only": True,
                "same_partition_only": False,
                "matching_method": "greedy_cluster",
                "max_cluster_size": 3,
            }
        },
    )

    sizes = sorted(assignment.cluster_sizes().tolist())
    assert sizes == [2, 3]


def test_greedy_cluster_matching_never_merges_cross_type_pairs_by_default():
    graph = HeteroGraph(
        num_nodes=4,
        node_type=np.array([0, 1, 0, 1], dtype=np.int32),
        relations={},
    )
    scored_pairs = np.array(
        [
            [0, 1, -10.0],
            [2, 3, -9.0],
            [0, 2, 1.0],
        ],
        dtype=np.float64,
    )

    assignment = run_greedy_cluster_matching(graph, scored_pairs, DEFAULT_CONFIG)

    assert assignment.assignment[0] != assignment.assignment[1]
    assert assignment.assignment[2] != assignment.assignment[3]
    assert assignment.assignment[0] == assignment.assignment[2]


def test_greedy_cluster_matching_respects_same_partition_by_default():
    graph = same_type_graph(4)
    scored_pairs = np.array(
        [
            [0, 1, -10.0],
            [0, 2, 0.1],
            [1, 3, 0.2],
        ],
        dtype=np.float64,
    )
    partition_id = np.array([0, 1, 0, 1], dtype=np.int32)

    assignment = run_greedy_cluster_matching(
        graph,
        scored_pairs,
        DEFAULT_CONFIG,
        partition_id=partition_id,
    )

    assert assignment.assignment[0] != assignment.assignment[1]
    assert assignment.assignment[0] == assignment.assignment[2]
    assert assignment.assignment[1] == assignment.assignment[3]


def test_streaming_mutual_best_matches_batch_matching_across_blocks():
    graph = same_type_graph(6)
    scored_pairs = np.array(
        [
            [0, 1, 0.5],
            [1, 2, 0.1],
            [2, 3, 0.2],
            [4, 5, 0.3],
            [0, 5, 0.3],
        ],
        dtype=np.float64,
    )
    config = {
        "coarsening": {
            "same_type_only": True,
            "same_partition_only": False,
            "max_matched_pairs": 2,
        }
    }

    state = initialize_mutual_best_state(graph)
    mutual_best_update_block(graph, state, scored_pairs[:2], config)
    mutual_best_update_block(graph, state, scored_pairs[2:], config)
    streaming = finalize_mutual_best(graph, state, config)
    batch = run_mutual_best_matching(graph, scored_pairs, config)

    assert np.array_equal(streaming.assignment, batch.assignment)
    assert np.array_equal(streaming.supernode_type, batch.supernode_type)


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
