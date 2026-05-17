import inspect

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _toy_graph() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=5,
        node_type=np.array([0, 1, 0, 1, 0], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 2, 4], dtype=np.int64),
                dst=np.array([1, 3, 1], dtype=np.int64),
                weight=np.array([1.0, 2.0, 1.5], dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            ),
            1: RelationAdj(
                src=np.array([1, 3], dtype=np.int64),
                dst=np.array([2, 4], dtype=np.int64),
                weight=np.array([1.0, 1.0], dtype=np.float32),
                src_type=1,
                dst_type=0,
                relation_id=1,
            ),
        },
        relation_specs={
            0: RelationSpec(0, "paper__written_by__author", 0, 1),
            1: RelationSpec(1, "author__writes__paper", 1, 0),
        },
    )


def _identity_assignment(graph: HeteroGraph) -> Assignment:
    return Assignment(np.arange(graph.num_nodes, dtype=np.int64), graph.node_type.copy())


def test_identity_assignment_preserves_typed_paths_and_counts():
    from hesf_coarsen.eval.metapath_retention import (
        evaluate_path_retention,
        sample_typed_paths,
        summarize_metapath_retention,
    )

    graph = _toy_graph()
    assignment = _identity_assignment(graph)
    samples = sample_typed_paths(
        graph,
        [{"name": "p_a_p", "steps": [0, 1], "start_type": 0, "end_type": 0}],
        sample_seed=7,
        max_samples_per_schema=8,
    )

    rows = evaluate_path_retention(samples, assignment, graph, original_graph=graph)
    assert rows
    assert all(row["typed_exact_step_survival_rate"] == 1.0 for row in rows)
    assert all(row["all_steps_survived"] for row in rows)
    assert all(row["endpoint_pair_collapse_rate"] == 0.0 for row in rows)
    assert all(abs(row["log_path_count_error"]) < 1.0e-9 for row in rows)

    summary = summarize_metapath_retention(rows)
    assert summary
    assert summary[0]["schema_path_typed_survival_mean"] == 1.0


def test_removing_required_relation_lowers_typed_survival():
    from hesf_coarsen.eval.metapath_retention import evaluate_path_retention, sample_typed_paths

    graph = _toy_graph()
    assignment = _identity_assignment(graph)
    samples = sample_typed_paths(
        graph,
        [{"name": "p_a_p", "steps": [0, 1], "start_type": 0, "end_type": 0}],
        sample_seed=11,
        max_samples_per_schema=6,
    )
    coarse = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations={0: graph.relations[0]},
        relation_specs={0: graph.relation_specs[0]},
    )

    rows = evaluate_path_retention(samples, assignment, coarse, original_graph=graph)
    assert min(row["typed_exact_step_survival_rate"] for row in rows) < 1.0
    assert any(not row["all_steps_survived"] for row in rows)


def test_flatten_relation_control_keeps_untyped_while_losing_typed():
    from hesf_coarsen.eval.metapath_retention import evaluate_path_retention, sample_typed_paths

    graph = _toy_graph()
    assignment = _identity_assignment(graph)
    samples = sample_typed_paths(
        graph,
        [{"name": "p_a_p", "steps": [0, 1], "start_type": 0, "end_type": 0}],
        sample_seed=17,
        max_samples_per_schema=5,
    )
    flattened = HeteroGraph(
        num_nodes=graph.num_nodes,
        node_type=graph.node_type.copy(),
        relations={
            99: RelationAdj(
                src=np.concatenate([graph.relations[0].src, graph.relations[1].src]),
                dst=np.concatenate([graph.relations[0].dst, graph.relations[1].dst]),
                weight=np.ones(graph.relations[0].num_edges + graph.relations[1].num_edges, dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=99,
            )
        },
        relation_specs={99: RelationSpec(99, "flattened", 0, 0)},
    )

    rows = evaluate_path_retention(samples, assignment, flattened, original_graph=graph)
    assert np.mean([row["typed_exact_step_survival_rate"] for row in rows]) == 0.0
    assert np.mean([row["untyped_step_survival_rate"] for row in rows]) == 1.0


def test_same_samples_are_method_sensitive_for_different_assignments():
    from hesf_coarsen.eval.metapath_retention import evaluate_path_retention, sample_typed_paths

    graph = _toy_graph()
    samples = sample_typed_paths(
        graph,
        [{"name": "p_a_p", "steps": [0, 1], "start_type": 0, "end_type": 0}],
        sample_seed=19,
        max_samples_per_schema=8,
    )
    identity = _identity_assignment(graph)
    merged = Assignment(
        np.array([0, 1, 0, 2, 3], dtype=np.int64),
        np.array([0, 1, 1, 0], dtype=np.int32),
    )
    coarse_identity = coarsen_graph(graph, identity)
    coarse_merged = coarsen_graph(graph, merged)

    identity_rows = evaluate_path_retention(samples, identity, coarse_identity, original_graph=graph)
    merged_rows = evaluate_path_retention(samples, merged, coarse_merged, original_graph=graph)

    assert [row["sample_id"] for row in identity_rows] == [row["sample_id"] for row in merged_rows]
    assert [row["cluster_path"] for row in identity_rows] != [row["cluster_path"] for row in merged_rows]


def test_path_count_expansion_respects_caps_and_marks_capped():
    from hesf_coarsen.eval.metapath_retention import evaluate_path_retention

    graph = HeteroGraph(
        num_nodes=8,
        node_type=np.zeros(8, dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.array([0, 0, 0, 1, 2, 3, 1, 2, 3], dtype=np.int64),
                dst=np.array([1, 2, 3, 7, 7, 7, 4, 5, 6], dtype=np.int64),
                weight=np.ones(9, dtype=np.float32),
                src_type=0,
                dst_type=0,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "r", 0, 0)},
    )
    sample = {
        "sample_id": "cap",
        "dataset": "toy",
        "seed": 1,
        "schema_path": "r_r",
        "relation_sequence": "0,0",
        "node_path": "0,1,7",
        "path_length": 2,
    }

    rows = evaluate_path_retention(
        [sample],
        _identity_assignment(graph),
        graph,
        original_graph=graph,
        max_count_frontier_per_step=2,
        max_count_per_endpoint_schema=2,
    )

    assert rows[0]["count_capped"] is True
    assert rows[0]["original_count_bounded"] <= 2


def test_invalid_schema_path_returns_zero_samples_with_clear_status():
    from hesf_coarsen.eval.metapath_retention import sample_typed_paths

    graph = _toy_graph()
    rows = sample_typed_paths(
        graph,
        [{"name": "impossible", "steps": [1, 1], "start_type": 0, "end_type": 0}],
        sample_seed=23,
        max_samples_per_schema=4,
        return_status_rows=True,
    )

    assert len(rows) == 1
    assert rows[0]["sample_status"] == "invalid_schema_path"


def test_metapath_retention_module_does_not_use_dense_products():
    import hesf_coarsen.eval.metapath_retention as module

    source = inspect.getsource(module)
    forbidden = ["apply_metapath_operator", "toarray(", "todense(", "matrix_power", "np.linalg.eig"]
    assert not any(token in source for token in forbidden)
