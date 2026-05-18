from __future__ import annotations

import inspect

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _toy_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1], dtype=np.int32)
    rel = RelationAdj(
        src=np.array([0, 1], dtype=np.int64),
        dst=np.array([2, 3], dtype=np.int64),
        weight=np.ones(2, dtype=np.float32),
        src_type=0,
        dst_type=1,
        relation_id=0,
    )
    back = RelationAdj(
        src=rel.dst,
        dst=rel.src,
        weight=rel.weight,
        src_type=1,
        dst_type=0,
        relation_id=1,
    )
    return HeteroGraph(
        num_nodes=4,
        node_type=node_type,
        relations={0: rel, 1: back},
        relation_specs={
            0: RelationSpec(0, "a-b", 0, 1),
            1: RelationSpec(1, "b-a", 1, 0),
        },
    )


def test_identity_assignment_has_near_zero_holdout_error() -> None:
    from hesf_coarsen.eval.holdout_operator_probes import evaluate_holdout_operator_probe

    graph = _toy_graph()
    assignment = Assignment(np.arange(graph.num_nodes), graph.node_type.copy())
    metrics = evaluate_holdout_operator_probe(graph, graph, assignment, dataset="toy", seed=7, probe_dim=8, cheb_order=3)
    assert metrics["holdout_operator_relative_error"] < 1.0e-6
    assert metrics["holdout_operator_cosine_similarity"] > 0.999999


def test_collapsed_assignment_has_positive_holdout_error_and_typewise_metrics() -> None:
    from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
    from hesf_coarsen.eval.holdout_operator_probes import evaluate_holdout_operator_probe

    graph = _toy_graph()
    assignment = Assignment(np.array([0, 0, 1, 1], dtype=np.int64), np.array([0, 1], dtype=np.int32))
    coarse = coarsen_graph(graph, assignment)
    metrics = evaluate_holdout_operator_probe(graph, coarse, assignment, dataset="toy", seed=7, probe_dim=8, cheb_order=3)
    assert metrics["holdout_operator_relative_error"] > 0.01
    assert "0" in metrics["holdout_operator_typewise_relative_error"]
    assert "1" in metrics["holdout_operator_typewise_relative_error"]


def test_holdout_probe_generation_is_deterministic_and_distinct_from_scoring_namespace() -> None:
    from hesf_coarsen.eval.holdout_operator_probes import make_holdout_probe_matrix, stable_probe_seed

    graph = _toy_graph()
    a = make_holdout_probe_matrix(graph, dataset="toy", seed=3, probe_dim=4, namespace="holdout_operator")
    b = make_holdout_probe_matrix(graph, dataset="toy", seed=3, probe_dim=4, namespace="holdout_operator")
    c = make_holdout_probe_matrix(graph, dataset="toy", seed=3, probe_dim=4, namespace="scoring")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert stable_probe_seed("toy", 3, "holdout_operator") != stable_probe_seed("toy", 3, "scoring")


def test_holdout_operator_probe_source_does_not_materialize_dense_adjacency() -> None:
    import hesf_coarsen.eval.holdout_operator_probes as module

    source = inspect.getsource(module)
    forbidden = ["toarray(", "todense(", "matrix_power", "np.linalg.eig"]
    assert not any(token in source for token in forbidden)
