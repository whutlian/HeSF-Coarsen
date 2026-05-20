import numpy as np
import pytest
from dataclasses import replace

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.io.schema import validate_schema
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.eval_protocol import (
    APPROX_FULL_TARGET_ADAPTER,
    COARSE_TRANSFER,
    REAL_FULL_TARGET_INFERENCE,
    evaluate_approx_full_target_adapter_protocol,
    evaluate_coarse_transfer_protocol,
    evaluate_real_full_target_protocol,
)
from hesf_coarsen.task_first.pipeline import (
    build_support_only_task_first_coarsening,
    build_target_preserve_assignment_template,
    task_first_support_merge_budget,
)
from tests.test_task_first_state import make_target_support_graph


class LiteBackbone:
    fidelity = "lite"

    def fit(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        return np.array([], dtype=np.int64)


class AdapterBackbone(LiteBackbone):
    fidelity = "adapter"


class FaithfulBackbone(LiteBackbone):
    fidelity = "faithful"


def test_target_preserve_template_is_target_identity_plus_support_singletons():
    graph = make_target_support_graph()
    cfg = TaskFirstConfig(target_node_type=0)

    assignment = build_target_preserve_assignment_template(graph, cfg)

    target_nodes = np.flatnonzero(graph.node_type == 0)
    support_nodes = np.flatnonzero(graph.node_type != 0)
    sizes = assignment.cluster_sizes()
    assert assignment.assignment[target_nodes].tolist() == list(range(len(target_nodes)))
    assert np.all(sizes[assignment.assignment[target_nodes]] == 1)
    assert np.all(sizes[assignment.assignment[support_nodes]] == 1)
    assert assignment.supernode_type[: len(target_nodes)].tolist() == [0, 0]


def test_support_only_pipeline_preserves_all_target_singletons_and_emits_diagnostics():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    store = ArrayCandidateStore(graph.node_type, K=4, same_type_only=True)
    store.add(2, 3, 0.1, "bucket")
    store.add(2, 4, 0.2, "bucket")
    store.add(0, 2, 0.0, "fallback")

    result = build_support_only_task_first_coarsening(
        graph,
        store,
        labels,
        train_mask,
        cfg,
    )

    target_nodes = np.flatnonzero(graph.node_type == 0)
    sizes = result.assignment.cluster_sizes()
    assert result.assignment.assignment[target_nodes].tolist() == [0, 1]
    assert np.all(sizes[result.assignment.assignment[target_nodes]] == 1)
    assert result.graph.num_nodes < graph.num_nodes
    assert set(result.graph.relation_specs) == set(graph.relation_specs)
    for relation_id, rel in graph.relations.items():
        original_weight = float(np.sum(rel.weight.astype(np.float64)))
        coarse_weight = float(np.sum(result.graph.relations[relation_id].weight.astype(np.float64)))
        assert coarse_weight == pytest.approx(original_weight)
    validate_schema(result.graph)
    for key in (
        "matching_method",
        "pipeline_steps",
        "target_spec_error",
        "relation_response_error",
        "support_coverage_error",
        "support_purity_error",
        "num_support_candidates_scored",
        "num_support_candidates_rejected_by_purity",
        "num_support_candidates_rejected_by_constraints",
    ):
        assert key in result.diagnostics


def test_pipeline_uses_greedy_cluster_on_support_nodes_not_pair_only_matching():
    graph = make_target_support_graph()
    graph.relations[0].src[2] = 0
    graph.relations[1].dst[2] = 0
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    store = ArrayCandidateStore(graph.node_type, K=4, same_type_only=True)
    store.add(2, 3, 0.1, "bucket")
    store.add(3, 4, 0.2, "bucket")

    result = build_support_only_task_first_coarsening(
        graph,
        store,
        labels,
        train_mask,
        cfg,
    )

    target_nodes = np.flatnonzero(graph.node_type == 0)
    support_nodes = np.flatnonzero(graph.node_type != 0)
    sizes = result.assignment.cluster_sizes()
    assert result.diagnostics["matching_method"] == "greedy_cluster"
    assert result.assignment.assignment[target_nodes].tolist() == [0, 1]
    assert np.max(sizes[result.assignment.assignment[support_nodes]]) == 3


def test_target_ratio_budget_limits_support_merges_and_reports_infeasible_floor():
    graph = make_target_support_graph()
    graph.relations[0].src[2] = 0
    graph.relations[1].dst[2] = 0
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, target_ratio=0.8)
    store = ArrayCandidateStore(graph.node_type, K=4, same_type_only=True)
    store.add(2, 3, 0.1, "bucket")
    store.add(3, 4, 0.2, "bucket")

    result = build_support_only_task_first_coarsening(
        graph,
        store,
        labels,
        train_mask,
        cfg,
    )

    assert result.diagnostics["max_support_merges"] == 1
    assert result.diagnostics["selected_support_merges"] == 1
    assert result.graph.num_nodes == 4

    infeasible = task_first_support_merge_budget(
        graph,
        replace(cfg, target_ratio=0.2),
    )
    assert infeasible["requested_ratio_infeasible"] is True
    assert infeasible["desired_total_nodes"] == 2


def test_real_full_target_protocol_rejects_lite_backbones():
    graph = make_target_support_graph()
    with pytest.raises(ValueError, match="official|faithful"):
        evaluate_real_full_target_protocol(graph, graph, LiteBackbone())


def test_eval_protocols_are_explicitly_distinguished():
    graph = make_target_support_graph()

    coarse = evaluate_coarse_transfer_protocol(graph, LiteBackbone())
    approx = evaluate_approx_full_target_adapter_protocol(graph, graph, AdapterBackbone())
    real = evaluate_real_full_target_protocol(graph, graph, FaithfulBackbone())

    assert coarse.protocol == COARSE_TRANSFER
    assert approx.protocol == APPROX_FULL_TARGET_ADAPTER
    assert real.protocol == REAL_FULL_TARGET_INFERENCE
    assert {coarse.protocol, approx.protocol, real.protocol} == {
        COARSE_TRANSFER,
        APPROX_FULL_TARGET_ADAPTER,
        REAL_FULL_TARGET_INFERENCE,
    }
