import numpy as np

from hesf_coarsen.task_first.config import SupportCoverageConfig, TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_coverage import (
    coverage_v2_components_for_merge,
    delta_support_coverage_for_merge,
)
from tests.test_task_first_state import make_target_support_graph


def test_coverage_v2_penalizes_disjoint_anchor_different_class_contexts():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, support_coverage=SupportCoverageConfig(mode="coverage_v2"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    components = coverage_v2_components_for_merge(2, 4, state, cfg)

    assert components["anchor_distribution_collision"] > 0.0
    assert components["class_context_collision"] > 0.0
    assert components["receptive_field_diversity_loss"] > 0.0
    assert delta_support_coverage_for_merge(2, 4, state, cfg) > 0.0


def test_coverage_v2_is_lower_for_similar_anchor_class_contexts():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, support_coverage=SupportCoverageConfig(mode="coverage_v2"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    similar = delta_support_coverage_for_merge(2, 3, state, cfg)
    different = delta_support_coverage_for_merge(2, 4, state, cfg)

    assert similar < different


def test_coverage_v1_common_anchor_zero_is_explicit_legacy_behavior():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, support_coverage=SupportCoverageConfig(mode="coverage_v1_legacy"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert delta_support_coverage_for_merge(2, 4, state, cfg) == 0.0
