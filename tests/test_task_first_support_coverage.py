import numpy as np
from dataclasses import replace

from hesf_coarsen.task_first.config import SupportCoverageConfig, TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_coverage import coverage_components_for_merge, delta_support_coverage_for_merge
from tests.test_task_first_state import make_target_support_graph


def test_cross_anchor_collision_is_positive_for_disjoint_class_contexts():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, support_coverage=SupportCoverageConfig(mode="cross_anchor_collision"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    components = coverage_components_for_merge(2, 4, state, cfg)

    assert components["same_anchor_loss"] == 0.0
    assert components["cross_anchor_collision_loss"] > 0.0
    assert delta_support_coverage_for_merge(2, 4, state, cfg) > 0.0


def test_combined_coverage_keeps_old_same_anchor_signal():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, support_coverage=SupportCoverageConfig(mode="combined"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert delta_support_coverage_for_merge(2, 3, state, cfg) > 0.0
