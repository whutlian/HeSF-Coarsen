import numpy as np

from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportFeatureConfig, SupportSelectorConfig
from hesf_coarsen.task_first.selection.support_features import build_support_selection_features
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_class_anchor_relation_prototype_background_reports_granular_diagnostics():
    graph = make_gate15_graph()
    labels = graph.labels
    train_mask, _, _ = split_masks()
    features = build_support_selection_features(graph, labels, train_mask, 0, None, SupportFeatureConfig())
    target_nodes = np.flatnonzero(graph.node_type == 0)
    selected_support = np.array([4], dtype=np.int64)

    coarse, assignment, diagnostics = build_selected_support_graph(
        graph,
        selected_support,
        SupportSelectorConfig(background_strategy="class_anchor_relation_prototype"),
        target_node_type=0,
        support_features=features,
    )

    assert diagnostics["background_strategy"] == "class_anchor_relation_prototype"
    assert diagnostics["selected_raw_support_count"] == 1
    assert diagnostics["unselected_support_count"] == 3
    assert diagnostics["prototype_background_count"] >= 2
    assert diagnostics["prototype_member_count_max"] >= 1
    assert len(np.unique(assignment.assignment[target_nodes])) == len(target_nodes)
    assert coarse.num_nodes == len(assignment.supernode_type)
