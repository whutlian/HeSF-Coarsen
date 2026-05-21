import numpy as np

from hesf_coarsen.task_first.selection.config import SupportFeatureConfig, SupportSelectorConfig
from hesf_coarsen.task_first.selection.contribution import compute_support_importance
from hesf_coarsen.task_first.selection.selector import select_support_nodes
from hesf_coarsen.task_first.selection.support_features import build_support_selection_features
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_teacher_diverse_topk_matches_support_budget():
    graph = make_gate15_graph()
    train_mask, _val_mask, _test_mask = split_masks()
    features = build_support_selection_features(
        graph,
        np.asarray(graph.labels),
        train_mask,
        target_node_type=0,
        teacher_outputs=None,
        cfg=SupportFeatureConfig(),
    )
    importance = compute_support_importance(features, teacher_outputs=None, mode="teacher_topk")

    selected = select_support_nodes(
        features,
        importance["importance"],
        0.5,
        SupportSelectorConfig(selector="teacher_diverse_topk"),
    )

    assert len(selected["selected_support_nodes"]) == 2
    assert selected["diagnostics"]["selected_support_count"] == 2
    assert abs(selected["diagnostics"]["realized_support_ratio"] - 0.5) <= 1.0e-12
    assert selected["diagnostics"]["class_coverage_after"] >= 1
