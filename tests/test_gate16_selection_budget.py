import numpy as np

from hesf_coarsen.task_first.selection.config import SupportFeatureConfig, SupportSelectorConfig
from hesf_coarsen.task_first.selection.contribution import compute_support_importance
from hesf_coarsen.task_first.selection.selector import select_support_nodes
from hesf_coarsen.task_first.selection.support_features import build_support_selection_features
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_budget_utility_reports_exact_support_count():
    from hesf_coarsen.task_first.selection.budget import (
        assert_budget_close,
        budget_diagnostics,
        desired_support_count,
    )

    assert desired_support_count(4, 0.5) == 2
    assert_budget_close(2, 2)
    diag = budget_diagnostics(num_support=4, support_ratio=0.5, realized_support_count=2)
    assert diag["requested_support_count"] == 2
    assert diag["realized_support_count"] == 2
    assert diag["support_budget_error"] == 0
    assert diag["support_budget_exact_match"] is True


def test_validation_proxy_diverse_is_explicitly_not_true_validation_feedback():
    graph = make_gate15_graph()
    labels = graph.labels
    train_mask, _, _ = split_masks()
    features = build_support_selection_features(graph, labels, train_mask, 0, None, SupportFeatureConfig())
    importance = compute_support_importance(features, None, mode="validation_proxy_diverse")["importance"]

    selected = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(selector="validation_proxy_diverse"),
    )

    assert selected["diagnostics"]["selector_uses_true_validation_feedback"] is False
    assert selected["diagnostics"]["selector_family"] == "proxy_selector_baseline"
    assert selected["diagnostics"]["support_budget_exact_match"] is True


def test_sensitivity_block_selector_is_budget_exact_and_test_label_free():
    graph = make_gate15_graph()
    labels = graph.labels
    train_mask, _, _ = split_masks()
    features = build_support_selection_features(graph, labels, train_mask, 0, None, SupportFeatureConfig())
    importance = compute_support_importance(features, None, mode="sensitivity_block_selector")["importance"]

    selected = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(selector="sensitivity_block_selector"),
    )

    assert len(selected["selected_support_nodes"]) == 2
    assert selected["diagnostics"]["support_budget_exact_match"] is True
    assert selected["diagnostics"]["selector_uses_test_labels"] is False
    assert selected["diagnostics"]["selector_uses_true_validation_feedback"] is False
    assert "selected_by_relation_bucket" in selected["diagnostics"]
