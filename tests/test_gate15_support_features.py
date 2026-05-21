import numpy as np

from hesf_coarsen.task_first.selection.config import SupportFeatureConfig
from hesf_coarsen.task_first.selection.support_features import build_support_selection_features
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_support_features_are_finite_and_report_footprint_state():
    graph = make_gate15_graph()
    train_mask, _val_mask, _test_mask = split_masks()

    result = build_support_selection_features(
        graph,
        np.asarray(graph.labels),
        train_mask,
        target_node_type=0,
        teacher_outputs=None,
        cfg=SupportFeatureConfig(),
    )

    assert result["support_nodes"].tolist() == [4, 5, 6, 7]
    assert result["feature_matrix"].shape[0] == 4
    assert np.isfinite(result["feature_matrix"]).all()
    assert result["diagnostics"]["support_feature_nan_count"] == 0
    assert "zero_footprint_support_share" in result["diagnostics"]
    assert "target_response_signature" in result["component_matrices"]


def test_support_features_do_not_change_when_only_test_labels_change():
    graph = make_gate15_graph()
    train_mask, _val_mask, test_mask = split_masks()
    labels = np.asarray(graph.labels).copy()
    changed = labels.copy()
    changed[test_mask] = 99

    original_features = build_support_selection_features(
        graph,
        labels,
        train_mask,
        target_node_type=0,
        teacher_outputs=None,
        cfg=SupportFeatureConfig(),
    )
    changed_features = build_support_selection_features(
        graph,
        changed,
        train_mask,
        target_node_type=0,
        teacher_outputs=None,
        cfg=SupportFeatureConfig(),
    )

    np.testing.assert_allclose(original_features["feature_matrix"], changed_features["feature_matrix"])
