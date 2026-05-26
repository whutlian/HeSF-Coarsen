from __future__ import annotations

import numpy as np
import pytest

from hesf_coarsen.io.schema import HeteroGraph


def _dblp_graph_with_zero_dim_venue() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=8,
        node_type=np.array([0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int32),
        relations={},
        features={
            0: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            1: np.arange(12, dtype=np.float32).reshape(3, 4),
            2: np.ones((2, 3), dtype=np.float32),
            3: np.zeros((1, 0), dtype=np.float32),
        },
    )


def test_zero_transform_preserves_zero_dimensional_feature_matrix() -> None:
    from hesf_coarsen.eval.official.safe_feature_transforms import transform_gate21_6_graph_features

    graph = _dblp_graph_with_zero_dim_venue()

    transformed = transform_gate21_6_graph_features(graph, "zero-venue-preserve-dim")

    assert transformed.features is not None
    assert transformed.features[3].shape == (1, 0)
    assert transformed.features[3].dtype == np.float32
    audit = transformed._gate21_6_transform_audit
    assert audit["original_feature_shape_by_type"]["venue"] == [1, 0]
    assert audit["transformed_feature_shape_by_type"]["venue"] == [1, 0]
    assert audit["shape_preserved_by_type"]["venue"] is True


def test_zero_all_features_preserves_dimensions_and_reports_no_test_leakage() -> None:
    from hesf_coarsen.eval.official.safe_feature_transforms import transform_gate21_6_graph_features

    graph = _dblp_graph_with_zero_dim_venue()

    transformed = transform_gate21_6_graph_features(graph, "zero-all-features-preserve-dim")

    assert transformed.features is not None
    assert transformed.features[0].shape == (2, 2)
    assert transformed.features[1].shape == (3, 4)
    assert transformed.features[2].shape == (2, 3)
    assert transformed.features[3].shape == (1, 0)
    assert np.count_nonzero(transformed.features[0]) == 0
    assert np.count_nonzero(transformed.features[1]) == 0
    assert np.count_nonzero(transformed.features[2]) == 0
    audit = transformed._gate21_6_transform_audit
    assert audit["feature_transform_leakage_flag"] is False
    assert audit["feature_transform_fit_split"] == "train_only_or_unsupervised"
    assert audit["uses_test_data_for_transform"] is False


def test_dimension_injection_requires_inject_prefix() -> None:
    from hesf_coarsen.eval.official.safe_feature_transforms import transform_gate21_6_graph_features

    graph = _dblp_graph_with_zero_dim_venue()

    with pytest.raises(ValueError, match="inject-"):
        transform_gate21_6_graph_features(graph, "zero-venue-dim4231")

    transformed = transform_gate21_6_graph_features(graph, "inject-zero-venue-dim5")
    assert transformed.features is not None
    assert transformed.features[3].shape == (1, 5)
    assert np.count_nonzero(transformed.features[3]) == 0
