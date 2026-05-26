from __future__ import annotations

import numpy as np


def test_zero_paper_features_keeps_shape_and_uses_no_labels() -> None:
    from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix

    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    transformed, audit = transform_feature_matrix(features, "zero-paper", seed=1)

    assert transformed.shape == features.shape
    assert float(transformed.sum()) == 0.0
    assert audit["fit_uses_labels"] is False
    assert audit["fit_uses_test_labels"] is False


def test_pca_paper_dim64_caps_to_requested_dim() -> None:
    from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix

    rng = np.random.default_rng(1)
    features = rng.normal(size=(12, 80)).astype(np.float32)

    transformed, audit = transform_feature_matrix(features, "pca-paper-64", seed=1)

    assert transformed.shape == (12, 64)
    assert audit["feature_dim"] == 64
    assert audit["transform_name"] == "pca-paper-64"


def test_random_projection_is_deterministic_for_seed() -> None:
    from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix

    features = np.arange(60, dtype=np.float32).reshape(10, 6)

    first, _ = transform_feature_matrix(features, "random_projection_dim4", seed=7)
    second, _ = transform_feature_matrix(features, "random_projection_dim4", seed=7)

    np.testing.assert_allclose(first, second)


def test_int8_sidecar_reports_scale_metadata() -> None:
    from hesf_coarsen.eval.official.paper_feature_transform import transform_feature_matrix

    features = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    transformed, audit = transform_feature_matrix(features, "int8-paper", seed=1)

    assert transformed.dtype == np.float32
    assert audit["feature_dtype"] == "int8"
    assert audit["sidecar_metadata_bytes"] > 0
    assert "scale" in audit["metadata_keys"]
