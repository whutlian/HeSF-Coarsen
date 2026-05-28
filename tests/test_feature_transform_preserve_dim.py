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
            1: np.arange(240, dtype=np.float32).reshape(3, 80),
            2: np.arange(10, dtype=np.float32).reshape(2, 5),
            3: np.zeros((1, 0), dtype=np.float32),
        },
    )


def _shape_by_type(graph: HeteroGraph) -> dict[int, tuple[int, int]]:
    assert graph.features is not None
    return {int(node_type): tuple(values.shape) for node_type, values in graph.features.items()}


def test_required_repaired_transforms_preserve_every_noninject_type_dimension() -> None:
    from hesf_coarsen.eval.official.feature_ablation_repaired import (
        REQUIRED_FEATURE_TRANSFORMS,
        apply_repaired_feature_transform,
        feature_ablation_shape_safe_pass,
        feature_shape_assertion_rows,
    )

    expected = {
        "raw",
        "zero-author-preserve-dim",
        "zero-paper-preserve-dim",
        "zero-term-preserve-dim",
        "zero-venue-preserve-dim",
        "zero-all-support-preserve-dim",
        "zero-all-features-preserve-dim",
        "paper-only-preserve-original-dims",
        "term-only-preserve-original-dims",
        "venue-only-preserve-original-dims",
        "paper-random-projection64",
        "paper-pca64",
        "inject-zero-venue-dim4231",
    }
    assert set(REQUIRED_FEATURE_TRANSFORMS) == expected

    graph = _dblp_graph_with_zero_dim_venue()
    before = _shape_by_type(graph)

    for transform_name in REQUIRED_FEATURE_TRANSFORMS:
        transformed = apply_repaired_feature_transform(graph, transform_name, seed=7)
        assert transformed.features is not None
        assertions = feature_shape_assertion_rows(
            transform_name,
            before=graph.features,
            after=transformed.features,
        )
        assert feature_ablation_shape_safe_pass(assertions) is True
        if not transform_name.startswith("inject-"):
            assert _shape_by_type(transformed) == before
            assert transformed.features[3].shape == (1, 0)
            assert all(row["pass"] for row in assertions)


@pytest.mark.parametrize(
    ("transform_name", "zero_types", "kept_types"),
    [
        ("zero-author-preserve-dim", {0}, {1, 2, 3}),
        ("zero-paper-preserve-dim", {1}, {0, 2, 3}),
        ("zero-term-preserve-dim", {2}, {0, 1, 3}),
        ("zero-venue-preserve-dim", {3}, {0, 1, 2}),
        ("zero-all-support-preserve-dim", {1, 2, 3}, {0}),
        ("zero-all-features-preserve-dim", {0, 1, 2, 3}, set()),
        ("paper-only-preserve-original-dims", {0, 2, 3}, {1}),
        ("term-only-preserve-original-dims", {0, 1, 3}, {2}),
        ("venue-only-preserve-original-dims", {0, 1, 2}, {3}),
    ],
)
def test_zero_and_only_transforms_keep_original_shapes_and_expected_values(
    transform_name: str,
    zero_types: set[int],
    kept_types: set[int],
) -> None:
    from hesf_coarsen.eval.official.feature_ablation_repaired import apply_repaired_feature_transform

    graph = _dblp_graph_with_zero_dim_venue()
    transformed = apply_repaired_feature_transform(graph, transform_name)
    assert transformed.features is not None

    for node_type in zero_types:
        assert transformed.features[node_type].shape == graph.features[node_type].shape
        assert int(np.count_nonzero(transformed.features[node_type])) == 0
    for node_type in kept_types:
        np.testing.assert_allclose(transformed.features[node_type], graph.features[node_type])


@pytest.mark.parametrize("transform_name", ["paper-random-projection64", "paper-pca64"])
def test_projection_transforms_are_shape_safe_adapter_diagnostics(transform_name: str) -> None:
    from hesf_coarsen.eval.official.feature_ablation_repaired import apply_repaired_feature_transform

    graph = _dblp_graph_with_zero_dim_venue()

    transformed = apply_repaired_feature_transform(graph, transform_name, seed=11)

    assert transformed.features is not None
    assert transformed.features[1].shape == graph.features[1].shape
    assert transformed.features[0].shape == graph.features[0].shape
    assert transformed.features[2].shape == graph.features[2].shape
    assert transformed.features[3].shape == (1, 0)
    audit = transformed._gate21_7_transform_audit
    assert audit["internal_projection_dim"] == 64
    assert audit["shape_safe_adapter_diagnostic"] is True


def test_zero_dim_venue_can_only_become_nonzero_dim_with_inject_prefix() -> None:
    from hesf_coarsen.eval.official.feature_ablation_repaired import apply_repaired_feature_transform

    graph = _dblp_graph_with_zero_dim_venue()

    with pytest.raises(ValueError, match="inject-"):
        apply_repaired_feature_transform(graph, "zero-venue-dim4231")

    transformed = apply_repaired_feature_transform(graph, "inject-zero-venue-dim4231")

    assert transformed.features is not None
    assert transformed.features[3].shape == (1, 4231)
    assert int(np.count_nonzero(transformed.features[3])) == 0
    assert transformed._gate21_7_transform_audit["diagnostic_only"] is True
