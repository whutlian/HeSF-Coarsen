import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.sketch.metapath import compute_metapath_weights
from hesf_coarsen.sketch.relation_weights import _basis_for_energy, compute_relation_weights


def _smooth_noisy_graph() -> HeteroGraph:
    relations = {
        0: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([1, 0], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=0,
            dst_type=0,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([2, 3], dtype=np.int64),
            dst=np.array([3, 2], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=0,
            dst_type=0,
            relation_id=1,
        ),
    }
    specs = {relation_id: RelationSpec(relation_id, f"r{relation_id}", 0, 0) for relation_id in relations}
    return HeteroGraph(4, np.zeros(4, dtype=np.int32), relations, specs)


def test_relation_weight_methods_are_normalized_and_non_negative():
    graph = _smooth_noisy_graph()
    basis = np.array([[0.0], [0.0], [10.0], [-10.0]], dtype=np.float32)

    for method in ["uniform", "volume", "inverse_energy"]:
        result = compute_relation_weights(
            graph,
            {
                "fusion": {
                    "relation_weighting": {
                        "method": method,
                        "eta": 0.5,
                        "gamma": 1.0,
                        "epsilon": 1e-6,
                    }
                }
            },
            basis=basis,
        )

        assert set(result.weights) == {0, 1}
        assert np.isclose(sum(result.weights.values()), 1.0)
        assert all(weight >= 0.0 for weight in result.weights.values())


def test_inverse_energy_relation_weight_prefers_smooth_relation():
    graph = _smooth_noisy_graph()
    basis = np.array([[0.0], [0.0], [10.0], [-10.0]], dtype=np.float32)

    result = compute_relation_weights(
        graph,
        {
            "fusion": {
                "relation_weighting": {
                    "method": "inverse_energy",
                    "eta": 0.0,
                    "gamma": 1.0,
                    "epsilon": 1e-3,
                }
            }
        },
        basis=basis,
    )

    assert result.weights[0] > result.weights[1]
    assert result.energy_estimates[0] < result.energy_estimates[1]
    assert result.diagnostics["relation_weighting_method"] == "inverse_energy"
    assert result.diagnostics["energy_basis_object"] == "Z_X"
    assert result.diagnostics["energy_estimator"] == "sampled_normalized_edge_energy"


def test_inverse_energy_metapath_weight_prefers_smooth_path():
    graph = _smooth_noisy_graph()
    basis = np.array([[0.0], [0.0], [10.0], [-10.0]], dtype=np.float32)
    paths = [
        {
            "name": "smooth_relation_path",
            "start_type": 0,
            "end_type": 0,
            "steps": [{"relation_id": 0, "direction": "forward"}],
        },
        {
            "name": "noisy_relation_path",
            "start_type": 0,
            "end_type": 0,
            "steps": [{"relation_id": 1, "direction": "forward"}],
        },
    ]

    result = compute_metapath_weights(
        graph,
        {
            "metapath_sketch": {
                "weighting": {
                    "method": "inverse_energy",
                    "eta": 0.0,
                    "gamma": 1.0,
                    "epsilon": 1e-3,
                }
            }
        },
        paths,
        basis=basis,
    )

    assert np.isclose(sum(result.weights.values()), 1.0)
    assert result.weights["smooth_relation_path"] > result.weights["noisy_relation_path"]
    assert result.energy_estimates["smooth_relation_path"] < result.energy_estimates["noisy_relation_path"]
    assert result.diagnostics["metapath_weighting_method"] == "inverse_energy"
    assert result.diagnostics["energy_basis_object"] == "Z_X"


def test_feature_smoothness_basis_uses_low_dim_feature_projection():
    graph = HeteroGraph(
        4,
        np.array([0, 0, 1, 1], dtype=np.int32),
        {},
        features={
            0: np.arange(2 * 8, dtype=np.float32).reshape(2, 8),
            1: np.arange(2 * 6, dtype=np.float32).reshape(2, 6),
        },
    )

    basis, source = _basis_for_energy(
        graph,
        {
            "seed": 17,
            "features": {"projected_dim": 3, "projection_dtype": "float16"},
            "fusion": {"relation_weighting": {"energy_basis_dim": 5}},
        },
        {"energy_basis_dim": 5},
        None,
        "feature_smoothness",
    )

    assert source == "features"
    assert basis.shape == (4, 3)
    assert basis.dtype == np.float16
