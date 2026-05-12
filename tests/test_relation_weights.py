import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.sketch.relation_weights import compute_relation_weights


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
