import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.ops.fusion_weights import compute_relation_fusion_weights
from hesf_coarsen.sketch import lowpass as lowpass_module


def _two_relation_graph() -> HeteroGraph:
    node_type = np.zeros(4, dtype=np.int32)
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
    specs = {
        relation_id: RelationSpec(relation_id, f"relation_{relation_id}", 0, 0)
        for relation_id in relations
    }
    return HeteroGraph(4, node_type, relations, specs)


def _row_normalized_after_mean_centering(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32)
    Z = Z - Z.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, 1e-6)


def test_reliability_relation_weights_downweight_high_energy_relation():
    graph = _two_relation_graph()
    signals = np.array([[0.0], [0.0], [10.0], [-10.0]], dtype=np.float32)

    weights = compute_relation_fusion_weights(
        graph,
        signals,
        {
            "fusion": {
                "relation_weighting": "reliability",
                "volume_eta": 0.0,
                "energy_epsilon": 1e-3,
            }
        },
    )

    assert set(weights) == {0, 1}
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights[0] > 0.99
    assert weights[1] < 0.01


def test_lowpass_sketch_uses_reliability_relation_weights(monkeypatch):
    graph = _two_relation_graph()
    probe = np.array([[0.0], [0.0], [10.0], [-10.0]], dtype=np.float32)

    def fixed_probe(num_nodes, dim, seed, probe="rademacher"):
        assert (num_nodes, dim) == (4, 1)
        return probe_matrix.copy()

    probe_matrix = probe
    monkeypatch.setattr(lowpass_module, "generate_probe", fixed_probe)

    config = {
        "seed": 7,
        "sketch": {"dim": 1, "order": 1, "num_scales": 1, "dtype": "float32"},
        "fusion": {
            "relation_weighting": "reliability",
            "volume_eta": 0.0,
            "energy_epsilon": 1e-3,
        },
    }

    weights = compute_relation_fusion_weights(graph, probe, config)
    expected = apply_fused_smoothing(graph, probe, weights)
    expected = _row_normalized_after_mean_centering(expected)

    actual = lowpass_module.compute_lowpass_sketch(graph, config)

    assert np.allclose(actual, expected)
