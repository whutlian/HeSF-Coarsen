import numpy as np

from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.sketch.chebyshev import chebyshev_heat_coefficients, chebyshev_heat_filter
from hesf_coarsen.sketch.lowpass import compute_lowpass_sketch
from hesf_coarsen.sketch.operators import apply_fused_operator


def _line_graph() -> HeteroGraph:
    relations = {
        0: RelationAdj(
            src=np.array([0, 1], dtype=np.int64),
            dst=np.array([1, 2], dtype=np.int64),
            weight=np.ones(2, dtype=np.float32),
            src_type=0,
            dst_type=0,
            relation_id=0,
        )
    }
    specs = {0: RelationSpec(0, "line", 0, 0)}
    return HeteroGraph(
        num_nodes=3,
        node_type=np.zeros(3, dtype=np.int32),
        relations=relations,
        relation_specs=specs,
    )


def _dense_fused_operator(graph: HeteroGraph, relation_weights: dict[int, float]) -> np.ndarray:
    eye = np.eye(graph.num_nodes, dtype=np.float32)
    columns = [apply_fused_operator(graph, eye[:, i : i + 1], relation_weights)[:, 0] for i in range(graph.num_nodes)]
    return np.stack(columns, axis=1)


def test_chebyshev_heat_coefficients_are_finite_and_expected_shape():
    coeffs = chebyshev_heat_coefficients(heat_time=1.5, order=5, quadrature_points=128)

    assert coeffs.shape == (6,)
    assert np.all(np.isfinite(coeffs))


def test_chebyshev_heat_filter_matches_tiny_dense_heat_kernel():
    graph = _line_graph()
    relation_weights = {0: 1.0}
    basis = np.array([[1.0, 0.5], [-0.25, 0.75], [0.0, -1.0]], dtype=np.float32)

    actual = chebyshev_heat_filter(
        graph,
        basis,
        relation_weights,
        heat_time=0.75,
        order=20,
        quadrature_points=256,
    )

    S = _dense_fused_operator(graph, relation_weights)
    L = np.eye(graph.num_nodes, dtype=np.float64) - S.astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    expected = eigvecs @ np.diag(np.exp(-0.75 * eigvals)) @ eigvecs.T @ basis.astype(np.float64)
    relative_error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)

    assert relative_error < 0.15
    assert np.all(np.isfinite(actual))


def test_chebyshev_lowpass_sketch_has_configured_dtype_and_no_invalid_values():
    graph = generate_synthetic_graph(num_users=6, num_items=4, num_tags=3, seed=21)
    config = dict(DEFAULT_CONFIG)
    config["sketch"] = {
        "method": "chebyshev_heat",
        "dim": 12,
        "order": 8,
        "heat_times": [1.0, 3.0],
        "chebyshev_quadrature_points": 128,
        "seed": 42,
        "dtype": "float16",
        "row_normalize": True,
    }
    config["fusion"] = {
        "symmetric_relation_operator": True,
        "reverse_relation_policy": "include_all",
        "relation_weighting": {"method": "inverse_energy", "seed": 42},
    }
    config["metapath_sketch"] = {"enabled": False}

    Z = compute_lowpass_sketch(graph, config)

    assert Z.shape == (graph.num_nodes, 12)
    assert Z.dtype == np.float16
    assert int(np.isnan(Z).sum()) == 0
    assert int(np.isinf(Z).sum()) == 0
    assert config["_last_sketch_diagnostics"]["sketch_method"] == "chebyshev_heat"
    assert config["_last_sketch_diagnostics"]["nan_count"] == 0
    assert config["_last_sketch_diagnostics"]["inf_count"] == 0
