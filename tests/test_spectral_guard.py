import numpy as np

from hesf_coarsen.scoring.guards import apply_spectral_guard


def _source_lookup(left: int, right: int) -> str:
    return {
        (0, 1): "bucket",
        (0, 2): "bucket",
        (1, 2): "onehop",
        (2, 3): "onehop",
    }.get((left, right), "bucket")


def test_spectral_guard_rejects_high_delta_spec_pairs():
    scored = np.array(
        [[0, 1, 0.1], [0, 2, 0.2], [1, 2, 0.3], [2, 3, 0.4]],
        dtype=np.float64,
    )
    terms = {"spec": np.array([0.1, 0.2, 3.0, 0.15], dtype=np.float32)}

    filtered, filtered_terms, diagnostics = apply_spectral_guard(
        scored,
        terms,
        node_type=np.zeros(4, dtype=np.int32),
        source_lookup=_source_lookup,
        config={
            "spectral_guard": {
                "enabled": True,
                "threshold_policy": "per_type_source_quantile",
                "quantile": 0.95,
                "warmup_from_bucket": True,
                "reject_high_delta_spec": True,
                "min_candidates_per_node_after_guard": 0,
            }
        },
    )

    assert filtered.tolist() == [[0.0, 1.0, 0.1], [0.0, 2.0, 0.2], [2.0, 3.0, 0.4]]
    assert np.allclose(filtered_terms["spec"], [0.1, 0.2, 0.15])
    assert diagnostics["rejected_by_spec_count"] == 1
    assert diagnostics["target_pressure_accept_count"] == 0


def test_spectral_guard_respects_min_candidates_per_node_and_logs_pressure():
    scored = np.array([[0, 1, 0.1], [0, 2, 0.2]], dtype=np.float64)
    terms = {"spec": np.array([0.1, 5.0], dtype=np.float32)}

    filtered, _terms, diagnostics = apply_spectral_guard(
        scored,
        terms,
        node_type=np.zeros(3, dtype=np.int32),
        source_lookup=lambda _left, _right: "bucket",
        config={
            "spectral_guard": {
                "enabled": True,
                "quantile": 0.50,
                "reject_high_delta_spec": True,
                "min_candidates_per_node_after_guard": 1,
            }
        },
    )

    assert filtered.shape[0] == 2
    assert diagnostics["rejected_by_spec_count"] == 0
    assert diagnostics["target_pressure_accept_count"] == 1
