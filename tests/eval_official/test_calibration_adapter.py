from __future__ import annotations

import numpy as np


def test_calibrate_logits_nested_is_deterministic_finite_and_shape_preserving() -> None:
    from hesf_coarsen.eval.official.calibration_adapter import calibrate_logits_nested

    val_logits = np.array(
        [
            [3.0, 0.2, 0.1],
            [0.1, 2.5, 0.3],
            [0.2, 0.4, 2.8],
            [1.8, 1.2, 0.1],
            [0.2, 1.1, 1.0],
            [0.6, 0.7, 1.5],
        ],
        dtype=np.float32,
    )
    val_labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    test_logits = np.array([[1.5, 0.3, 0.2], [0.2, 0.6, 1.6]], dtype=np.float32)

    first = calibrate_logits_nested(val_logits, val_labels, test_logits, split_seeds=(11, 22))
    second = calibrate_logits_nested(val_logits, val_labels, test_logits, split_seeds=(11, 22))

    assert np.asarray(first["calibrated_test_logits"]).shape == test_logits.shape
    assert first["best_temperature"] == second["best_temperature"]
    assert first["class_bias_vector"] == second["class_bias_vector"]
    assert first["nested_accuracy_mean"] == second["nested_accuracy_mean"]
    assert first["calibration_uses_test_labels"] is False
    assert first["constraint_satisfied_rate"] >= 0.0
    for key in ("ece_before", "ece_after", "nll_before", "nll_after", "brier_before", "brier_after"):
        assert np.isfinite(float(first[key]))


def test_calibration_candidate_log_records_macro_guard_and_validation_gains() -> None:
    from hesf_coarsen.eval.official.calibration_adapter import calibrate_logits_nested

    val_logits = np.array(
        [[2.0, 0.1, 0.0], [0.0, 2.0, 0.1], [0.1, 0.2, 2.1], [1.5, 0.8, 0.2]],
        dtype=np.float32,
    )
    val_labels = np.array([0, 1, 2, 0], dtype=np.int64)
    test_logits = np.array([[0.9, 0.5, 0.2]], dtype=np.float32)

    result = calibrate_logits_nested(
        val_logits,
        val_labels,
        test_logits,
        split_seeds=(11,),
        temperature_grid=(0.5, 1.0),
        class_bias_grid=(-0.25, 0.0, 0.25),
    )

    candidates = result["calibration_candidates"]
    assert candidates
    assert all("macro_guard_satisfied" in row for row in candidates)
    assert all("accuracy_gain" in row for row in candidates)
    assert all("macro_loss" in row for row in candidates)
    assert all(row["calibration_uses_test_labels"] is False for row in candidates)
