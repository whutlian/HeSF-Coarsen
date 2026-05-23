from __future__ import annotations

import numpy as np

from experiments.scripts.summarize_gate19_1 import build_gate19_1_pareto, summarize_rows
from hesf_coarsen.eval.calibration import calibrate_logits_temperature_bias, nested_calibration_split
from hesf_coarsen.eval.per_class import confusion_matrix_rows, per_class_metrics


def test_nested_calibration_split_is_deterministic_disjoint_and_label_aware():
    val_idx = np.arange(12, dtype=np.int64)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    first = nested_calibration_split(val_idx, labels, seed=23456)
    second = nested_calibration_split(val_idx, labels, seed=23456)

    assert np.array_equal(first["val_calib"], second["val_calib"])
    assert np.array_equal(first["val_select"], second["val_select"])
    assert set(first["val_calib"]).isdisjoint(set(first["val_select"]))
    assert set(first["val_calib"]) | set(first["val_select"]) == set(val_idx.tolist())
    assert {int(labels[i]) for i in first["val_calib"]} == {0, 1, 2}
    assert {int(labels[i]) for i in first["val_select"]} == {0, 1, 2}


def test_calibration_reports_cost_and_never_uses_test_labels():
    logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)

    fit = calibrate_logits_temperature_bias(
        logits,
        labels,
        temperature_grid=[0.5, 1.0],
        class_bias_grid=[-0.25, 0.0, 0.25],
        macro_guard_epsilon=0.02,
    )

    assert fit["calibrator_uses_test_labels"] is False
    assert fit["calibration_split"] == "val"
    assert fit["constraint_satisfied"] is True
    assert fit["calibration_param_bytes"] == 12


def test_gate19_1_pareto_includes_calibrated_support_and_excludes_aliases():
    rows = [
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "H6-no-spec-support-only-logit-calibrated",
            "method_family": "calibrated_support_baseline",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.03",
            "macro_f1": "0.91",
            "accuracy": "0.92",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "ClusterGate-H6-units-logit-calibrated",
            "method_family": "cluster_diagnostic",
            "eligible_for_main_decision": "false",
            "diagnostic_only": "true",
            "alias_of": "H6-no-spec-support-only-logit-calibrated",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.03",
            "macro_f1": "0.91",
            "accuracy": "0.92",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "STC-feature-cache-quantized-fp16",
            "method_family": "stc_compressed",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.50",
            "macro_f1": "0.87",
            "accuracy": "0.88",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "TypedHash-ChebHeat-support-only",
            "method_family": "support_baseline",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.04",
            "macro_f1": "0.80",
            "accuracy": "0.81",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "TypedHash-ChebHeat-support-only",
            "method_family": "support_baseline",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.04",
            "macro_f1": "0.80",
            "accuracy": "0.81",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
    ]

    frontier = build_gate19_1_pareto(rows)
    methods = {row["method"] for row in frontier}

    assert "H6-no-spec-support-only-logit-calibrated" in methods
    assert "ClusterGate-H6-units-logit-calibrated" not in methods
    assert {row["cost_axis_used"] for row in frontier} == {"total_storage_ratio_vs_full_stc"}


def test_gate19_1_decision_blocks_gate20_when_calibrated_support_dominates():
    rows = [
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "H6-no-spec-support-only-logit-calibrated",
            "method_family": "calibrated_support_baseline",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.03",
            "macro_f1": "0.909",
            "accuracy": "0.914",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
            "nested_calibration_pass": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "STC-feature-cache-quantized-fp16",
            "method_family": "stc_compressed",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.51",
            "macro_f1": "0.870",
            "accuracy": "0.877",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "TypedHash-ChebHeat-support-only",
            "method_family": "support_baseline",
            "eligible_for_main_decision": "true",
            "diagnostic_only": "false",
            "status": "success",
            "total_storage_ratio_vs_full_stc": "0.04",
            "macro_f1": "0.80",
            "accuracy": "0.81",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
    ]

    result = summarize_rows(rows, per_class_present=True, confusion_present=True)

    assert result["decision"] == "GATE20_BLOCKED_BY_CALIBRATED_SUPPORT_BASELINE"
    assert result["gate20_allowed"] is False
    assert result["dblp_best_stc_accuracy_gap_vs_calibrated_support"] == -0.037


def test_per_class_and_confusion_helpers_emit_required_delta_fields():
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    pred = np.asarray([0, 1, 1, 1], dtype=np.int64)
    baseline = {0: {"precision": 1.0, "recall": 1.0, "f1": 1.0}, 1: {"precision": 0.5, "recall": 1.0, "f1": 2.0 / 3.0}}

    per_class = per_class_metrics(
        dataset="DBLP",
        seed=23456,
        method="H6-no-spec-support-only-logit-calibrated",
        method_family="calibrated_support_baseline",
        requested_budget=0.3,
        cost_ratio=0.03,
        total_storage_ratio_vs_full_stc=0.03,
        calibrated=True,
        source_method="H6-no-spec-support-only",
        y_true=truth,
        y_pred=pred,
        train_labels=np.asarray([0, 1], dtype=np.int64),
        val_labels=np.asarray([0, 1], dtype=np.int64),
        baseline_per_class=baseline,
        best_uncalibrated_support_per_class=baseline,
    )
    confusion = confusion_matrix_rows(
        dataset="DBLP",
        seed=23456,
        method="H6-no-spec-support-only-logit-calibrated",
        requested_budget=0.3,
        calibrated=True,
        source_method="H6-no-spec-support-only",
        y_true=truth,
        y_pred=pred,
    )

    assert {"delta_precision_vs_uncalibrated_source", "delta_recall_vs_best_uncalibrated_support"} <= set(per_class[0])
    assert {"true_class", "predicted_class", "normalized_by_true", "normalized_by_pred"} <= set(confusion[0])
