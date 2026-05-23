from __future__ import annotations

import math

import numpy as np

from experiments.scripts.summarize_gate19_2 import (
    build_gate19_2_pareto,
    summarize_rows,
    validation_selected_rows,
)
from hesf_coarsen.eval.logit_ensemble import search_global_convex_ensemble
from hesf_coarsen.task_first.feature_condensation.support_teacher_distill import (
    teacher_unavailable_result,
)


def _base_row(**extra):
    row = {
        "stage": "Gate19.2",
        "dataset": "DBLP",
        "seed": 23456,
        "status": "success",
        "diagnostic_only": False,
        "eligible_for_main_decision": True,
        "primary_eval_mode": "compressed_projected",
        "no_test_leakage": True,
        "typedhash_included": True,
        "total_storage_ratio_vs_full_stc": 0.05,
        "total_storage_ratio_vs_full_graph": 0.03,
        "support_node_ratio": 0.03,
        "support_edge_ratio": 0.04,
        "unit_count_ratio": 0.0,
        "feature_cache_size_ratio": 0.0,
        "path_channel_count_ratio": 0.0,
        "feature_cache_bytes": 0,
        "logit_cache_bytes": 0,
        "model_param_bytes": 0,
        "calibration_param_bytes": 16,
        "ensemble_param_bytes": 0,
        "total_inference_storage_bytes": 100,
        "macro_f1": 0.89,
        "accuracy": 0.89,
        "validation_macro_f1": 0.89,
        "validation_accuracy": 0.89,
    }
    row.update(extra)
    return row


def test_validation_selected_prefers_validation_over_test_oracle():
    rows = [
        _base_row(method="STC-feature-cache-quantized-int8", method_family="stc_compressed", validation_macro_f1=0.91, validation_accuracy=0.90, macro_f1=0.86, accuracy=0.86),
        _base_row(method="STC-feature-cache-quantized-int8", method_family="stc_compressed", validation_macro_f1=0.80, validation_accuracy=0.80, macro_f1=0.94, accuracy=0.94),
    ]

    selected = validation_selected_rows(rows)

    assert selected[0]["accuracy"] == 0.86
    assert selected[0]["validation_accuracy"] == 0.90


def test_summary_keeps_test_oracle_diagnostic_only():
    rows = [
        _base_row(method="TypedHash-ChebHeat-support-only-logit-calibrated", method_family="calibrated_support_baseline", validation_accuracy=0.82, accuracy=0.82),
        _base_row(method="HeSF-CAL-H6", method_family="hesf_cal_support", validation_accuracy=0.90, validation_macro_f1=0.90, accuracy=0.89, macro_f1=0.89, nested_calibration_pass=True),
        _base_row(method="STC-support-teacher-distill-int8", method_family="stc_support_teacher_distill", validation_accuracy=0.88, validation_macro_f1=0.88, accuracy=0.86, macro_f1=0.86),
        _base_row(method="STC-support-teacher-distill-int8", method_family="stc_support_teacher_distill", validation_accuracy=0.70, validation_macro_f1=0.70, accuracy=0.93, macro_f1=0.93),
    ]

    result = summarize_rows(rows, nested_rows=[], per_class_present=True, confusion_present=True)

    assert result["test_oracle_used_for_decision"] is False
    assert result["best_stc_by_validation_selection"] == "STC-support-teacher-distill-int8"
    assert result["best_stc_by_test_oracle"] == "STC-support-teacher-distill-int8"
    assert result["dblp_best_stc_validation_selected_accuracy"] == 0.86
    assert result["decision"] in {
        "STC_DEMOTED_TO_DEPLOYMENT_AUXILIARY",
        "GATE20_BLOCKED_BY_STC_DOMINATED",
        "CONTINUE_TO_GATE20_HESF_CAL_MULTI_SEED",
    }


def test_global_convex_ensemble_weights_sum_to_one_and_use_validation_guard():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    source_val = {
        "a": np.asarray([[4, 0], [0, 4], [3, 0], [0, 3]], dtype=np.float32),
        "b": np.asarray([[0, 3], [4, 0], [0, 2], [3, 0]], dtype=np.float32),
        "c": np.asarray([[2, 0], [0, 2], [1, 0], [0, 1]], dtype=np.float32),
    }

    result = search_global_convex_ensemble(
        source_val,
        labels,
        source_val,
        labels,
        macro_floor=0.95,
        grid_step=0.5,
    )

    assert math.isclose(sum(result["weights"].values()), 1.0, abs_tol=1e-9)
    assert result["constraint_satisfied"] is True
    assert result["val_accuracy"] == 1.0


def test_teacher_unavailable_marks_distillation_failed():
    result = teacher_unavailable_result(
        dataset="DBLP",
        student_method="STC-support-teacher-distill-int8",
        teacher_method="HeSF-CAL-H6",
        quantization_mode="int8",
        feature_cache_size_ratio=0.5,
        path_channel_count_ratio=0.5,
    )

    assert result["teacher_status"] == "unavailable"
    assert result["method_failed"] is True
    assert result["student_teacher_KL"] == "NaN"


def test_pareto_accounts_for_stc_feature_cost_when_support_ratio_zero():
    rows = [
        _base_row(
            method="HeSF-CAL-H6",
            method_family="hesf_cal_support",
            total_storage_ratio_vs_full_stc=0.04,
            macro_f1=0.90,
            accuracy=0.90,
        ),
        _base_row(
            method="STC-feature-cache-quantized-int8",
            method_family="stc_compressed",
            support_node_ratio=0.0,
            support_edge_ratio=0.0,
            feature_cache_size_ratio=0.25,
            path_channel_count_ratio=1.0,
            feature_cache_bytes=256,
            total_inference_storage_bytes=256,
            total_storage_ratio_vs_full_stc=0.25,
            macro_f1=0.88,
            accuracy=0.88,
        ),
    ]

    frontier = build_gate19_2_pareto(rows)
    stc = next(row for row in frontier if row["method"] == "STC-feature-cache-quantized-int8")

    assert stc["total_storage_ratio_vs_full_stc"] > 0.0
    assert stc["pareto_dominated_by"] == "HeSF-CAL-H6"
