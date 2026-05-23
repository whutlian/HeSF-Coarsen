from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.scripts.run_gate19_cost_normalized_stc import leakage_audit_row
from experiments.scripts.summarize_gate19 import build_cost_normalized_pareto, normalize_header, read_csv
from hesf_coarsen.task_first.costs.accounting import CompressionCost, assert_cost_finite, compute_total_storage_ratio
from hesf_coarsen.task_first.feature_condensation.distillation import teacher_kl_diagnostics


def test_cost_accounting_stc_nonzero_when_support_ratio_zero():
    """STC methods may have support ratio 0, but feature cache cost must be > 0."""
    cost = CompressionCost(
        method="STC-path-prune-energy",
        dataset="DBLP",
        seed=23456,
        requested_budget=0.5,
        support_node_count=0,
        feature_cache_elements=50,
        feature_cache_bytes=200,
        full_feature_cache_elements=100,
        full_feature_cache_bytes=400,
    )

    computed = compute_total_storage_ratio(cost)
    assert_cost_finite(computed)

    assert computed.support_node_ratio == 0.0
    assert computed.feature_cache_size_ratio == pytest.approx(0.5)
    assert computed.total_storage_bytes > 0
    assert computed.total_storage_ratio_vs_full_stc > 0.0


def test_full_stc_baseline_ratio_one():
    """Full-STC baseline must have feature_cache_size_ratio == 1 and path_channel_count_ratio == 1."""
    cost = CompressionCost(
        method="Full-STC-MLP",
        dataset="DBLP",
        seed=23456,
        requested_budget=1.0,
        path_channel_count=2,
        feature_cache_elements=100,
        feature_cache_bytes=400,
        model_param_bytes=80,
        full_path_channel_count=2,
        full_feature_cache_elements=100,
        full_feature_cache_bytes=400,
        full_model_param_bytes=80,
    )

    computed = compute_total_storage_ratio(cost)

    assert computed.feature_cache_size_ratio == pytest.approx(1.0)
    assert computed.path_channel_count_ratio == pytest.approx(1.0)
    assert computed.total_storage_ratio_vs_full_stc == pytest.approx(1.0)


def test_compressed_stc_ratio_below_one():
    """Compressed STC method at budget 0.5 must not report feature_cache_size_ratio == 1 unless marked full-cache ceiling."""
    cost = CompressionCost(
        method="STC-feature-cache-MLP-compressed",
        dataset="DBLP",
        seed=23456,
        requested_budget=0.5,
        path_channel_count=1,
        feature_cache_elements=50,
        feature_cache_bytes=200,
        full_path_channel_count=2,
        full_feature_cache_elements=100,
        full_feature_cache_bytes=400,
    )

    computed = compute_total_storage_ratio(cost)

    assert computed.feature_cache_size_ratio < 1.0
    assert computed.path_channel_count_ratio < 1.0
    assert computed.total_storage_ratio_vs_full_stc <= 0.5


def test_teacher_kl_not_self_reference():
    """If teacher logits are unavailable, teacher_kl_status must be unavailable, not zero/self-reference."""
    student = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    unavailable = teacher_kl_diagnostics(None, student)
    same = teacher_kl_diagnostics(student, student, teacher_source="student_self")

    assert unavailable["teacher_kl_status"] == "unavailable"
    assert unavailable["teacher_student_kl"] == ""
    assert same["teacher_kl_status"] == "self_reference"


def test_no_test_leakage_in_path_selection_and_calibration():
    """Path/channel selection and calibration must not inspect test labels."""
    row = leakage_audit_row(
        method="STC-path-channel-hard-gate-logit-calibrated",
        uses_train_labels=True,
        uses_val_labels=True,
        uses_test_labels_before_final_eval=False,
        calibration_split="val",
        path_selection_split="train_val",
        teacher_training_split="train",
        student_training_split="train",
    )

    assert row["leakage_pass"] is True
    assert row["uses_test_labels_before_final_eval"] is False
    assert row["calibration_split"] == "val"


def test_header_normalization_gate19(tmp_path: Path):
    """BOM/quoted dataset headers normalize to dataset."""
    path = tmp_path / "rows.csv"
    path.write_text('"\\ufeffdataset",method\\nDBLP,STC-path-prune-energy\\n', encoding="utf-8")

    rows = read_csv(path)

    assert normalize_header('"\\ufeffdataset"') == "dataset"
    assert rows[0]["dataset"] == "DBLP"


def test_pareto_uses_total_storage_ratio():
    """Gate19 Pareto frontier must not use actual_support_ratio alone for STC methods."""
    rows = [
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "STC-small",
            "method_family": "stc_compressed",
            "status": "success",
            "actual_support_ratio": "0.0",
            "total_storage_ratio_vs_full_stc": "0.5",
            "macro_f1": "0.80",
            "accuracy": "0.80",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
        {
            "dataset": "DBLP",
            "seed": "23456",
            "method": "STC-large",
            "method_family": "stc_compressed",
            "status": "success",
            "actual_support_ratio": "0.0",
            "total_storage_ratio_vs_full_stc": "0.7",
            "macro_f1": "0.82",
            "accuracy": "0.82",
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": "true",
        },
    ]

    frontier = build_cost_normalized_pareto(rows)
    methods = {row["method"] for row in frontier}

    assert methods == {"STC-small", "STC-large"}
    assert {row["cost_axis_used"] for row in frontier} == {"total_storage_ratio_vs_full_stc"}
