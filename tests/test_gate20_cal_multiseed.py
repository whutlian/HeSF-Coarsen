from __future__ import annotations

from experiments.scripts.summarize_gate20_cal import (
    build_gate20_pareto,
    summarize_rows,
    validation_selected_rows,
)
from experiments.scripts.run_gate20_cal_multiseed import make_repro_metadata
from experiments.scripts.run_gate20_cal_multiseed import (
    fit_gate20_calibration_from_logits,
    select_best_support_candidate,
    with_repro_metadata,
)


def _row(**extra):
    row = {
        "stage": "Gate20-CAL",
        "dataset": "DBLP",
        "seed": 12345,
        "method": "HeSF-CAL-best-support",
        "method_family": "hesf_cal",
        "ratio": 0.30,
        "status": "success",
        "diagnostic_only": False,
        "eligible_for_main_decision": True,
        "primary_eval_mode": "compressed_projected",
        "no_test_leakage": True,
        "calibration_uses_test_labels": False,
        "selector_uses_test_labels": False,
        "accuracy": 0.90,
        "macro_f1": 0.89,
        "validation_accuracy": 0.91,
        "validation_macro_f1": 0.90,
        "uncalibrated_accuracy": 0.87,
        "uncalibrated_macro_f1": 0.86,
        "total_storage_ratio_vs_full_stc": 0.03,
        "total_storage_ratio_vs_full_graph": 0.02,
    }
    row.update(extra)
    return row


def test_validation_selection_prefers_validation_not_test_accuracy():
    rows = [
        _row(method="HeSF-CAL-H6", ratio=0.30, validation_accuracy=0.91, validation_macro_f1=0.90, accuracy=0.88),
        _row(method="HeSF-CAL-H6", ratio=0.50, validation_accuracy=0.80, validation_macro_f1=0.80, accuracy=0.95),
    ]

    selected = validation_selected_rows(rows)

    assert selected[0]["ratio"] == 0.30
    assert selected[0]["accuracy"] == 0.88
    assert selected[0]["test_oracle_used_for_selection"] is False


def test_summary_binds_nested_stats_to_exact_best_method_ratio():
    rows = [
        _row(method="HeSF-CAL-best-support", ratio=0.30, accuracy=0.90, macro_f1=0.89, validation_accuracy=0.91),
        _row(method="HeSF-CAL-H6", ratio=0.50, accuracy=0.93, macro_f1=0.92, validation_accuracy=0.80),
        _row(method="TypedHash-ChebHeat-support-only", method_family="support_baseline", ratio=0.30, accuracy=0.81, macro_f1=0.80),
    ]
    nested = [
        {
            "dataset": "DBLP",
            "method": "HeSF-CAL-best-support",
            "ratio": 0.30,
            "nested_accuracy_mean": 0.90,
            "nested_accuracy_std": 0.014,
            "nested_macro_mean": 0.89,
            "nested_macro_std": 0.013,
            "calibration_constraint_satisfied_rate": 1.0,
            "temperature_mean": 1.0,
            "temperature_std": 0.1,
            "class_bias_l2_mean": 0.2,
            "class_bias_l2_std": 0.03,
        },
        {
            "dataset": "DBLP",
            "method": "HeSF-CAL-H6",
            "ratio": 0.50,
            "nested_accuracy_mean": 0.93,
            "nested_accuracy_std": 0.0,
            "nested_macro_mean": 0.92,
            "nested_macro_std": 0.0,
            "calibration_constraint_satisfied_rate": 1.0,
        },
    ]

    result = summarize_rows(rows, nested_rows=nested, quality_rows=[], per_class_present=True, confusion_present=True)

    assert result["best_method"] == "HeSF-CAL-best-support"
    assert result["best_method_ratio"] == 0.30
    assert result["best_method_nested_accuracy_std"] == 0.014
    assert result["test_oracle_used_for_decision"] is False


def test_diagnostic_methods_are_excluded_from_main_decision():
    rows = [
        _row(method="HeSF-CAL-best-support", ratio=0.30, accuracy=0.90, macro_f1=0.89, validation_accuracy=0.91),
        _row(
            method="HeSF-CAL-LogitEnsemble",
            method_family="hesf_cal_ensemble",
            ratio=0.30,
            accuracy=0.99,
            macro_f1=0.99,
            validation_accuracy=0.99,
            validation_macro_f1=0.99,
            diagnostic_only=True,
            eligible_for_main_decision=False,
            exclude_from_nested_gate=True,
        ),
    ]

    result = summarize_rows(rows, nested_rows=[], quality_rows=[], per_class_present=True, confusion_present=True)

    assert result["best_method"] == "HeSF-CAL-best-support"
    assert result["best_validation_selected_method"] == "HeSF-CAL-best-support"


def test_pareto_marks_dominated_rows_and_keeps_main_only_frontier():
    rows = [
        _row(method="HeSF-CAL-H6", ratio=0.30, accuracy=0.90, macro_f1=0.89, total_storage_ratio_vs_full_stc=0.03),
        _row(method="HeSF-CAL-flatten", ratio=0.30, accuracy=0.88, macro_f1=0.87, total_storage_ratio_vs_full_stc=0.05),
    ]

    frontier = build_gate20_pareto(rows)
    dominated = next(row for row in frontier if row["method"] == "HeSF-CAL-flatten")

    assert dominated["pareto_dominated_by"] == "HeSF-CAL-H6"


def test_repro_metadata_contains_required_fields():
    meta = make_repro_metadata(
        script_name="experiments/scripts/run_gate20_cal_multiseed.py",
        config={"datasets": ["DBLP"], "seeds": [12345]},
        run_id="gate20-test",
    )

    assert {"git_commit", "script_name", "config_hash", "run_id", "timestamp"} <= set(meta)
    assert meta["run_id"] == "gate20-test"


def test_gate20_calibration_logs_candidates_and_never_uses_test_labels():
    val_logits = [[3.0, 0.0], [0.0, 3.0], [2.5, 0.0], [0.0, 2.5]]
    val_labels = [0, 1, 0, 1]
    test_logits = [[2.0, 0.0], [0.0, 2.0]]
    test_labels = [0, 1]

    calibrated, diag, nested_rows, quality_rows, candidate_rows = fit_gate20_calibration_from_logits(
        val_logits,
        val_labels,
        test_logits,
        test_labels,
        dataset="DBLP",
        seed=12345,
        method="HeSF-CAL-H6",
        ratio=0.30,
        split_seeds=[11],
    )

    assert calibrated["accuracy"] == 1.0
    assert diag["calibration_uses_test_labels"] is False
    assert nested_rows[0]["split_seed"] == 11
    assert quality_rows[0]["calibrated_accuracy"] == 1.0
    assert len(candidate_rows) > 1
    assert all(row["uses_test_labels"] is False for row in candidate_rows)


def test_best_support_selection_uses_nested_validation_not_test():
    candidates = [
        _row(method="HeSF-CAL-H6", source_method="H6-no-spec-support-only", validation_accuracy=0.91, validation_macro_f1=0.90, accuracy=0.88),
        _row(method="HeSF-CAL-flatten", source_method="flatten-sum-support-only", validation_accuracy=0.80, validation_macro_f1=0.80, accuracy=0.95),
    ]

    selected = select_best_support_candidate(candidates, best_uncalibrated_support_val_macro=0.89)

    assert selected["method"] == "HeSF-CAL-H6"
    assert selected["selection_uses_test_labels"] is False


def test_with_repro_metadata_adds_required_fields_without_overwriting_row():
    meta = {"git_commit": "abc", "script_name": "runner.py", "config_hash": "def", "run_id": "rid", "timestamp": "now"}
    row = with_repro_metadata({"dataset": "DBLP", "method": "HeSF-CAL-H6"}, meta)

    assert row["dataset"] == "DBLP"
    assert row["git_commit"] == "abc"
    assert row["run_id"] == "rid"
