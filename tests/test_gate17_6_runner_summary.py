from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.scripts.run_gate17_6_accuracy_calibrated_h6_fill import (
    DEFAULT_METHODS,
    DIAGNOSTIC_ONLY_METHODS,
    GATE17_6_SINGLE_SEED_BY_DATASET,
    MAIN_CANDIDATE_METHODS,
    accuracy_objective_components,
    parse_dataset_seeds,
    per_class_audit_rows,
)
from experiments.scripts.summarize_gate17_6 import summarize


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(method: str, dataset: str, ratio: float, macro: float, accuracy: float, *, eligible: bool = False) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": 45678 if dataset == "IMDB" else 23456,
        "method": method,
        "requested_support_ratio": ratio,
        "requested_support_count": 100,
        "selected_support_count": 100,
        "status": "success",
        "macro_f1": macro,
        "accuracy": accuracy,
        "validation_macro_f1": macro - 0.01,
        "validation_accuracy": accuracy - 0.01,
        "validation_micro_f1_available": False,
        "primary_eval_mode": "compressed_projected",
        "node_budget_exact_match": True,
        "support_budget_exact_match": True,
        "effective_support_node_budget_pass": True,
        "represented_context_budget_pass": True,
        "represented_context_exact_or_bounded": True,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "no_test_leakage": True,
        "full_residual_upperbound": False,
        "diagnostic_only": not eligible,
        "eligible_for_main_decision": eligible,
    }


def _summary_rows(
    *,
    better_accuracy_method: str = "HeSF-SS-validation-H6-fill-acc0.50",
    bad_dblp_accuracy: bool = False,
    h6_fill_only_high: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["ACM", "DBLP", "IMDB"]:
        for ratio, base_macro, base_acc in [(0.30, 0.70, 0.80), (0.70, 0.72, 0.82)]:
            rows.append(_row("H6-no-spec-support-only", dataset, ratio, base_macro - 0.03, base_acc - 0.01))
            rows.append(_row("flatten-sum-support-only", dataset, ratio, base_macro - 0.02, base_acc - 0.02))
            rows.append(_row("TypedHash-ChebHeat-support-only", dataset, ratio, base_macro, base_acc))
            h6_delta = 0.04 if h6_fill_only_high else -0.04
            rows.append(_row("HeSF-SS-H6-fill-only", dataset, ratio, base_macro + h6_delta, base_acc + h6_delta, eligible=False))
            rows.append(_row("HeSF-SS-validation-H6-fill-acc0.25", dataset, ratio, base_macro + 0.012, base_acc - 0.012, eligible=True))
            acc_gap = -0.006 if bad_dblp_accuracy and dataset == "DBLP" and ratio == 0.30 else -0.004
            rows.append(_row(better_accuracy_method, dataset, ratio, base_macro + 0.010, base_acc + acc_gap, eligible=True))
    return rows


def test_gate17_6_dataset_seeds_and_method_sets_match_prompt():
    pairs = parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])

    assert GATE17_6_SINGLE_SEED_BY_DATASET == {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
    assert pairs == [("ACM", 23456), ("DBLP", 23456), ("IMDB", 45678)]
    for method in [
        "full-graph-hettree-lite-tuned",
        "target-only-empty-support",
        "random-support-only",
        "H6-no-spec-support-only",
        "flatten-sum-support-only",
        "TypedHash-ChebHeat-support-only",
        "HeSF-SS-validation-only-neutral-fill",
        "HeSF-SS-validation-H6-fill",
        "HeSF-SS-validation-H6-fill-acc0.25",
        "HeSF-SS-validation-H6-fill-acc0.50",
        "HeSF-SS-validation-H6-fill-acc1.00",
        "HeSF-SS-H6-fill-only",
        "HeSF-SS-random-fill-after-validation",
    ]:
        assert method in DEFAULT_METHODS
    assert "HeSF-SS-validation-H6-fill-acc0.50" in MAIN_CANDIDATE_METHODS
    assert "HeSF-SS-H6-fill-only" in MAIN_CANDIDATE_METHODS
    assert "HeSF-SS-H6-equivalence-control" in DIAGNOSTIC_ONLY_METHODS
    assert "HeSF-SS-H6-cluster-validation-coverage-gated" in DIAGNOSTIC_ONLY_METHODS


def test_accuracy_objective_components_falls_back_to_accuracy_when_micro_missing():
    components = accuracy_objective_components(
        validation_macro_f1=0.40,
        validation_accuracy=0.80,
        validation_micro_f1=None,
        requested_support_count=10,
        selected_support_count=8,
        baseline_num_predicted_classes=4,
        num_predicted_classes=3,
        alpha_accuracy=0.50,
        delta_micro=0.25,
        beta_underfill=0.10,
        gamma_class_collapse=0.05,
    )

    assert components["validation_micro_f1_available"] is False
    assert components["validation_micro_f1"] == 0.80
    assert components["underfill_penalty"] == pytest.approx(0.20)
    assert components["class_collapse_penalty"] == pytest.approx(0.25)
    assert components["score_macro_component"] == pytest.approx(0.40)
    assert components["score_accuracy_component"] == pytest.approx(0.40)
    assert components["score_micro_component"] == pytest.approx(0.20)
    assert components["score_underfill_component"] == pytest.approx(-0.02)
    assert components["score_class_collapse_component"] == pytest.approx(-0.0125)
    assert components["score_total"] == pytest.approx(0.9675)


def test_per_class_audit_rows_emit_metrics_and_confusion_long_format():
    per_class, confusion = per_class_audit_rows(
        dataset="DBLP",
        seed=23456,
        method="HeSF-SS-validation-H6-fill",
        ratio=0.30,
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        selected_support_labels=[0, 1, 1],
        baseline_per_class={0: {"precision": 1.0, "recall": 1.0, "f1": 1.0}, 1: {"precision": 0.5, "recall": 1.0, "f1": 2 / 3}},
    )

    class_zero = next(row for row in per_class if row["class_id"] == 0)
    assert class_zero["test_label_count"] == 2
    assert class_zero["predicted_count"] == 1
    assert class_zero["precision"] == pytest.approx(1.0)
    assert class_zero["recall"] == pytest.approx(0.5)
    assert class_zero["delta_recall_vs_best_strong"] == pytest.approx(-0.5)
    assert {tuple(row[key] for key in ["true_class", "pred_class", "count"]) for row in confusion} == {
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 2),
    }


def test_gate17_6_summary_excludes_h6_fill_only_and_uses_dblp_accuracy_tiebreaker(tmp_path):
    _write_rows(tmp_path / "gate17_6_raw_rows.csv", _summary_rows())

    result = summarize(tmp_path, tmp_path)

    assert result["typedhash_included"] is True
    assert result["best_eligible_method"] == "HeSF-SS-validation-H6-fill-acc0.50"
    assert result["best_eligible_method"] != "HeSF-SS-H6-fill-only"
    assert result["best_eligible_dblp_ratio_0_3_gap_macro"] == pytest.approx(0.01)
    assert result["best_eligible_dblp_ratio_0_3_gap_accuracy"] == pytest.approx(-0.004)
    assert result["h6_fill_only_beats_validation_flag"] is True
    assert (tmp_path / "gate17_6_result.json").exists()
    assert (tmp_path / "gate17_6_final_report.md").exists()
    assert (tmp_path / "diagnostics" / "gate17_6_header_normalization_check.csv").exists()


def test_gate17_6_summary_blocks_gate18_on_dblp_accuracy_threshold(tmp_path):
    _write_rows(tmp_path / "gate17_6_raw_rows.csv", _summary_rows(bad_dblp_accuracy=True, h6_fill_only_high=False))

    result = summarize(tmp_path, tmp_path)

    assert result["best_eligible_method"] == "HeSF-SS-validation-H6-fill-acc0.50"
    assert result["best_eligible_dblp_ratio_0_3_gap_accuracy"] == pytest.approx(-0.006)
    assert result["accuracy_blocker"] is True
    assert result["accuracy_pass"] is False
    assert result["gate18_allowed"] is False
    assert result["decision"] == "CONTINUE_GATE17_X_ACCURACY_CALIBRATION"
    saved = json.loads((tmp_path / "gate17_6_result.json").read_text(encoding="utf-8"))
    assert saved["primary_eval_mode"] == "compressed_projected"
