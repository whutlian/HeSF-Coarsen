from __future__ import annotations

import csv
import json

from experiments.scripts.summarize_gate17_4 import summarize


def _write_rows(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_row(method, ratio, macro, *, dataset="DBLP", eligible=False, upperbound=False):
    return {
        "dataset": dataset,
        "seed": 23456,
        "method": method,
        "requested_support_ratio": ratio,
        "status": "success",
        "macro_f1": macro,
        "accuracy": macro + 0.01,
        "validation_macro_f1": macro - 0.01,
        "primary_eval_mode": "compressed_projected",
        "node_budget_exact_match": True,
        "represented_context_exact_or_bounded": True,
        "full_residual_upperbound": upperbound,
        "eligible_for_main_decision": eligible,
        "no_test_leakage": True,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "occlusion_task_signal_pass": True,
        "validation_signal_pass": True,
    }


def test_gate17_4_summary_uses_best_eligible_method_before_dblp_gap(tmp_path):
    rows = [
        _base_row("H6-no-spec-support-only", 0.3, 0.8118),
        _base_row("H6-no-spec-support-only", 0.7, 0.7904),
        _base_row("flatten-sum-support-only", 0.3, 0.80),
        _base_row("flatten-sum-support-only", 0.7, 0.78),
        _base_row("HeSF-SS-real-validation-neutral-fill", 0.3, 0.7957, eligible=True),
        _base_row("HeSF-SS-real-validation-neutral-fill", 0.7, 0.7925, eligible=True),
        _base_row("HeSF-SS-sensitivity-selection-only", 0.3, 0.6938, eligible=True),
        _base_row("HeSF-SS-sensitivity-selection-only", 0.7, 0.7269, eligible=True),
        _base_row("HeSF-SS-full-residual-prototype-upperbound", 0.3, 0.99, eligible=False, upperbound=True),
        _base_row("HeSF-SS-H6-equivalence-control", 0.3, 0.8118, eligible=True),
    ]
    _write_rows(tmp_path / "gate17_4_raw_rows.csv", rows)
    _write_rows(
        tmp_path / "diagnostics" / "gate17_4_h6_equivalence.csv",
        [
            {
                "dataset": "DBLP",
                "seed": 23456,
                "requested_support_ratio": 0.3,
                "mode": "construction",
                "macro_gap_vs_h6": 0.0,
                "tree_l2_delta_vs_h6": 0.0,
                "construction_equivalence_pass": True,
            }
        ],
    )

    result = summarize(tmp_path, tmp_path)

    assert result["best_eligible_method"] == "HeSF-SS-real-validation-neutral-fill"
    assert result["best_eligible_dblp_ratio_0_3_gap"] == -0.0161
    assert result["best_eligible_dblp_ratio_0_7_gap"] == 0.0021
    assert result["best_eligible_dblp_exact_delta_macro"] == -0.007
    assert "FAIL_DBLP_REAL_FEEDBACK_GAP" in result["main_failure_reasons"]
    assert "PASS_H6_EQUIVALENCE_CONTROL" in result["main_failure_reasons"]
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["acm_support_saturated"] is False


def test_gate17_4_summary_excludes_upperbound_and_baselines_from_best_method(tmp_path):
    rows = [
        _base_row("H6-no-spec-support-only", 0.3, 0.70),
        _base_row("flatten-sum-support-only", 0.3, 0.71),
        _base_row("HeSF-SS-full-residual-prototype-upperbound", 0.3, 0.99, eligible=False, upperbound=True),
        _base_row("HeSF-SS-H6-selected-set-control", 0.3, 0.90, eligible=False),
        _base_row("HeSF-SS-real-occlusion-neutral-fill", 0.3, 0.72, eligible=True),
    ]
    _write_rows(tmp_path / "gate17_4_raw_rows.csv", rows)
    _write_rows(tmp_path / "diagnostics" / "gate17_4_h6_equivalence.csv", [])

    result = summarize(tmp_path, tmp_path)

    assert result["best_eligible_method"] == "HeSF-SS-real-occlusion-neutral-fill"
    assert result["eligible_method_count"] == 1


def test_gate17_4_summary_reads_large_raw_diagnostic_fields(tmp_path):
    rows = [
        _base_row("H6-no-spec-support-only", 0.3, 0.70),
        {
            **_base_row("HeSF-SS-real-validation-neutral-fill", 0.3, 0.72, eligible=True),
            "feature_mean_by_type": "x" * 150_000,
        },
    ]
    _write_rows(tmp_path / "gate17_4_raw_rows.csv", rows)
    _write_rows(tmp_path / "diagnostics" / "gate17_4_h6_equivalence.csv", [])

    result = summarize(tmp_path, tmp_path)

    assert result["best_eligible_method"] == "HeSF-SS-real-validation-neutral-fill"
