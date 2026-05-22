from __future__ import annotations

import csv
import json

from experiments.scripts.summarize_gate17 import (
    assert_dataset_integrity,
    normalize_dataset_value,
    normalize_header,
    read_csv_normalized,
    validation_selected,
)
from experiments.scripts.summarize_gate17_4 import summarize as summarize_gate17_4


def _write_rows_with_header(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _gate17_4_row(method, dataset, ratio, macro, *, eligible=False, exact=True):
    return {
        '\ufeff"dataset"': dataset,
        "seed": 23456 if dataset != "IMDB" else 45678,
        "method": method,
        "requested_support_ratio": ratio,
        "status": "success",
        "macro_f1": macro,
        "accuracy": macro + 0.01,
        "validation_macro_f1": macro - 0.01,
        "primary_eval_mode": "compressed_projected",
        "node_budget_exact_match": exact,
        "represented_context_exact_or_bounded": True,
        "full_residual_upperbound": False,
        "eligible_for_main_decision": eligible,
        "no_test_leakage": True,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "validation_signal_pass": True,
        "occlusion_task_signal_pass": False,
    }


def test_normalized_reader_handles_bom_quoted_dataset_and_same_seed_grouping(tmp_path):
    path = tmp_path / "gate17_4_raw_rows.csv"
    rows = [
        _gate17_4_row("HeSF-SS-real-validation-neutral-fill", "ACM", 0.3, 0.80, eligible=True),
        _gate17_4_row("HeSF-SS-real-validation-neutral-fill", "DBLP", 0.3, 0.70, eligible=True),
        _gate17_4_row("HeSF-SS-real-validation-neutral-fill", "IMDB", 0.3, 0.60, eligible=True),
    ]
    _write_rows_with_header(path, list(rows[0].keys()), rows)

    parsed = read_csv_normalized(path)
    assert normalize_header('\ufeff"dataset"') == "dataset"
    assert normalize_dataset_value('"DBLP"') == "DBLP"
    assert_dataset_integrity(parsed, expected={"ACM", "DBLP", "IMDB"})

    selected = validation_selected(parsed)
    assert len(selected) == 3
    assert {row["dataset"] for row in selected} == {"ACM", "DBLP", "IMDB"}


def test_gate17_4_corrected_summary_writes_corrected_outputs_and_nonblank_dataset_gaps(tmp_path):
    input_dir = tmp_path / "gate17_4_h6_equivalence"
    rows = []
    for dataset, baseline_macro, method_macro in [
        ("ACM", 0.80, 0.80),
        ("DBLP", 0.70, 0.71),
        ("IMDB", 0.60, 0.61),
    ]:
        rows.append(_gate17_4_row("H6-no-spec-support-only", dataset, 0.3, baseline_macro))
        rows.append(_gate17_4_row("flatten-sum-support-only", dataset, 0.3, baseline_macro - 0.01))
        rows.append(_gate17_4_row("HeSF-SS-real-validation-neutral-fill", dataset, 0.3, method_macro, eligible=True))
    rows.append(_gate17_4_row("H6-no-spec-support-only", "DBLP", 0.7, 0.72))
    rows.append(_gate17_4_row("flatten-sum-support-only", "DBLP", 0.7, 0.71))
    rows.append(_gate17_4_row("HeSF-SS-real-validation-neutral-fill", "DBLP", 0.7, 0.73, eligible=True, exact=False))
    _write_rows_with_header(input_dir / "gate17_4_raw_rows.csv", list(rows[0].keys()), rows)
    _write_rows_with_header(
        input_dir / "diagnostics" / "gate17_4_h6_equivalence.csv",
        ["dataset", "seed", "requested_support_ratio", "mode", "macro_gap_vs_h6", "tree_l2_delta_vs_h6", "construction_equivalence_pass"],
        [
            {
                "dataset": dataset,
                "seed": 23456 if dataset != "IMDB" else 45678,
                "requested_support_ratio": 0.3,
                "mode": "construction",
                "macro_gap_vs_h6": 0.0,
                "tree_l2_delta_vs_h6": 0.0,
                "construction_equivalence_pass": True,
            }
            for dataset in ["ACM", "DBLP", "IMDB"]
        ],
    )
    output_dir = tmp_path / "gate17_4_h6_equivalence_corrected"

    result = summarize_gate17_4(input_dir, output_dir)

    assert result["best_eligible_method"] == "HeSF-SS-real-validation-neutral-fill"
    assert result["best_eligible_dblp_ratio_0_3_gap"] == 0.01
    assert result["best_eligible_dblp_ratio_0_7_gap"] is None
    assert result["gate18_allowed"] is False
    assert (output_dir / "gate17_4_validation_selected_by_method_corrected.csv").exists()
    assert (output_dir / "gate17_4_exact_budget_paired_gaps_corrected.csv").exists()
    assert (output_dir / "gate17_4_by_dataset_selected_corrected.csv").exists()
    assert (output_dir / "gate17_4_result_corrected.json").exists()
    assert (output_dir / "gate17_4_decision_corrected.md").exists()
    gaps = read_csv_normalized(output_dir / "gate17_4_exact_budget_paired_gaps_corrected.csv")
    assert gaps
    assert all(row["dataset"] for row in gaps)
    saved = json.loads((output_dir / "gate17_4_result_corrected.json").read_text(encoding="utf-8"))
    assert saved["best_eligible_dblp_ratio_0_7_gap"] is None
