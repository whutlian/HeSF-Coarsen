import csv
import json

from experiments.scripts.summarize_gate17 import summarize


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


def test_gate17_summary_uses_method_level_mean_not_single_max(tmp_path):
    rows = [
        {
            "dataset": "ACM",
            "seed": 1,
            "method": "H6-no-spec-support-only",
            "requested_support_ratio": 0.3,
            "requested_support_count": 3,
            "support_budget_exact_match": True,
            "status": "success",
            "macro_f1": 0.50,
            "accuracy": 0.50,
            "validation_macro_f1": 0.50,
            "validation_accuracy": 0.50,
            "primary_eval_mode": "compressed_projected",
        },
        {
            "dataset": "ACM",
            "seed": 1,
            "method": "HeSF-SS-real-occlusion-block",
            "requested_support_ratio": 0.3,
            "requested_support_count": 3,
            "support_budget_exact_match": True,
            "status": "success",
            "macro_f1": 0.90,
            "accuracy": 0.90,
            "validation_macro_f1": 0.90,
            "validation_accuracy": 0.90,
            "primary_eval_mode": "compressed_projected",
            "occlusion_trial_count": 2,
        },
        {
            "dataset": "DBLP",
            "seed": 1,
            "method": "flatten-sum-support-only",
            "requested_support_ratio": 0.3,
            "requested_support_count": 3,
            "support_budget_exact_match": True,
            "status": "success",
            "macro_f1": 0.50,
            "accuracy": 0.50,
            "validation_macro_f1": 0.50,
            "validation_accuracy": 0.50,
            "primary_eval_mode": "compressed_projected",
        },
        {
            "dataset": "DBLP",
            "seed": 1,
            "method": "HeSF-SS-real-occlusion-block",
            "requested_support_ratio": 0.3,
            "requested_support_count": 3,
            "support_budget_exact_match": True,
            "status": "success",
            "macro_f1": 0.30,
            "accuracy": 0.30,
            "validation_macro_f1": 0.30,
            "validation_accuracy": 0.30,
            "primary_eval_mode": "compressed_projected",
            "occlusion_trial_count": 2,
        },
    ]
    _write_rows(tmp_path / "gate17_raw_rows.csv", rows)

    result = summarize(tmp_path, tmp_path / "tables")

    assert result["best_validation_selected_method"] == "HeSF-SS-real-occlusion-block"
    assert result["best_validation_selected_macro_f1_mean"] == 0.6
    assert result["best_single_run_macro_f1"] == 0.9
    assert result["best_single_run_method"] == "HeSF-SS-real-occlusion-block"
    assert result["occlusion_trial_count_total"] == 4

    saved = json.loads((tmp_path / "tables" / "result.json").read_text(encoding="utf-8"))
    assert saved["best_validation_selected_macro_f1_mean"] == 0.6
    assert (tmp_path / "tables" / "gate17_validation_selected_by_method.csv").exists()


def test_gate17_validation_selection_tie_breaks_by_lower_support_ratio(tmp_path):
    rows = [
        {
            "dataset": "ACM",
            "seed": 1,
            "method": "HeSF-SS-real-validation-block-greedy",
            "requested_support_ratio": 0.7,
            "requested_support_count": 7,
            "support_budget_exact_match": True,
            "status": "success",
            "macro_f1": 0.40,
            "accuracy": 0.40,
            "validation_macro_f1": 0.60,
            "validation_accuracy": 0.50,
            "primary_eval_mode": "compressed_projected",
        },
        {
            "dataset": "ACM",
            "seed": 1,
            "method": "HeSF-SS-real-validation-block-greedy",
            "requested_support_ratio": 0.3,
            "requested_support_count": 3,
            "support_budget_exact_match": True,
            "status": "success",
            "macro_f1": 0.55,
            "accuracy": 0.55,
            "validation_macro_f1": 0.60,
            "validation_accuracy": 0.50,
            "primary_eval_mode": "compressed_projected",
        },
    ]
    _write_rows(tmp_path / "gate17_raw_rows.csv", rows)

    summarize(tmp_path, tmp_path / "tables")
    selected = list(csv.DictReader((tmp_path / "tables" / "gate17_validation_selected_by_dataset.csv").open(encoding="utf-8")))

    assert selected[0]["requested_support_ratio"] == "0.3"
