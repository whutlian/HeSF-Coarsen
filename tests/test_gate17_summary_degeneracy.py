import csv
import json

from experiments.scripts.summarize_gate17_1 import summarize


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


def test_gate17_1_tied_methods_do_not_emit_best_method(tmp_path):
    rows = []
    for method in [
        "H6-no-spec-support-only",
        "random-support-only",
        "HeSF-SS-real-occlusion-block",
        "HeSF-SS-real-validation-block-greedy",
    ]:
        rows.append(
            {
                "dataset": "ACM",
                "seed": 12345,
                "method": method,
                "requested_support_ratio": 0.3,
                "requested_support_count": 3,
                "support_budget_exact_match": True,
                "status": "success",
                "macro_f1": 0.25,
                "accuracy": 0.30,
                "validation_macro_f1": 0.20,
                "validation_accuracy": 0.30,
                "projected_macro_f1": 0.25,
                "transfer_macro_f1": 0.25,
                "primary_eval_mode": "compressed_projected",
            }
        )
    _write_rows(tmp_path / "gate17_1_raw_rows.csv", rows)
    _write_rows(
        tmp_path / "diag" / "semantic_tree_delta.csv",
        [
            {
                "dataset": "ACM",
                "seed": 12345,
                "method": "HeSF-SS-real-occlusion-block",
                "tree_tensor_l2_delta_vs_full": 0.0,
                "target_path_feature_changed_fraction": 0.0,
            }
        ],
    )

    result = summarize(tmp_path, tmp_path / "main", tmp_path / "diag")

    assert result["all_methods_tied"] is True
    assert result["best_validation_selected_method"] is None
    assert result["decision"] in {"FAIL_SUPPORT_BLIND_EVALUATOR", "CODEPATH_SMOKE_PASS_METRIC_DEGENERATE"}
    assert result["decision"] != "PARTIAL_DBLP_BLOCKER"

    saved = json.loads((tmp_path / "main" / "result.json").read_text(encoding="utf-8"))
    assert saved["best_validation_selected_method"] is None
    assert (tmp_path / "main" / "gate17_1_validation_selected_by_method.csv").exists()
    assert (tmp_path / "diag" / "evaluator_metric_nunique.csv").exists()
