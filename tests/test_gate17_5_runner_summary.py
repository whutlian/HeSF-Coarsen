from __future__ import annotations

import csv
import json

from experiments.scripts.run_gate17_5_h6_cluster_gating import (
    DEFAULT_METHODS,
    DIAGNOSTIC_ONLY_METHODS,
    GATE17_5_SINGLE_SEED_BY_DATASET,
    H6_CLUSTER_GATED_METHODS,
    H6_CONSTRUCTION_CONTROL_METHOD,
    H6_SELECTED_SET_CONTROL_METHOD,
    _h6_delta_alias_fields,
    parse_dataset_seeds,
)
from experiments.scripts.summarize_gate17_5 import summarize


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


def _row(method, dataset, ratio, macro, *, eligible=False, exact=True, diagnostic=False):
    return {
        "dataset": dataset,
        "seed": 23456 if dataset != "IMDB" else 45678,
        "method": method,
        "requested_support_ratio": ratio,
        "status": "success",
        "macro_f1": macro,
        "accuracy": macro + 0.01,
        "validation_macro_f1": macro - 0.01,
        "validation_accuracy": macro,
        "projected_macro_f1": macro,
        "transfer_macro_f1": macro - 0.02,
        "projected_vs_transfer_macro_gap": 0.02,
        "primary_eval_mode": "compressed_projected",
        "primary_task_metric_name": "projected_original_macro_f1",
        "node_budget_exact_match": exact,
        "represented_context_exact_or_bounded": True,
        "full_residual_upperbound": False,
        "eligible_for_main_decision": eligible,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "no_test_leakage": True,
        "diagnostic_only": diagnostic,
    }


def test_gate17_5_dataset_seeds_and_method_sets_match_prompt():
    pairs = parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])

    assert GATE17_5_SINGLE_SEED_BY_DATASET == {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
    assert pairs == [("ACM", 23456), ("DBLP", 23456), ("IMDB", 45678)]
    for method in [
        "full-graph-hettree-lite-tuned",
        "target-only-empty-support",
        "random-support-only",
        "H6-no-spec-support-only",
        "flatten-sum-support-only",
        "HeSF-SS-real-validation-budget-penalty-fill",
        "HeSF-SS-real-validation-H6-fill",
        "HeSF-SS-H6-cluster-validation-gated",
        "HeSF-SS-H6-cluster-validation-budget-penalty",
    ]:
        assert method in DEFAULT_METHODS
    assert H6_CONSTRUCTION_CONTROL_METHOD in DIAGNOSTIC_ONLY_METHODS
    assert H6_SELECTED_SET_CONTROL_METHOD in DIAGNOSTIC_ONLY_METHODS
    assert H6_CONSTRUCTION_CONTROL_METHOD not in H6_CLUSTER_GATED_METHODS


def test_gate17_5_h6_delta_alias_fields_match_prompt_column_names():
    fields = _h6_delta_alias_fields(
        {"edge_mass_l1_delta_vs_h6": 1.25},
        {"feature_mean_l2_delta_vs_h6": 0.5},
    )

    assert fields == {
        "edge_mass_l1_delta_vs_h6_by_relation": 1.25,
        "feature_mean_l2_delta_vs_h6_by_type": 0.5,
    }


def test_gate17_5_summary_reports_missing_dblp_exact_ratio_and_blocks_gate18(tmp_path):
    rows = []
    for dataset, baseline, method_macro in [
        ("ACM", 0.80, 0.80),
        ("DBLP", 0.70, 0.71),
        ("IMDB", 0.60, 0.61),
    ]:
        rows.append(_row("H6-no-spec-support-only", dataset, 0.3, baseline))
        rows.append(_row("flatten-sum-support-only", dataset, 0.3, baseline - 0.01))
        rows.append(_row("HeSF-SS-real-validation-neutral-fill", dataset, 0.3, method_macro, eligible=True))
    rows.extend(
        [
            _row("H6-no-spec-support-only", "DBLP", 0.7, 0.72),
            _row("flatten-sum-support-only", "DBLP", 0.7, 0.71),
            _row("HeSF-SS-real-validation-neutral-fill", "DBLP", 0.7, 0.73, eligible=True, exact=False),
            _row("HeSF-SS-H6-equivalence-control", "DBLP", 0.3, 0.70, eligible=False, diagnostic=True),
            _row("HeSF-SS-H6-cluster-validation-gated", "DBLP", 0.7, 0.715, eligible=True),
        ]
    )
    _write_rows(tmp_path / "gate17_5_raw_rows.csv", rows)
    diag = tmp_path / "diagnostics"
    _write_rows(
        diag / "gate17_5_h6_equivalence.csv",
        [
            {
                "dataset": "DBLP",
                "seed": 23456,
                "requested_support_ratio": 0.3,
                "mode": "construction",
                "construction_equivalence_pass": True,
                "macro_gap_vs_h6": 0.0,
                "tree_l2_delta_vs_h6": 0.0,
            }
        ],
    )

    result = summarize(tmp_path, tmp_path)

    assert result["best_eligible_method"] == "HeSF-SS-real-validation-neutral-fill"
    assert result["best_eligible_dblp_ratio_0_3_gap_macro"] == 0.01
    assert result["best_eligible_dblp_ratio_0_7_gap_macro"] is None
    assert result["best_eligible_dblp_missing_exact_ratios"] == [0.7]
    assert result["acm_used_for_success_evidence"] is False
    assert result["gate18_allowed"] is False
    assert "DBLP_0_7_EXACT_BUDGET_MISSING" in result["main_failure_reasons"]
    saved = json.loads((tmp_path / "gate17_5_result.json").read_text(encoding="utf-8"))
    assert saved["typedhash_note"] == "TypedHash skipped for Gate17.5 speed; Gate18 requires TypedHash."
