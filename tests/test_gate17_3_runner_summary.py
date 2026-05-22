from __future__ import annotations

import csv
import json

import numpy as np

from experiments.scripts.run_gate17_3_lossy_prototype_feedback import (
    DEFAULT_METHODS,
    DIAGNOSTIC_ONLY_METHODS,
    GATE17_3_SINGLE_SEED_BY_DATASET,
    _h6_seed_nodes_for_method,
    _support_representatives,
    parse_dataset_seeds,
)
from experiments.scripts.summarize_gate17_3 import summarize


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


def test_gate17_3_dataset_seeds_are_exact_pairs_not_cartesian_product():
    pairs = parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])

    assert GATE17_3_SINGLE_SEED_BY_DATASET == {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
    assert pairs == [("ACM", 23456), ("DBLP", 23456), ("IMDB", 45678)]
    assert "HeSF-SS-real-occlusion-lossy-prototype" in DEFAULT_METHODS
    assert "HeSF-SS-full-residual-prototype-upperbound" in DIAGNOSTIC_ONLY_METHODS


def test_gate17_3_h6_seed_set_is_capped_to_requested_budget():
    class Graph:
        node_type = np.asarray([0, 1, 1, 1, 1, 1], dtype=np.int32)

    assignment = np.asarray([0, 10, 10, 11, 12, 12], dtype=np.int64)

    reps = _support_representatives(Graph(), assignment, target_type=0, max_nodes=2)
    lossy = _h6_seed_nodes_for_method(
        reps,
        support_count=5,
        ratio=0.40,
        method="HeSF-SS-H6-seeded-lossy-prototype",
        prototype_budget_fraction=0.10,
    )

    assert len(reps) == 2
    assert set(reps.tolist()).issubset({1, 3, 4})
    assert len(lossy) == 1


def test_gate17_3_summary_excludes_upperbound_and_reports_failure_priority(tmp_path):
    rows = [
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "H6-no-spec-support-only",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.70,
            "accuracy": 0.70,
            "validation_macro_f1": 0.70,
            "node_budget_exact_match": True,
            "support_budget_exact_match": True,
            "no_test_leakage": True,
        },
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "HeSF-SS-real-occlusion-lossy-prototype",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.69,
            "accuracy": 0.69,
            "validation_macro_f1": 0.69,
            "node_budget_exact_match": True,
            "represented_context_exact_or_bounded": False,
            "represented_context_ratio": 0.9,
            "eligible_for_main_decision": False,
            "occlusion_task_signal_pass": True,
            "validation_signal_pass": True,
            "prototype_saturation_rate": 0.1,
            "candidate_allclose_to_full": False,
            "no_test_leakage": True,
        },
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "HeSF-SS-full-residual-prototype-upperbound",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.99,
            "accuracy": 0.99,
            "validation_macro_f1": 0.99,
            "node_budget_exact_match": True,
            "represented_context_exact_or_bounded": False,
            "represented_context_ratio": 1.0,
            "full_residual_upperbound": True,
            "eligible_for_main_decision": False,
            "candidate_allclose_to_full": True,
            "no_test_leakage": True,
        },
    ]
    _write_rows(tmp_path / "gate17_3_raw_rows.csv", rows)
    diag = tmp_path / "diagnostics"
    _write_rows(diag / "gate17_3_budget_breakdown.csv", rows)
    _write_rows(diag / "gate17_3_represented_context_budget.csv", rows)
    _write_rows(diag / "gate17_3_candidate_semantic_delta.csv", rows[1:])
    _write_rows(diag / "gate17_3_prototype_saturation.csv", rows[1:])

    result = summarize(tmp_path, tmp_path)

    assert result["decision"] == "FAIL_SELECTOR_AND_LOSSY_PROTOTYPE"
    assert result["failure_reasons"][0] == "FAIL_REPRESENTED_CONTEXT_BUDGET"
    assert "FAIL_FULL_RESIDUAL_SHORTCUT_ONLY" in result["failure_reasons"]
    assert result["best_main_method"] is None
    assert result["best_method_is_meaningful"] is False
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["full_residual_upperbound_excluded_from_decision"] is True


def test_gate17_3_summary_allows_meaningful_best_when_gates_pass(tmp_path):
    rows = [
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "flatten-sum-support-only",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.70,
            "accuracy": 0.70,
            "validation_macro_f1": 0.70,
            "node_budget_exact_match": True,
        },
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "HeSF-SS-real-occlusion-lossy-prototype",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.695,
            "accuracy": 0.695,
            "validation_macro_f1": 0.695,
            "node_budget_exact_match": True,
            "represented_context_exact_or_bounded": True,
            "represented_context_ratio": 0.38,
            "eligible_for_main_decision": True,
            "occlusion_task_signal_pass": True,
            "validation_signal_pass": True,
            "prototype_saturation_rate": 0.1,
            "prototype_member_count_p90": 10,
            "prototype_member_count_p99": 10,
            "max_members_per_prototype": 512,
            "candidate_allclose_to_full": False,
            "no_test_leakage": True,
        },
    ]
    _write_rows(tmp_path / "gate17_3_raw_rows.csv", rows)
    diag = tmp_path / "diagnostics"
    _write_rows(diag / "gate17_3_budget_breakdown.csv", rows)
    _write_rows(diag / "gate17_3_represented_context_budget.csv", rows)
    _write_rows(diag / "gate17_3_candidate_semantic_delta.csv", [rows[1]])
    _write_rows(diag / "gate17_3_prototype_saturation.csv", [rows[1]])

    result = summarize(tmp_path, tmp_path)

    assert result["decision"] == "CONTINUE_LOSSY_FEEDBACK_DBLP_CLOSE"
    assert result["failure_reasons"] == []
    assert result["best_main_method"] == "HeSF-SS-real-occlusion-lossy-prototype"
    assert result["dblp_gap_vs_best_strong_baseline"] == -0.005
