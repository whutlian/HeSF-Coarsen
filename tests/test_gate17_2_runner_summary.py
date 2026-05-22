import csv
import json

from experiments.scripts.run_gate17_2_single_seed_effective_budget import (
    DEFAULT_METHODS,
    GATE17_2_SINGLE_SEED_BY_DATASET,
    parse_dataset_seed_map,
    resolve_dataset_seed_pairs,
)
from experiments.scripts.summarize_gate17_2 import summarize


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


def test_gate17_2_default_seed_policy_uses_only_best_single_seed_per_dataset():
    assert GATE17_2_SINGLE_SEED_BY_DATASET == {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
    pairs = resolve_dataset_seed_pairs(["ACM", "DBLP", "IMDB"], "best_single", "")

    assert pairs == [("ACM", 23456), ("DBLP", 23456), ("IMDB", 45678)]
    assert len(pairs) == 3
    assert "HeSF-SS-real-validation-no-fallback" in DEFAULT_METHODS
    assert "HeSF-SS-real-occlusion-plus-dblp-prototype-budgeted" in DEFAULT_METHODS


def test_gate17_2_explicit_dataset_seed_map_override():
    parsed = parse_dataset_seed_map("ACM:1,DBLP:2")

    assert parsed == {"ACM": 1, "DBLP": 2}
    assert resolve_dataset_seed_pairs(["ACM"], "explicit", "ACM:7") == [("ACM", 7)]


def test_gate17_2_summary_records_failure_reasons_and_disables_best_method(tmp_path):
    rows = [
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "H6-no-spec-support-only",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "validation_macro_f1": 0.5,
            "primary_eval_mode": "compressed_projected",
            "support_budget_exact_match": True,
        },
        {
            "dataset": "DBLP",
            "seed": 23456,
            "method": "HeSF-SS-dblp-aware-prototype-no-free-raw",
            "requested_support_ratio": 0.3,
            "status": "success",
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "validation_macro_f1": 0.5,
            "primary_eval_mode": "compressed_projected",
            "support_budget_exact_match": True,
            "no_test_leakage": True,
            "effective_support_node_ratio": 0.6,
            "budget_leak_ratio": 0.3,
            "candidate_allclose_to_full": True,
            "tree_tensor_l2_delta_vs_full": 0.0,
            "prototype_saturation_rate": 1.0,
            "prototype_member_count_p90": 512,
            "prototype_member_count_p99": 512,
            "max_members_per_prototype": 512,
            "rare_class_fallback_count": 0,
        },
    ]
    _write_rows(tmp_path / "gate17_2_raw_rows.csv", rows)
    _write_rows(
        tmp_path / "diagnostics" / "effective_budget.csv",
        [rows[1]],
    )
    _write_rows(
        tmp_path / "diagnostics" / "candidate_semantic_delta.csv",
        [rows[1]],
    )
    _write_rows(
        tmp_path / "diagnostics" / "prototype_budget_saturation.csv",
        [rows[1]],
    )

    result = summarize(tmp_path)

    assert result["decision"] == "FAIL_EFFECTIVE_BUDGET_LEAK"
    assert "FAIL_EFFECTIVE_BUDGET_LEAK" in result["failure_reasons"]
    assert "FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT" in result["failure_reasons"]
    assert "FAIL_PROTOTYPE_SATURATION_DBLP" in result["failure_reasons"]
    assert result["best_validation_selected_method"] is None
    assert result["best_method_is_meaningful"] is False
    assert (tmp_path / "result.json").exists()
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["failure_reasons"]
