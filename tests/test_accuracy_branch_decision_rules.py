from hesf_coarsen.accuracy.accuracy_branch_decision import decide_accuracy_branch


def test_decision_drops_branch_when_no_faithful_results_exist() -> None:
    verdict = decide_accuracy_branch([])

    assert verdict["decision"] == "DROP_ENTIRE_ACCURACY_BRANCH"
    assert "no official or faithful" in verdict["reason"].lower()


def test_decision_keeps_minimal_when_a1_beats_comparators_on_two_datasets() -> None:
    rows = [
        {"dataset": "ACM", "method": "A1_target_preserve", "model_fidelity": "faithful_reproduction", "eval_mode": "real_full_target_inference", "macro_f1": 0.8},
        {"dataset": "ACM", "method": "flatten-sum_keep_target", "model_fidelity": "faithful_reproduction", "eval_mode": "real_full_target_inference", "macro_f1": 0.7},
        {"dataset": "DBLP", "method": "A1_target_preserve", "model_fidelity": "faithful_reproduction", "eval_mode": "real_full_target_inference", "macro_f1": 0.75},
        {"dataset": "DBLP", "method": "H6_keep_target", "model_fidelity": "faithful_reproduction", "eval_mode": "real_full_target_inference", "macro_f1": 0.72},
        {"dataset": "IMDB", "method": "A1_target_preserve", "model_fidelity": "faithful_reproduction", "eval_mode": "real_full_target_inference", "macro_f1": 0.6},
        {"dataset": "IMDB", "method": "flatten-sum_keep_target", "model_fidelity": "faithful_reproduction", "eval_mode": "real_full_target_inference", "macro_f1": 0.65},
    ]

    verdict = decide_accuracy_branch(rows)

    assert verdict["decision"] == "KEEP_ACCURACY_BRANCH_MINIMAL"
    assert verdict["wins_vs_internal_comparator"] >= 2
