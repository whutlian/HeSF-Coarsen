from __future__ import annotations


def test_structural_budget_matching_uses_tolerance_and_marks_infeasible() -> None:
    from hesf_coarsen.eval.official.external_tp_task_runner import budget_match_status

    assert budget_match_status({"budget_type": "structural_storage_ratio", "requested_budget": 0.16, "actual_structural_storage_ratio": 0.171})["budget_match_pass"] is True
    mismatch = budget_match_status({"budget_type": "structural_storage_ratio", "requested_budget": 0.16, "actual_structural_storage_ratio": 0.57})
    assert mismatch["budget_match_pass"] is False
    assert mismatch["budget_match_status"] == "budget_infeasible"


def test_single_seed_smoke_is_not_5x5_ready() -> None:
    from hesf_coarsen.eval.official.external_tp_task_runner import external_tp_by_method

    rows = [{"dataset": "DBLP", "method": "Random-HG-TP", "budget_type": "structural_storage_ratio", "requested_budget": 0.16, "actual_structural_storage_ratio": 0.16, "graph_seed": 1, "training_seed": 1, "training_executed": True, "official_hgb_exported": True, "official_sehgnn_unmodified": True, "test_micro_f1": 0.8, "test_macro_f1": 0.78}]

    summary = external_tp_by_method(rows, required_methods=["Random-HG-TP"])[0]

    assert summary["ready_row_count"] == 1
    assert summary["expected_row_count"] == 25
    assert summary["ready_5x5_flag"] is False
