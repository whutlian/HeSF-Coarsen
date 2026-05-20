from hesf_coarsen.task_first.pipeline import task_first_budget_stop_reason


def test_ratio_budget_stop_reason_reports_candidate_exhaustion():
    reason, floor = task_first_budget_stop_reason(
        current_support_nodes=10,
        desired_support_nodes=2,
        max_support_merges=8,
        candidate_pair_count=0,
        eligible_candidate_pair_count=0,
        selected_support_merges=0,
    )

    assert reason == "candidate_exhaustion"
    assert "candidate" in floor


def test_ratio_budget_stop_reason_reports_reached_ratio():
    reason, floor = task_first_budget_stop_reason(
        current_support_nodes=2,
        desired_support_nodes=2,
        max_support_merges=0,
        candidate_pair_count=0,
        eligible_candidate_pair_count=0,
        selected_support_merges=0,
    )

    assert reason == "reached_requested_support_ratio"
    assert floor == ""
