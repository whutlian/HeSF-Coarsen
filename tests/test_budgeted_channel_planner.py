from __future__ import annotations


def test_dblp_budgeted_planner_outputs_apv12_and_apv16_like_plans_without_test_leakage() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import plan_budgeted_channels

    result = plan_budgeted_channels(dataset="DBLP", structural_budgets=[0.12, 0.16, 0.20])
    by_budget = {round(row["budget_target"], 2): row for row in result["plan_rows"]}

    assert by_budget[0.12]["AP_keep"] == 1.0
    assert by_budget[0.12]["PV_keep"] == 1.0
    assert by_budget[0.12]["PA_keep"] == 0.0
    assert by_budget[0.12]["VP_keep"] == 0.0
    assert by_budget[0.12]["PT_keep"] == 0.0
    assert by_budget[0.12]["TP_keep"] == 0.0
    assert by_budget[0.16]["AP_keep"] == 1.0
    assert by_budget[0.16]["PV_keep"] == 1.0
    assert by_budget[0.16]["PA_keep"] == 0.5
    assert by_budget[0.16]["VP_keep"] == 0.5
    assert by_budget[0.16]["PT_keep"] == 0.0
    assert by_budget[0.16]["TP_keep"] == 0.0
    assert all(row["selection_uses_test_metrics"] is False for row in result["plan_rows"])
    assert all(row["uses_test_labels_for_selection"] is False for row in result["plan_rows"])
    assert result["utility_rows"][0]["selection_signal_source"] in {"train_val_only", "structure_only", "feature_only", "cached_probe"}


def test_non_dblp_planner_uses_generic_relation_keys() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import plan_budgeted_channels

    result = plan_budgeted_channels(dataset="ACM", structural_budgets=[0.20])

    assert result["plan_rows"][0]["dataset"] == "ACM"
    assert "AP_keep" in result["plan_rows"][0]
    assert result["plan_rows"][0]["selected_channel_plan_json"]
    assert all("AP" not in row["channel_key"] for row in result["utility_rows"])
