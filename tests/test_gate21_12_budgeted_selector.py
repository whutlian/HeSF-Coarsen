from __future__ import annotations


def test_gate21_12_selector_uses_distinct_hashes_and_separates_plan_rows() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import plan_gate21_12_budgeted_channels

    result = plan_gate21_12_budgeted_channels("DBLP", structural_budgets=[0.12, 0.16, 0.20, 0.30])
    plans = {
        row["requested_budget_name"]: row
        for row in result["selector_rows"]
        if row["row_kind"] == "planner_plan"
    }
    linked = {
        row["method"]: row
        for row in result["selector_rows"]
        if row["row_kind"] == "linked_task_result"
    }

    assert plans["budget12"]["selected_canonical_method"] == "HeSF-RCS-APV12"
    assert plans["budget16"]["selected_canonical_method"] == "HeSF-RCS-APV16"
    assert plans["budget12"]["selected_edge_hash"] != plans["budget16"]["selected_edge_hash"]
    assert plans["budget12"]["selected_edge_hash"] == linked["HeSF-RCS-APV12"]["selected_edge_hash"]
    assert plans["budget16"]["selected_edge_hash"] == linked["HeSF-RCS-APV16"]["selected_edge_hash"]

    for row in plans.values():
        assert row["official_hgb_exported"] is False
        assert row["official_sehgnn_unmodified"] is False
        assert row["training_executed"] is False
        assert row["eligible_for_planner_decision"] is True
        assert row["eligible_for_official_main_table"] is False
        assert row["planner_config_hash"]
        assert row["planner_input_graph_hash"]
        assert row["selected_edge_hash_by_relation"]
        assert row["export_file_hash"]
        assert row["linked_official_result_hash"]

    assert plans["budget20"]["actual_structural_storage_ratio"] == 0.15916
    assert plans["budget20"]["budget_padding_policy"] == "none"
    assert plans["budget20"]["budget_slack"] > 0.03
    assert plans["budget30"]["budget_slack"] > 0.13
    assert result["hash_audit"]["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] is True


def test_gate21_12_apv16_deterministic_proof_has_required_fields() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import gate21_12_apv16_deterministic_proof

    proof = gate21_12_apv16_deterministic_proof(dataset="DBLP", graph_seed_values=[5, 1, 3, 2, 4])

    assert proof["method"] == "HeSF-RCS-APV16"
    assert proof["deterministic_proof_pass"] is True
    assert proof["graph_seed_values_tested"] == [1, 2, 3, 4, 5]
    assert proof["expected_export_hash_unique_count"] == 1
    assert proof["actual_export_hash_unique_count"] == 1
    assert proof["selected_edge_hash_by_relation"]
    assert proof["repeat_export_hashes"] == [proof["repeat_export_hashes"][0]] * proof["repeat_count"]
