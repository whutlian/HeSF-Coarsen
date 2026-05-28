from __future__ import annotations

import json


def test_gate21_13_selector_links_budget_rows_to_distinct_official_hashes() -> None:
    from hesf_coarsen.eval.official.selector_result_linkage import gate21_13_budgeted_selector_linkage

    result = gate21_13_budgeted_selector_linkage("DBLP", [0.12, 0.16, 0.20, 0.30])
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
    audit = {row["budget_name"]: row for row in result["hash_audit_rows"]}

    assert plans["budget12"]["selected_canonical_method"] == "HeSF-RCS-APV12"
    assert plans["budget16"]["selected_canonical_method"] == "HeSF-RCS-APV16"
    assert plans["budget12"]["selected_edge_hash"] != plans["budget16"]["selected_edge_hash"]
    assert audit["budget12"]["selected_hash_matches_official_main"] is True
    assert audit["budget16"]["selected_hash_matches_official_main"] is True
    assert audit["budget12"]["apv12_hash_differs_from_apv16_hash"] is True
    assert plans["budget12"]["selected_edge_hash"] == linked["HeSF-RCS-APV12"]["selected_edge_hash"]
    assert plans["budget16"]["selected_edge_hash"] == linked["HeSF-RCS-APV16"]["selected_edge_hash"]

    for budget_name in ("budget20", "budget30"):
        row = plans[budget_name]
        assert row["selected_canonical_method"] == "HeSF-RCS-APV16"
        assert row["budget_padding_policy"] == "none"
        assert row["budget_slack"] > 0
        assert audit[budget_name]["selector_hash_audit_pass"] is True

    for row in plans.values():
        assert row["uses_test_metrics_for_selection"] is False
        assert row["uses_test_labels_for_selection"] is False
        assert row["official_hgb_exported"] is False
        assert row["training_executed"] is False
        assert row["planner_trace_hash"]
        assert row["selection_config_hash"]
        assert row["selection_input_hash"]


def test_gate21_13_apv16_deterministic_proof_is_single_hash() -> None:
    from hesf_coarsen.eval.official.selector_result_linkage import gate21_13_deterministic_selector_proof

    proof = gate21_13_deterministic_selector_proof(
        dataset="DBLP",
        method="HeSF-RCS-APV16",
        graph_seed_values=[5, 1, 3, 2, 4],
    )

    assert proof["method"] == "HeSF-RCS-APV16"
    assert proof["deterministic_proof_pass"] is True
    assert proof["actual_export_hash_unique_count"] == 1
    assert proof["expected_export_hash_unique_count"] == 1
    assert json.loads(proof["graph_seed_values_tested"]) == [1, 2, 3, 4, 5]
    assert proof["selected_edge_hash"]
    assert proof["export_file_hash"]
