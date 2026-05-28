from __future__ import annotations

from hesf_coarsen.eval.official.gate21_14_decision import gate21_14_decision


def test_selector_hash_linkage_requires_planner_rows_not_main_and_linked_hashes() -> None:
    selector_rows = [
        {
            "requested_structural_budget": budget,
            "planner_row": True,
            "eligible_for_planner_decision": True,
            "eligible_for_official_main_table": False,
            "linked_official_task_method": "HeSF-RCS-APV12" if budget == 0.12 else "HeSF-RCS-APV16",
            "linked_task_result_hash": f"linked-{budget}",
            "uses_test_metrics_for_selection": False,
        }
        for budget in (0.12, 0.16, 0.20, 0.30, 0.50)
    ]
    audit_rows = [
        {"assertion_name": name, "pass": True}
        for name in (
            "APV12_selected_edge_hash != APV16_selected_edge_hash",
            "budget12_selected_edge_hash == official_main_APV12_selected_edge_hash",
            "budget16_selected_edge_hash == official_main_APV16_selected_edge_hash",
            "same_input_different_graph_seed_same_selected_edge_hash_for_deterministic_plans",
            "same_selected_edge_hash_same_export_hash",
            "planner_rows_not_marked_official_main_eligible",
            "linked_task_rows_marked_official_main_eligible_if_unmodified",
        )
    ]

    flags = gate21_14_decision(budgeted_selector_rows=selector_rows, selector_hash_audit=audit_rows)

    assert flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] is True
    assert flags["BUDGETED_SELECTOR_LINKAGE_PASS"] is True
    assert flags["BUDGETED_SELECTOR_NO_TEST_LEAKAGE_PASS"] is True


def test_selector_linkage_fails_if_planner_row_is_marked_official_main() -> None:
    selector_rows = [
        {
            "requested_structural_budget": budget,
            "planner_row": True,
            "eligible_for_planner_decision": True,
            "eligible_for_official_main_table": budget == 0.12,
            "linked_official_task_method": "HeSF-RCS-APV12",
            "linked_task_result_hash": f"linked-{budget}",
            "uses_test_metrics_for_selection": False,
        }
        for budget in (0.12, 0.16, 0.20, 0.30, 0.50)
    ]

    flags = gate21_14_decision(budgeted_selector_rows=selector_rows)

    assert flags["BUDGETED_SELECTOR_LINKAGE_PASS"] is False
