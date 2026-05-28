from __future__ import annotations


def test_gate21_11_budget20_reports_slack_instead_of_fake_filling() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import plan_gate21_11_budgeted_channels

    result = plan_gate21_11_budgeted_channels("DBLP", structural_budgets=[0.12, 0.16, 0.20])
    by_name = {row["requested_budget_name"]: row for row in result["selector_rows"]}

    assert by_name["budget12"]["selected_canonical_method"] == "HeSF-RCS-APV12"
    assert by_name["budget16"]["selected_canonical_method"] == "HeSF-RCS-APV16"
    assert by_name["budget20"]["selected_canonical_method"] == "HeSF-RCS-APV16"
    assert by_name["budget20"]["actual_structural_storage_ratio"] == 0.15916
    assert by_name["budget20"]["budget_slack"] > 0.03
    assert by_name["budget20"]["budget_padding_policy"] == "none"
    assert by_name["budget20"]["budget_matched_within_tolerance"] is False
    assert all(row["uses_test_metric"] is False for row in result["trace_rows"])


def test_gate21_11_apv16_deterministic_proof_hashes_are_seed_stable() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import gate21_11_apv16_deterministic_proof

    proof_a = gate21_11_apv16_deterministic_proof(dataset="DBLP", graph_seed_values=[1, 2, 3, 4, 5])
    proof_b = gate21_11_apv16_deterministic_proof(dataset="DBLP", graph_seed_values=[5, 4, 3, 2, 1])

    assert proof_a["deterministic_proof_pass"] is True
    assert proof_a["repeat_count"] >= 3
    assert proof_a["actual_export_hash_unique_count"] == 1
    assert proof_a["selected_edge_hash"] == proof_b["selected_edge_hash"]
    assert proof_a["export_hashes_for_repeated_runs"] == proof_b["export_hashes_for_repeated_runs"]
