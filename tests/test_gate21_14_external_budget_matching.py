from __future__ import annotations

from experiments.scripts.gate21_14_common import budget_match


def test_structural_budget_match_uses_structural_ratio_not_support_node_ratio() -> None:
    row = {
        "budget_type": "structural_storage_ratio",
        "requested_budget": 0.16,
        "actual_structural_storage_ratio": 0.159,
        "actual_support_node_ratio": 0.50,
    }

    assert budget_match(row) is True


def test_structural_budget_mismatch_is_not_fair_even_if_support_ratio_matches() -> None:
    row = {
        "budget_type": "structural_storage_ratio",
        "requested_budget": 0.16,
        "actual_structural_storage_ratio": 0.57,
        "actual_support_node_ratio": 0.16,
    }

    assert budget_match(row) is False
