import numpy as np

from experiments.scripts.gate17_2_effective_budget import compute_effective_budget_fields
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def test_gate17_2_safe_selector_defaults_disable_free_raw_bridges():
    cfg = SupportSelectorConfig()

    assert cfg.force_raw_bridge_nodes is False
    assert cfg.force_raw_keep_high_degree_bridges is False
    assert cfg.raw_bridge_mode == "off"


def test_effective_budget_counts_raw_and_prototype_nodes_separately():
    fields = compute_effective_budget_fields(
        original_support_nodes=100,
        requested_support_ratio=0.30,
        selected_support_count=25,
        graph_diagnostics={
            "forced_raw_bridge_count": 0,
            "prototype_background_count": 5,
            "prototype_member_count_sum": 40,
        },
        candidate_allclose_to_full=False,
    )

    assert fields["selected_budget_support_count"] == 25
    assert fields["forced_raw_support_count"] == 0
    assert fields["prototype_background_count"] == 5
    assert fields["prototype_member_count_sum"] == 40
    assert fields["effective_raw_support_count"] == 25
    assert fields["effective_support_node_count"] == 30
    assert fields["represented_support_context_count"] == 65
    assert fields["effective_support_node_ratio"] == 0.30
    assert fields["represented_support_context_ratio"] == 0.65
    assert abs(fields["budget_leak_ratio"]) <= 1.0e-12
    assert fields["represented_context_leak_ratio"] == 0.35
    assert fields["effective_budget_exact_match"] is True


def test_effective_budget_detects_unbudgeted_forced_raw_leak():
    fields = compute_effective_budget_fields(
        original_support_nodes=100,
        requested_support_ratio=0.30,
        selected_support_count=30,
        graph_diagnostics={"forced_raw_bridge_count": 20, "prototype_background_count": 0},
        candidate_allclose_to_full=True,
    )

    assert fields["effective_support_node_ratio"] == 0.50
    assert fields["budget_leak_ratio"] == 0.20
    assert fields["effective_budget_exact_match"] is False
    assert fields["candidate_allclose_to_full"] is True

