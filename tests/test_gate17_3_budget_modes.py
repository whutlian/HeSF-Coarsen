from __future__ import annotations

import numpy as np

from experiments.scripts.gate17_3_budget import compute_gate17_3_budget_fields
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def _tiny_graph() -> tuple[HeteroGraph, dict]:
    node_type = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    labels = np.array([0, 1, -1, -1, -1, -1], dtype=np.int64)
    src = np.array([0, 0, 1, 1], dtype=np.int64)
    dst = np.array([2, 3, 4, 5], dtype=np.int64)
    graph = HeteroGraph(
        num_nodes=6,
        node_type=node_type,
        relations={
            0: RelationAdj(src=src, dst=dst, weight=None, src_type=0, dst_type=1, relation_id=0),
            1: RelationAdj(src=dst, dst=src, weight=None, src_type=1, dst_type=0, relation_id=1),
        },
        relation_specs={
            0: RelationSpec(0, "target_to_support", 0, 1),
            1: RelationSpec(1, "support_to_target", 1, 0),
        },
        features={0: np.eye(2, dtype=np.float32), 1: np.ones((4, 2), dtype=np.float32)},
        labels=labels,
    )
    support_nodes = np.array([2, 3, 4, 5], dtype=np.int64)
    one = np.ones((graph.num_nodes, 1), dtype=np.float32)
    return graph, {
        "support_nodes": support_nodes,
        "all_node_component_matrices": {
            "class_footprint": one,
            "relation_profile": one,
            "anchor_distribution": one,
            "degree_profile": one,
        },
    }


def test_selection_only_mode_drops_unselected_support_nodes():
    graph, support_features = _tiny_graph()

    coarse, assignment, diagnostics = build_selected_support_graph(
        graph,
        np.array([2], dtype=np.int64),
        SupportSelectorConfig(
            background_strategy="drop",
            residual_prototype_mode="none",
            allow_background_bucket=False,
        ),
        target_node_type=0,
        support_features=support_features,
    )

    assert coarse.num_nodes == 3
    assert int(np.sum(coarse.node_type != 0)) == 1
    assert diagnostics["residual_prototype_mode"] == "none"
    assert diagnostics["prototype_background_count"] == 0
    assert diagnostics["dropped_support_count"] == 3
    assert diagnostics["represented_support_context_count"] == 1
    assert assignment.assignment[0] != assignment.assignment[1]
    assert assignment.assignment[2] != assignment.assignment[0]


def test_lossy_budget_fields_bound_represented_context():
    fields = compute_gate17_3_budget_fields(
        original_support_nodes=100,
        requested_support_ratio=0.30,
        selected_raw_support_count=30,
        forced_raw_support_count=0,
        prototype_background_count=5,
        prototype_member_count_sum=8,
        prototype_member_budget_total=10,
        full_residual_upperbound=False,
        method="HeSF-SS-real-occlusion-lossy-prototype",
    )

    assert fields["node_budget_count"] == 35
    assert fields["node_budget_ratio"] == 0.35
    assert fields["node_budget_exact_match"] is False
    assert fields["represented_context_count"] == 38
    assert fields["represented_context_ratio"] == 0.38
    assert fields["represented_context_exact_or_bounded"] is True
    assert fields["eligible_for_main_decision"] is False


def test_full_residual_upperbound_is_diagnostic_only():
    fields = compute_gate17_3_budget_fields(
        original_support_nodes=100,
        requested_support_ratio=0.30,
        selected_raw_support_count=30,
        forced_raw_support_count=0,
        prototype_background_count=20,
        prototype_member_count_sum=70,
        prototype_member_budget_total=70,
        full_residual_upperbound=True,
        method="HeSF-SS-full-residual-prototype-upperbound",
    )

    assert fields["represented_context_ratio"] == 1.0
    assert fields["full_residual_upperbound"] is True
    assert fields["eligible_for_main_decision"] is False
