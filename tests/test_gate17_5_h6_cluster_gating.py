from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj
from hesf_coarsen.task_first.selection.h6_cluster_gating import (
    build_gated_h6_graph,
    extract_h6_cluster_units,
    h6_fill_support_nodes,
    select_h6_clusters_by_budget,
)


def _toy_original_graph() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=6,
        node_type=np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int32),
        relations={},
        features={
            0: np.asarray([[1.0], [2.0]], dtype=np.float32),
            1: np.asarray([[3.0], [4.0], [5.0], [6.0]], dtype=np.float32),
        },
        labels=np.asarray([0, 1, -1, -1, -1, -1]),
    )


def _toy_h6_coarse_graph() -> HeteroGraph:
    return HeteroGraph(
        num_nodes=5,
        node_type=np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
        relations={
            0: RelationAdj(
                src=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
                dst=np.asarray([2, 3, 0, 1, 1], dtype=np.int64),
                weight=np.ones(5, dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        features={
            0: np.asarray([[1.0], [2.0]], dtype=np.float32),
            1: np.asarray([[3.5], [5.0], [6.0]], dtype=np.float32),
        },
    )


def test_extract_h6_cluster_units_reports_member_weighted_support_clusters():
    graph = _toy_original_graph()
    assignment = np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int64)

    units = extract_h6_cluster_units(graph, assignment, target_type=0)

    assert [unit.cluster_id for unit in units] == [2, 3, 4]
    assert [unit.member_count for unit in units] == [2, 1, 1]
    assert all(unit.cluster_type == 1 for unit in units)
    assert units[0].member_nodes.tolist() == [2, 3]


def test_select_h6_clusters_by_budget_is_member_weighted_and_logs_fill_counts():
    graph = _toy_original_graph()
    assignment = np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int64)
    units = extract_h6_cluster_units(graph, assignment, target_type=0)

    selected = select_h6_clusters_by_budget(
        units,
        support_count=4,
        requested_support_ratio=0.5,
        validation_scores={2: 0.2, 3: -0.0002, 4: 0.1},
        min_gain=1.0e-4,
        neutral_fill_max_drop=1.0e-4,
        negative_fill_max_drop=5.0e-4,
        budget_penalty_lambda=0.05,
        underfill_penalty_lambda=0.10,
    )

    assert selected.selected_cluster_ids == [2]
    assert selected.member_count_selected == 2
    assert selected.member_ratio_selected == 0.5
    assert selected.positive_gain_block_count == 1
    assert selected.negative_fill_block_count == 0
    assert selected.proxy_fill_block_count == 0


def test_build_gated_h6_graph_keeps_targets_and_rebuilds_induced_graph():
    original = _toy_original_graph()
    h6 = _toy_h6_coarse_graph()
    assignment = np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int64)

    gated, gated_assignment, kept = build_gated_h6_graph(
        original=original,
        h6_coarse=h6,
        h6_assignment=assignment,
        selected_cluster_ids=[2, 4],
        target_type=0,
    )

    assert kept.tolist() == [0, 1, 2, 4]
    assert gated.num_nodes == 4
    assert gated.num_nodes < h6.num_nodes
    assert gated_assignment[0] == 0
    assert gated_assignment[1] == 1
    assert set(gated_assignment[[2, 3, 5]].tolist()) == {2, 3}


def test_h6_fill_support_nodes_uses_h6_diversity_before_generic_fill():
    graph = _toy_original_graph()
    assignment = np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int64)

    filled, diag = h6_fill_support_nodes(
        graph=graph,
        h6_assignment=assignment,
        target_type=0,
        selected_support_nodes=np.asarray([4], dtype=np.int64),
        requested_support_count=3,
    )

    assert len(filled) == 3
    assert 4 in set(filled.tolist())
    assert diag["h6_fill_block_count"] == 2
    assert diag["h6_fill_support_count"] == 2
    assert diag["h6_fill_budget_fraction"] == 2 / 3
