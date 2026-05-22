from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def _graph_and_features(num_support: int = 8, bridge_node: int | None = None) -> tuple[HeteroGraph, dict]:
    node_type = np.array([0, 0] + [1] * num_support, dtype=np.int32)
    labels = np.array([0, 1] + [-1] * num_support, dtype=np.int64)
    support_nodes = np.arange(2, 2 + num_support, dtype=np.int64)
    src = np.repeat(np.array([0, 1], dtype=np.int64), num_support)
    dst = np.tile(support_nodes, 2)
    graph = HeteroGraph(
        num_nodes=2 + num_support,
        node_type=node_type,
        relations={
            0: RelationAdj(src=src, dst=dst, weight=None, src_type=0, dst_type=1, relation_id=0),
            1: RelationAdj(src=dst, dst=src, weight=None, src_type=1, dst_type=0, relation_id=1),
        },
        relation_specs={
            0: RelationSpec(0, "target_to_support", 0, 1),
            1: RelationSpec(1, "support_to_target", 1, 0),
        },
        features={0: np.eye(2, dtype=np.float32), 1: np.ones((num_support, 2), dtype=np.float32)},
        labels=labels,
    )
    class_fp = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    class_fp[support_nodes, 0] = 1.0
    relation = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    relation[support_nodes, 0] = 1.0
    anchor = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    anchor[support_nodes, 0] = 1.0
    degree = np.zeros((graph.num_nodes, 2), dtype=np.float32)
    degree[support_nodes, 0] = 1.0
    if bridge_node is not None:
        class_fp[bridge_node] = np.array([1.0, 1.0], dtype=np.float32)
        relation[bridge_node] = np.array([1.0, 1.0], dtype=np.float32)
        anchor[bridge_node] = np.array([1.0, 1.0], dtype=np.float32)
        degree[bridge_node] = np.array([100.0, 0.0], dtype=np.float32)
    return graph, {
        "all_node_component_matrices": {
            "class_footprint": class_fp,
            "relation_profile": relation,
            "anchor_distribution": anchor,
            "degree_profile": degree,
        }
    }


def test_dblp_aware_large_prototype_split_reports_saturation_rate():
    graph, support_features = _graph_and_features(num_support=8)

    _coarse, _assignment, diagnostics = build_selected_support_graph(
        graph,
        np.array([], dtype=np.int64),
        SupportSelectorConfig(
            background_strategy="dblp_aware_prototype",
            max_members_per_prototype=2,
            force_raw_bridge_nodes=False,
        ),
        target_node_type=0,
        support_features=support_features,
    )

    assert diagnostics["prototype_member_count_max"] <= 2
    assert diagnostics["large_prototype_split_count"] > 0
    assert diagnostics["prototype_saturation_rate"] > 0.0
    assert diagnostics["prototype_fallback_member_count"] == 0


def test_forced_raw_bridge_nodes_are_preserved_and_diagnosed():
    graph, support_features = _graph_and_features(num_support=6, bridge_node=2)

    _coarse, assignment, diagnostics = build_selected_support_graph(
        graph,
        np.array([], dtype=np.int64),
        SupportSelectorConfig(
            background_strategy="dblp_aware_prototype",
            max_members_per_prototype=2,
            force_raw_bridge_nodes=True,
            min_raw_bridge_per_relation_channel=1,
            rare_class_never_fallback=True,
        ),
        target_node_type=0,
        support_features=support_features,
    )

    assert diagnostics["forced_raw_bridge_count"] > 0
    assert diagnostics["raw_bridge_by_type"].get("1", 0) > 0
    assert diagnostics["raw_bridge_by_relation_channel"].get("0", 0) > 0
    assert diagnostics["rare_class_fallback_count"] == 0
    assert len([node for node in range(graph.num_nodes) if assignment.assignment[node] == assignment.assignment[2]]) == 1
