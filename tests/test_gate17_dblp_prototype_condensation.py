import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def _large_bucket_graph(num_support: int = 8) -> tuple[HeteroGraph, dict]:
    node_type = np.array([0, 0] + [1] * num_support, dtype=np.int32)
    labels = np.array([0, 1] + [-1] * num_support, dtype=np.int64)
    support_nodes = np.arange(2, 2 + num_support, dtype=np.int64)
    src = np.repeat(np.array([0, 1], dtype=np.int64), num_support)
    dst = np.tile(support_nodes, 2)
    relations = {
        0: RelationAdj(src=src, dst=dst, weight=None, src_type=0, dst_type=1, relation_id=0),
        1: RelationAdj(src=dst, dst=src, weight=None, src_type=1, dst_type=0, relation_id=1),
    }
    graph = HeteroGraph(
        num_nodes=2 + num_support,
        node_type=node_type,
        relations=relations,
        relation_specs={
            0: RelationSpec(0, "target_to_support", 0, 1),
            1: RelationSpec(1, "support_to_target", 1, 0),
        },
        features={0: np.eye(2, dtype=np.float32), 1: np.ones((num_support, 2), dtype=np.float32)},
        labels=labels,
    )
    all_nodes = graph.num_nodes
    class_fp = np.zeros((all_nodes, 2), dtype=np.float32)
    class_fp[support_nodes, 0] = 1.0
    relation = np.zeros((all_nodes, 2), dtype=np.float32)
    relation[support_nodes, 0] = 1.0
    anchor = np.zeros((all_nodes, 2), dtype=np.float32)
    anchor[support_nodes, 0] = 1.0
    degree = np.zeros((all_nodes, 2), dtype=np.float32)
    degree[support_nodes, 0] = np.arange(1, num_support + 1, dtype=np.float32)
    support_features = {
        "all_node_component_matrices": {
            "class_footprint": class_fp,
            "relation_profile": relation,
            "anchor_distribution": anchor,
            "degree_profile": degree,
        }
    }
    return graph, support_features


def test_dblp_aware_prototype_splits_large_semantic_bucket():
    graph, support_features = _large_bucket_graph(num_support=8)

    _coarse, _assignment, diagnostics = build_selected_support_graph(
        graph,
        np.array([], dtype=np.int64),
        SupportSelectorConfig(
            background_strategy="dblp_aware_prototype",
            max_members_per_prototype=2,
            max_prototypes_per_class_anchor_relation=8,
        ),
        target_node_type=0,
        support_features=support_features,
    )

    assert diagnostics["background_strategy"] == "dblp_aware_prototype"
    assert diagnostics["large_prototype_count"] > 0
    assert diagnostics["large_prototype_split_count"] > 0
    assert diagnostics["prototype_member_count_max"] <= 2
    assert diagnostics["prototype_key_mode"] == "dblp_aware"
