import numpy as np

from experiments.scripts.run_gate17_1_support_sensitivity import semantic_tree_delta_row
from hesf_coarsen.eval.hettree_task import build_semantic_tree_features
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _full_and_target_only_graphs():
    full = HeteroGraph(
        num_nodes=3,
        node_type=np.array([0, 0, 1], dtype=np.int32),
        relations={
            1: RelationAdj(
                src=np.array([2, 2], dtype=np.int64),
                dst=np.array([0, 1], dtype=np.int64),
                weight=np.ones(2, dtype=np.float32),
                src_type=1,
                dst_type=0,
                relation_id=1,
            )
        },
        relation_specs={1: RelationSpec(1, "support_to_target", 1, 0)},
        features={
            0: np.zeros((2, 1), dtype=np.float32),
            1: np.array([[3.0]], dtype=np.float32),
        },
        labels=np.array([0, 1, -1], dtype=np.int64),
    )
    target_only = HeteroGraph(
        num_nodes=2,
        node_type=np.array([0, 0], dtype=np.int32),
        relations={
            1: RelationAdj(
                src=np.array([], dtype=np.int64),
                dst=np.array([], dtype=np.int64),
                weight=np.array([], dtype=np.float32),
                src_type=1,
                dst_type=0,
                relation_id=1,
            )
        },
        relation_specs={1: RelationSpec(1, "support_to_target", 1, 0)},
        features={0: np.zeros((2, 1), dtype=np.float32), 1: np.zeros((0, 1), dtype=np.float32)},
        labels=np.array([0, 1], dtype=np.int64),
    )
    return full, target_only


def test_tiny_semantic_tree_probe_changes_when_support_removed():
    full, target_only = _full_and_target_only_graphs()
    paths = [(), (1,)]
    full_tree = build_semantic_tree_features(full, target_type=0, paths=paths, feature_width=1, type_ids=(0, 1))
    target_tree = build_semantic_tree_features(target_only, target_type=0, paths=paths, feature_width=1, type_ids=(0, 1))

    row = semantic_tree_delta_row(
        dataset="tiny",
        seed=1,
        method="target-only-empty-support",
        requested_support_ratio=0.0,
        primary_eval_mode="compressed_projected",
        max_paths=2,
        paths=paths,
        compressed_tree=target_tree,
        full_tree=full_tree,
        target_only_tree=target_tree,
        original_target_nodes=np.array([0, 1], dtype=np.int64),
        original_to_compressed=np.array([0, 1, 0], dtype=np.int64),
        original_to_target_only=np.array([0, 1, 0], dtype=np.int64),
    )

    assert row["path_count"] == 2
    assert row["nonself_path_count"] == 1
    assert row["support_dependent_path_count"] >= 1
    assert row["tree_tensor_l2_delta_vs_full"] > 0.0
    assert row["target_path_feature_changed_fraction"] > 0.0
    assert row["allclose_to_full"] is False
