from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def make_tiny_official_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    labels = np.array([0, 1, 2, 0, 1, 2, -1, -1, -1], dtype=np.int64)
    features = {
        0: np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.8, 0.2],
                [0.1, 0.1, 0.8],
            ],
            dtype=np.float32,
        ),
        1: np.array([[1.0, 0.2], [0.1, 0.9], [0.4, 0.6]], dtype=np.float32),
    }
    relations = {
        0: RelationAdj(
            src=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
            dst=np.array([6, 6, 7, 7, 8, 8], dtype=np.int64),
            weight=np.ones(6, dtype=np.float32),
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([6, 6, 7, 7, 8, 8], dtype=np.int64),
            dst=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
            weight=np.ones(6, dtype=np.float32),
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
    }
    relation_specs = {
        0: RelationSpec(0, "paper__to__author", 0, 1),
        1: RelationSpec(1, "author__to__paper", 1, 0),
    }
    return HeteroGraph(
        num_nodes=9,
        node_type=node_type,
        relations=relations,
        relation_specs=relation_specs,
        features=features,
        labels=labels,
    )


def test_export_hgb_graph_preserves_mapping_splits_labels_types_and_edges(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph

    graph = make_tiny_official_graph()
    result = export_hgb_graph(
        graph,
        dataset_name="Tiny",
        method_name="H6",
        seed=23456,
        support_ratio=0.30,
        output_dir=tmp_path,
        target_type="type_0",
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4, 5], dtype=np.int64),
        labels=graph.labels,
        original_target_ids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        metadata={"primary_eval_mode": "compressed_projected"},
    )

    export_dir = Path(result["export_dir"])
    assert result["target_type"] == "type_0"
    assert result["target_count"] == 6
    assert result["mapping_bijective"] is True
    assert result["split_disjoint"] is True
    assert result["no_test_label_export_leakage"] is True
    assert result["label_distribution_train"] == {"0": 1, "1": 1, "2": 1}
    assert result["num_nodes_by_type"] == {"type_0": 6, "type_1": 3}
    assert result["num_edges_by_relation"] == {"paper__to__author": 6, "author__to__paper": 6}

    assert (export_dir / "node_features" / "type_0.npy").exists()
    assert (export_dir / "node_features" / "type_1.npy").exists()
    assert (export_dir / "edges" / "paper__to__author.npy").exists()
    assert (export_dir / "edges" / "author__to__paper.npy").exists()
    assert np.load(export_dir / "splits" / "train_idx.npy").tolist() == [0, 1, 2]
    assert np.load(export_dir / "splits" / "val_idx.npy").tolist() == [3]
    assert np.load(export_dir / "splits" / "test_idx.npy").tolist() == [4, 5]
    assert np.load(export_dir / "labels.npy").tolist() == [0, 1, 2, 0, 1, 2]
    assert np.load(export_dir / "splits" / "train_labels.npy").tolist() == [0, 1, 2]
    assert np.load(export_dir / "splits" / "val_labels.npy").tolist() == [0]
    assert not (export_dir / "splits" / "test_labels_for_training.npy").exists()

    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["dataset"] == "Tiny"
    assert metadata["method"] == "H6"
    assert metadata["relation_type_names"] == ["paper__to__author", "author__to__paper"]
    assert metadata["relation_schemas"] == [
        {"name": "paper__to__author", "src_type": "type_0", "dst_type": "type_1"},
        {"name": "author__to__paper", "src_type": "type_1", "dst_type": "type_0"},
    ]
    audit = json.loads((export_dir / "export_audit.json").read_text())
    assert audit["target_count_original"] == audit["target_count_exported"] == 6


def test_export_hgb_graph_preserves_zero_count_support_types(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.graph_export import export_hgb_graph

    node_type = np.array([0, 0, 0], dtype=np.int32)
    labels = np.array([0, 1, 2], dtype=np.int64)
    graph = HeteroGraph(
        num_nodes=3,
        node_type=node_type,
        relations={
            0: RelationAdj(
                src=np.array([], dtype=np.int64),
                dst=np.array([], dtype=np.int64),
                weight=np.array([], dtype=np.float32),
                src_type=0,
                dst_type=1,
                relation_id=0,
            )
        },
        relation_specs={0: RelationSpec(0, "paper__to__author", 0, 1)},
        features={
            0: np.eye(3, dtype=np.float32),
            1: np.zeros((0, 2), dtype=np.float32),
        },
        labels=labels,
    )
    result = export_hgb_graph(
        graph,
        dataset_name="Tiny",
        method_name="target-only",
        seed=23456,
        support_ratio=None,
        output_dir=tmp_path,
        target_type="type_0",
        train_idx=np.array([0], dtype=np.int64),
        val_idx=np.array([1], dtype=np.int64),
        test_idx=np.array([2], dtype=np.int64),
        labels=labels,
    )

    export_dir = Path(result["export_dir"])
    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert result["export_status"] == "success"
    assert metadata["num_nodes_by_type"] == {"type_0": 3, "type_1": 0}
    assert (export_dir / "node_features" / "type_1.npy").exists()
    assert np.load(export_dir / "node_features" / "type_1.npy").shape == (0, 2)
