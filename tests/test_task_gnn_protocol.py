import numpy as np

from hesf_coarsen.eval.task_gnn import (
    f1_scores,
    resolve_target_node_type,
    select_task_protocol_split,
    train_only_coarse_labels,
)
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def test_train_only_coarse_labels_do_not_use_test_labels():
    labels = np.array([0, 1, 1, 1], dtype=np.int64)
    original_to_coarse = np.array([0, 0, 0, 1], dtype=np.int64)
    train_nodes = np.array([0], dtype=np.int64)
    test_nodes = np.array([1, 2, 3], dtype=np.int64)

    coarse_labels, diagnostics = train_only_coarse_labels(
        labels,
        original_to_coarse,
        train_nodes,
        num_coarse_nodes=2,
        test_nodes=test_nodes,
    )

    assert coarse_labels.tolist() == [0, -1]
    assert diagnostics["train_only_label_coverage"] == 0.5
    assert diagnostics["test_label_leakage_check"] == "passed"
    assert diagnostics["cluster_train_label_entropy"] == 0.0


def test_protocol_split_resolves_ogbn_paper_type_and_reports_split_sanity():
    node_type = np.array([0, 0, 1, 1, 3, 3, 3, 3, 3, 3], dtype=np.int32)
    labels = np.array([-1, -1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    graph = HeteroGraph(
        num_nodes=len(node_type),
        node_type=node_type,
        relations={
            0: RelationAdj(np.array([0, 1]), np.array([4, 5]), None, 0, 3, 0),
            1: RelationAdj(np.array([4, 5]), np.array([2, 3]), None, 3, 1, 1),
        },
        relation_specs={
            0: RelationSpec(0, "author__writes__paper", 0, 3),
            1: RelationSpec(1, "paper__has_topic__field_of_study", 3, 1),
        },
        labels=labels,
    )

    assert resolve_target_node_type(graph, "paper") == 3

    train_nodes, val_nodes, test_nodes, diagnostics = select_task_protocol_split(
        graph,
        labels,
        seed=12345,
        target_node_type="paper",
        train_fraction=0.5,
        val_fraction=1 / 6,
    )

    selected = np.concatenate([train_nodes, val_nodes, test_nodes])
    assert set(selected.tolist()) == {4, 5, 6, 7, 8, 9}
    assert np.all(graph.node_type[selected] == 3)
    assert diagnostics["target_node_type"] == "paper"
    assert diagnostics["target_node_type_id"] == 3
    assert diagnostics["num_labeled_nodes_train"] > 0
    assert diagnostics["num_labeled_nodes_val"] > 0
    assert diagnostics["num_labeled_nodes_test"] > 0
    assert diagnostics["num_classes_present_train"] == 2
    assert diagnostics["official_split_consistency"] == "synthetic_stratified_target_type"
    assert diagnostics["coarse_train_label_source"] == "train_only"


def test_protocol_split_accepts_official_target_type_split():
    node_type = np.array([0, 0, 3, 3, 3, 3, 3, 3], dtype=np.int32)
    labels = np.array([-1, -1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    graph = HeteroGraph(
        num_nodes=len(node_type),
        node_type=node_type,
        relations={
            0: RelationAdj(np.array([0, 1]), np.array([2, 3]), None, 0, 3, 0),
            1: RelationAdj(np.array([2, 3]), np.array([4, 5]), None, 3, 3, 1),
        },
        relation_specs={
            0: RelationSpec(0, "author__writes__paper", 0, 3),
            1: RelationSpec(1, "paper__cites__paper", 3, 3),
        },
        labels=labels,
    )

    train_nodes, val_nodes, test_nodes, diagnostics = select_task_protocol_split(
        graph,
        labels,
        seed=12345,
        target_node_type="paper",
        official_split_nodes={
            "train": np.array([2, 3], dtype=np.int64),
            "valid": np.array([4], dtype=np.int64),
            "test": np.array([5, 6, 7], dtype=np.int64),
        },
    )

    assert train_nodes.tolist() == [2, 3]
    assert val_nodes.tolist() == [4]
    assert test_nodes.tolist() == [5, 6, 7]
    assert diagnostics["task_split_policy"] == "official"
    assert diagnostics["official_split_consistency"] == "official_target_type"
    assert diagnostics["num_classes_present_train"] == 2
    assert diagnostics["num_classes_present_val"] == 1
    assert diagnostics["num_classes_present_test"] == 2


def test_f1_scores_can_use_eval_present_macro_policy():
    truth = np.array([0, 0], dtype=np.int64)
    pred = np.array([0, 1], dtype=np.int64)

    union_scores = f1_scores(truth, pred)
    eval_present_scores = f1_scores(truth, pred, macro_empty_class_policy="eval_present")

    assert np.isclose(union_scores["macro_f1"], 1 / 3)
    assert np.isclose(eval_present_scores["macro_f1"], 2 / 3)
