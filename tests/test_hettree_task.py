import numpy as np
import pytest

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _tiny_typed_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    labels = np.array([0, 1, 0, -1, -1, -1], dtype=np.int64)
    relations = {
        0: RelationAdj(
            src=np.array([0, 1, 2], dtype=np.int64),
            dst=np.array([3, 3, 4], dtype=np.int64),
            weight=None,
            src_type=0,
            dst_type=1,
            relation_id=0,
        ),
        1: RelationAdj(
            src=np.array([3, 3, 4], dtype=np.int64),
            dst=np.array([0, 1, 2], dtype=np.int64),
            weight=None,
            src_type=1,
            dst_type=0,
            relation_id=1,
        ),
        2: RelationAdj(
            src=np.array([5], dtype=np.int64),
            dst=np.array([3], dtype=np.int64),
            weight=None,
            src_type=2,
            dst_type=1,
            relation_id=2,
        ),
    }
    specs = {
        0: RelationSpec(0, "paper__to__author", 0, 1),
        1: RelationSpec(1, "author__to__paper", 1, 0),
        2: RelationSpec(2, "term__to__author", 2, 1),
    }
    features = {
        0: np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
        1: np.array([[0.5, 1.0, 0.0], [0.1, 0.2, 0.9]], dtype=np.float32),
        2: np.array([[0.7]], dtype=np.float32),
    }
    return HeteroGraph(
        num_nodes=6,
        node_type=node_type,
        relations=relations,
        relation_specs=specs,
        features=features,
        labels=labels,
    )


def test_hettree_infers_labeled_target_type():
    from hesf_coarsen.eval.hettree_task import infer_target_node_type

    assert infer_target_node_type(_tiny_typed_graph()) == 0


def test_hettree_builds_consistent_path_features_for_mixed_feature_dims():
    from hesf_coarsen.eval.hettree_task import build_semantic_tree_features, enumerate_target_paths

    graph = _tiny_typed_graph()
    paths = enumerate_target_paths(graph, target_type=0, max_hops=2)
    features = build_semantic_tree_features(graph, target_type=0, paths=paths)

    assert features.target_nodes.tolist() == [0, 1, 2]
    assert features.tensor.ndim == 3
    assert features.tensor.shape[0] == 3
    assert features.tensor.shape[1] == len(paths)
    assert features.tensor.shape[2] >= 3
    assert () in paths


def test_hettree_task_eval_reports_accuracy_on_projected_original_test():
    pytest.importorskip("torch")
    from hesf_coarsen.eval.hettree_task import evaluate_hettree_task

    graph = _tiny_typed_graph()
    mapping = np.arange(graph.num_nodes, dtype=np.int64)

    metrics = evaluate_hettree_task(
        graph,
        graph,
        mapping,
        seed=42,
        epochs=1,
        hidden_dim=8,
        device="cpu",
        train_fraction=0.34,
        val_fraction=0.0,
    ).metrics

    assert metrics["model"] == "hettree_lite"
    assert metrics["eval_on"] == "original_test_transfer"
    for name in ("macro_f1", "micro_f1", "accuracy"):
        assert name in metrics
        assert 0.0 <= metrics[name] <= 1.0
