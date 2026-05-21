import numpy as np
import pytest

from tests.gate15_test_utils import make_gate15_graph, split_masks


def _evaluate(mode: str):
    pytest.importorskip("torch")
    from hesf_coarsen.eval.hettree_task import evaluate_hettree_task

    graph = make_gate15_graph()
    train_mask, val_mask, test_mask = split_masks()
    return evaluate_hettree_task(
        graph,
        graph,
        np.arange(graph.num_nodes, dtype=np.int64),
        seed=123,
        epochs=2,
        hidden_dim=8,
        device="cpu",
        target_node_type=0,
        official_split_nodes={
            "train": np.flatnonzero(train_mask),
            "val": np.flatnonzero(val_mask),
            "test": np.flatnonzero(test_mask),
        },
        primary_eval_mode=mode,
        early_stopping=True,
        patience=2,
    ).metrics


def test_compressed_projected_mode_is_primary_metric():
    metrics = _evaluate("compressed_projected")

    assert metrics["primary_eval_mode"] == "compressed_projected"
    assert metrics["primary_task_metric_name"] == "projected_original_macro_f1"
    assert metrics["macro_f1"] == metrics["projected_original_macro_f1"]
    assert metrics["accuracy"] == metrics["projected_original_accuracy"]
    assert "validation_macro_f1" in metrics
    assert "projected_original_val_macro_f1" in metrics
    assert "projected_vs_transfer_macro_gap" in metrics
    assert "best_epoch" in metrics
    assert "early_stopped" in metrics


def test_original_transfer_mode_remains_available_as_diagnostic_primary():
    metrics = _evaluate("original_transfer")

    assert metrics["primary_eval_mode"] == "original_transfer"
    assert metrics["primary_task_metric_name"] == "transfer_original_macro_f1"
    assert metrics["macro_f1"] == metrics["transfer_original_macro_f1"]
    assert metrics["accuracy"] == metrics["transfer_original_accuracy"]
