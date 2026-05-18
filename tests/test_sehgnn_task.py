import numpy as np
import pytest

from hesf_coarsen.eval.sehgnn_task import evaluate_sehgnn_task
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_sehgnn_task_runs_on_synthetic_graph():
    pytest.importorskip("torch")
    graph = generate_synthetic_graph(num_users=14, num_items=8, num_tags=5, seed=1515)
    mapping = np.arange(graph.num_nodes, dtype=np.int64)

    metrics = evaluate_sehgnn_task(
        graph,
        graph,
        mapping,
        seed=1515,
        hidden_dim=8,
        epochs=1,
        device="cpu",
        train_fraction=0.5,
        val_fraction=0.2,
    ).metrics

    assert metrics["model"] == "sehgnn_lite"
    assert metrics["architecture_reference"] == "ICT-GIMLab/SeHGNN"
    assert metrics["skipped"] is False
    assert metrics["path_count"] == metrics["sehgnn_num_channels"]
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "accuracy" in metrics
