import numpy as np
import pytest

from hesf_coarsen.accuracy.full_target_inference import evaluate_full_target_inference
from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.target_preserve import build_target_preserving_assignment
from hesf_coarsen.io.edge_list import generate_synthetic_graph


def test_full_target_inference_reports_mode_b_protocol_tags():
    pytest.importorskip("torch")
    graph = generate_synthetic_graph(num_users=14, num_items=7, num_tags=4, seed=22)
    base = Assignment(
        assignment=np.array([0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13], dtype=np.int64),
        supernode_type=np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int32),
    )
    hybrid_assignment = build_target_preserving_assignment(graph, base, target_node_type=0)
    hybrid = coarsen_graph(graph, hybrid_assignment)

    metrics = evaluate_full_target_inference(
        original=graph,
        hybrid=hybrid,
        original_to_hybrid=hybrid_assignment.assignment,
        target_node_type=0,
        model_name="sehgnn_lite",
        seed=22,
        epochs=1,
        hidden_dim=8,
        device="cpu",
    ).metrics

    assert metrics["eval_mode"] == "full_target_inference"
    assert metrics["full_target_inference"] is True
    assert metrics["model_name"] == "sehgnn_lite"
    assert "macro_f1" in metrics
