import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def _chain_graph() -> HeteroGraph:
    node_type = np.zeros(6, dtype=np.int32)
    src = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    dst = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    rel = RelationAdj(src=src, dst=dst, weight=np.ones(len(src), dtype=np.float32), src_type=0, dst_type=0, relation_id=0)
    return HeteroGraph(num_nodes=6, node_type=node_type, relations={0: rel}, relation_specs={0: RelationSpec(0, "r0", 0, 0)})


def test_lowpass_reconstruction_identity_beats_collapsed_assignment():
    from hesf_coarsen.eval.structure_tasks import evaluate_lowpass_signal_reconstruction

    graph = _chain_graph()
    identity = Assignment(np.arange(graph.num_nodes), graph.node_type)
    collapsed = Assignment(np.array([0, 0, 1, 1, 2, 2]), np.zeros(3, dtype=np.int32))
    id_row = evaluate_lowpass_signal_reconstruction(graph, coarsen_graph(graph, identity), identity, seed=5, num_signals=3)
    collapsed_row = evaluate_lowpass_signal_reconstruction(graph, coarsen_graph(graph, collapsed), collapsed, seed=5, num_signals=3)
    assert id_row["signal_mse"] <= collapsed_row["signal_mse"]
    assert "signal_cosine" in id_row
    assert "per_type_signal_mse" in id_row


def test_feature_free_label_propagation_reports_refinement_curve():
    from hesf_coarsen.eval.structure_tasks import evaluate_feature_free_label_propagation

    graph = _chain_graph()
    assignment = Assignment(np.array([0, 0, 1, 1, 2, 2]), np.zeros(3, dtype=np.int32))
    row = evaluate_feature_free_label_propagation(graph, coarsen_graph(graph, assignment), assignment, seed=7)
    for key in ("projected", "refined@0", "refined@1", "refined@3", "refined@5", "best", "AUC"):
        assert key in row
        assert 0.0 <= float(row[key]) <= 1.0
