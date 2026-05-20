import numpy as np
import pytest

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.eval_protocol import evaluate_real_full_target_protocol
from hesf_coarsen.task_first.pipeline import build_support_only_task_first_coarsening
from tests.test_task_first_state import make_target_support_graph


class LiteBackbone:
    fidelity = "lite"

    def fit(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        return np.array([], dtype=np.int64)


def test_support_only_pipeline_preserves_all_target_singletons_and_emits_diagnostics():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    store = ArrayCandidateStore(graph.node_type, K=4, same_type_only=True)
    store.add(2, 3, 0.1, "bucket")
    store.add(2, 4, 0.2, "bucket")
    store.add(0, 2, 0.0, "fallback")

    result = build_support_only_task_first_coarsening(
        graph,
        store,
        labels,
        train_mask,
        cfg,
    )

    target_nodes = np.flatnonzero(graph.node_type == 0)
    sizes = result.assignment.cluster_sizes()
    assert result.assignment.assignment[target_nodes].tolist() == [0, 1]
    assert np.all(sizes[result.assignment.assignment[target_nodes]] == 1)
    assert result.graph.num_nodes < graph.num_nodes
    for key in (
        "target_spec_error",
        "relation_response_error",
        "support_coverage_error",
        "support_purity_error",
        "num_support_candidates_scored",
        "num_support_candidates_rejected_by_purity",
        "num_support_candidates_rejected_by_constraints",
    ):
        assert key in result.diagnostics


def test_real_full_target_protocol_rejects_lite_backbones():
    graph = make_target_support_graph()
    with pytest.raises(ValueError, match="official|faithful"):
        evaluate_real_full_target_protocol(graph, graph, LiteBackbone())
