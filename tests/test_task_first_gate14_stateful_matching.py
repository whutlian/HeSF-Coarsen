import numpy as np

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.stateful_matching import (
    TaskFirstClusterSignature,
    run_stateful_signature_matching,
)
from tests.test_task_first_state import make_target_support_graph


def test_stateful_signature_updates_after_merge():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)
    store = ArrayCandidateStore(graph.node_type, K=4, same_type_only=True)
    store.add(2, 3, 0.0, "unit")
    store.add(3, 4, 0.1, "unit")

    result = run_stateful_signature_matching(graph, store, state, cfg, max_support_merges=1)

    assert result.diagnostics["stateful_update_count"] == 1
    assert result.diagnostics["rescore_count"] > 0
    assert result.diagnostics["stateful_signature_drift"] > 0.0
    assert result.selected_pairs.shape == (1, 2)


def test_stateful_sequence_can_differ_from_static_input_order():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)
    store = ArrayCandidateStore(graph.node_type, K=4, same_type_only=True)
    store.add(2, 4, 0.0, "unit")
    store.add(2, 3, 1.0, "unit")

    result = run_stateful_signature_matching(graph, store, state, cfg, max_support_merges=1)

    assert tuple(result.selected_pairs[0].tolist()) == (2, 3)


def test_cluster_signature_dataclass_has_required_fields():
    sig = TaskFirstClusterSignature(
        cluster_id=1,
        node_ids=np.array([2, 3]),
        node_type=1,
        target_response_signature=np.ones(2),
        relation_response_signature=np.ones(2),
        class_footprint=np.ones(2),
        anchor_distribution=np.ones(2),
        feature_centroid=None,
        support_size=2,
    )

    assert sig.support_size == 2
    assert sig.feature_centroid is None
