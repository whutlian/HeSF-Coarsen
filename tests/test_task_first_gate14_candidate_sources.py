import numpy as np

from hesf_coarsen.task_first.candidates import (
    build_hybrid_task_aware_candidates,
    build_relation_response_knn_candidates,
    build_target_response_signature_knn_candidates,
)
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from tests.test_task_first_state import make_target_support_graph


def test_target_response_signature_knn_excludes_target_nodes():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    store, _diag = build_target_response_signature_knn_candidates(
        graph, state, target_type=0, candidate_k=4
    )

    for block in store.iter_pair_blocks():
        for u, v, _score in block:
            assert graph.node_type[int(u)] != 0
            assert graph.node_type[int(v)] != 0


def test_relation_response_knn_respects_same_type_candidates():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    store, _diag = build_relation_response_knn_candidates(graph, state, target_type=0, candidate_k=4)

    for block in store.iter_pair_blocks():
        for u, v, _score in block:
            assert graph.node_type[int(u)] == graph.node_type[int(v)]


def test_hybrid_task_aware_deduplicates_sources():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    store, diag = build_hybrid_task_aware_candidates(graph, state, target_type=0, candidate_k=4)
    pairs = [tuple(sorted((int(u), int(v)))) for block in store.iter_pair_blocks() for u, v, _score in block]

    assert len(pairs) == len(set(pairs))
    assert diag["candidate_source"] == "hybrid_task_aware"
