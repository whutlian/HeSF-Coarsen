import numpy as np

from hesf_coarsen.task_first.candidates import (
    build_class_footprint_knn_candidates,
    build_target_anchor_co_support_candidates,
    build_target_response_knn_candidates,
)
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from tests.test_task_first_state import make_target_support_graph


def test_target_aware_candidate_sources_emit_support_only_pairs():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    builders = [
        build_target_anchor_co_support_candidates,
        build_class_footprint_knn_candidates,
        build_target_response_knn_candidates,
    ]
    for builder in builders:
        store, diag = builder(graph, state, target_type=0, candidate_k=4)
        assert diag["candidate_pairs_retained"] >= 0
        for block in store.iter_pair_blocks():
            for u, v, _score in block:
                assert graph.node_type[int(u)] != 0
                assert graph.node_type[int(v)] != 0
