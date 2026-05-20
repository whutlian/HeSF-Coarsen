import numpy as np
from dataclasses import replace

from hesf_coarsen.task_first.config import TaskFirstConfig, TaskFirstScoringConfig
from hesf_coarsen.task_first.scoring import compute_task_first_delta
from hesf_coarsen.task_first.state import build_task_first_state
from tests.test_task_first_state import make_target_support_graph


def test_large_lambda_relation_response_changes_pair_ordering():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg0 = TaskFirstConfig(
        target_node_type=0,
        scoring=TaskFirstScoringConfig(pair_delta_mode="local_surrogate", lambda_rel_response=0.0),
    )
    cfg_big = replace(
        cfg0,
        scoring=replace(cfg0.scoring, lambda_rel_response=10.0, lambda_target_spec=0.0, lambda_support_coverage=0.0, lambda_support_purity=0.0, lambda_feat=0.0),
    )
    state = build_task_first_state(graph, labels, train_mask, cfg0)

    no_rel = compute_task_first_delta(graph, 2, 3, state, cfg0).score_task_first
    with_rel = compute_task_first_delta(graph, 2, 3, state, cfg_big).score_task_first

    assert with_rel != no_rel
