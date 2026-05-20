import numpy as np
import pytest

from hesf_coarsen.task_first.config import TaskFirstConfig, TaskFirstScoringConfig
from hesf_coarsen.task_first.scoring import compute_task_first_delta
from hesf_coarsen.task_first.state import build_task_first_state
from tests.test_task_first_state import make_target_support_graph


def test_exact_pair_isolated_alias_matches_exact_delta():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    exact_cfg = TaskFirstConfig(target_node_type=0, scoring=TaskFirstScoringConfig(pair_delta_mode="exact"))
    alias_cfg = TaskFirstConfig(target_node_type=0, scoring=TaskFirstScoringConfig(pair_delta_mode="exact_pair_isolated"))
    state = build_task_first_state(graph, labels, train_mask, exact_cfg)

    exact = compute_task_first_delta(graph, 2, 3, state, exact_cfg)
    alias = compute_task_first_delta(graph, 2, 3, state, alias_cfg)

    assert alias.delta_target_spec == pytest.approx(exact.delta_target_spec)
    assert alias.delta_rel_response == pytest.approx(exact.delta_rel_response)


def test_response_signature_mode_returns_nonnegative_score():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, scoring=TaskFirstScoringConfig(pair_delta_mode="response_signature"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    delta = compute_task_first_delta(graph, 2, 3, state, cfg)

    assert delta.delta_target_spec >= 0.0
    assert delta.score_task_first >= 0.0


def test_stateful_approx_is_explicitly_not_implemented():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0, scoring=TaskFirstScoringConfig(pair_delta_mode="stateful_approx"))
    state = build_task_first_state(graph, labels, train_mask, cfg)

    with pytest.raises(NotImplementedError):
        compute_task_first_delta(graph, 2, 3, state, cfg)
