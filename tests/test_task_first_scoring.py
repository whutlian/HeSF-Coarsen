import numpy as np
from dataclasses import replace

from hesf_coarsen.task_first.config import TaskFirstConfig, TaskFirstScoringConfig
from hesf_coarsen.task_first.constraints import allow_task_first_merge
from hesf_coarsen.task_first.relation_response import delta_relation_response_for_merge
from hesf_coarsen.task_first.scoring import compute_task_first_delta, score_task_first_delta
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_coverage import delta_support_coverage_for_merge
from hesf_coarsen.task_first.support_purity import merge_is_purity_allowed
from tests.test_task_first_state import make_target_support_graph


def test_purity_hard_block_rejects_high_js_divergence_pair():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert merge_is_purity_allowed(2, 3, state, cfg)
    assert not merge_is_purity_allowed(2, 4, state, cfg)


def test_coverage_delta_is_larger_when_merge_harms_same_anchor_neighborhood():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    same_anchor = delta_support_coverage_for_merge(2, 3, state, cfg)
    different_anchor = delta_support_coverage_for_merge(2, 4, state, cfg)

    assert same_anchor > different_anchor


def test_constraints_enforce_support_only_same_type_partition_and_purity():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert allow_task_first_merge(graph, 2, 3, None, state, cfg)
    assert not allow_task_first_merge(graph, 0, 2, None, state, cfg)
    assert not allow_task_first_merge(graph, 0, 1, None, state, cfg)
    assert not allow_task_first_merge(graph, 2, 4, None, state, cfg)

    graph.partitions[3] = 1
    assert not allow_task_first_merge(graph, 2, 3, None, state, cfg)


def test_task_first_delta_contains_all_terms_and_weighted_score():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    delta = compute_task_first_delta(graph, 2, 3, state, cfg)
    scored = score_task_first_delta(delta, cfg)

    assert scored.delta_target_spec >= 0.0
    assert scored.delta_rel_response >= 0.0
    assert scored.delta_support_coverage >= 0.0
    assert scored.delta_support_purity >= 0.0
    assert scored.delta_feat >= 0.0
    assert scored.score_task_first > 0.0


def test_delta_relation_response_for_merge_accepts_support_pair_interface():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(target_node_type=0)
    state = build_task_first_state(graph, labels, train_mask, cfg)

    delta = delta_relation_response_for_merge(graph, 2, 3, state, cfg)

    assert delta >= 0.0


def test_local_surrogate_delta_uses_fast_nonnegative_terms_and_same_score_formula():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, True, False, False, False])
    cfg = TaskFirstConfig(
        target_node_type=0,
        scoring=TaskFirstScoringConfig(pair_delta_mode="local_surrogate"),
    )
    state = build_task_first_state(graph, labels, train_mask, cfg)

    delta = compute_task_first_delta(graph, 2, 3, state, cfg)

    expected = (
        cfg.scoring.lambda_target_spec * delta.delta_target_spec
        + cfg.scoring.lambda_rel_response * delta.delta_rel_response
        + cfg.scoring.lambda_support_coverage * delta.delta_support_coverage
        + cfg.scoring.lambda_support_purity * delta.delta_support_purity
        + cfg.scoring.lambda_feat * delta.delta_feat
    )
    assert delta.delta_target_spec >= 0.0
    assert delta.delta_rel_response >= 0.0
    assert delta.score_task_first == expected
