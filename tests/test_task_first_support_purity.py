import numpy as np

from hesf_coarsen.task_first.config import SupportPurityConfig, TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_purity import (
    FOOTPRINT_KNOWN,
    FOOTPRINT_UNKNOWN_TARGET_CONNECTED,
    merge_is_purity_allowed,
    support_purity_pair_kind,
)
from tests.test_task_first_state import make_target_support_graph


def test_unknown_blocks_known_rejects_zero_footprint_known_merge():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, False, False, False, False])
    cfg = TaskFirstConfig(
        target_node_type=0,
        support_purity=SupportPurityConfig(zero_policy="unknown_blocks_known"),
    )
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert state.support_footprint_states[2] == FOOTPRINT_KNOWN
    assert state.support_footprint_states[4] == FOOTPRINT_UNKNOWN_TARGET_CONNECTED
    assert support_purity_pair_kind(2, 4, state) == "known_unknown"
    assert not merge_is_purity_allowed(2, 4, state, cfg)


def test_zero_as_no_conflict_preserves_legacy_policy():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, False, False, False, False])
    cfg = TaskFirstConfig(
        target_node_type=0,
        support_purity=SupportPurityConfig(zero_policy="zero_as_no_conflict", js_merge_block_threshold=1.0),
    )
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert merge_is_purity_allowed(2, 4, state, cfg)
