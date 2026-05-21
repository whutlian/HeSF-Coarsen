import numpy as np

from hesf_coarsen.task_first.config import SupportPurityConfig, TaskFirstConfig
from hesf_coarsen.task_first.state import build_task_first_state
from hesf_coarsen.task_first.support_purity import (
    FOOTPRINT_KNOWN,
    FOOTPRINT_UNKNOWN_TARGET_CONNECTED,
    build_support_class_footprints,
    merge_is_purity_allowed,
    purity_v2_diagnostics,
)
from tests.test_task_first_state import make_target_support_graph


def test_purity_v2_twohop_and_hybrid_modes_are_available_and_stable():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, False, False, False, False])

    onehop_cfg = TaskFirstConfig(
        target_node_type=0,
        support_purity=SupportPurityConfig(support_footprint_mode="onehop_train"),
    )
    hybrid_cfg = TaskFirstConfig(
        target_node_type=0,
        support_purity=SupportPurityConfig(support_footprint_mode="hybrid_propagated"),
    )

    onehop = build_support_class_footprints(graph, labels, train_mask, onehop_cfg)
    hybrid = build_support_class_footprints(graph, labels, train_mask, hybrid_cfg)

    assert np.count_nonzero(hybrid.sum(axis=1) > 0.0) >= np.count_nonzero(onehop.sum(axis=1) > 0.0)
    assert np.isfinite(hybrid).all()
    assert hybrid.shape == onehop.shape


def test_purity_v2_blocks_known_unknown_structured_by_default():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, False, False, False, False])
    cfg = TaskFirstConfig(
        target_node_type=0,
        support_purity=SupportPurityConfig(
            zero_policy="purity_v2",
            support_footprint_mode="onehop_train",
        ),
    )
    state = build_task_first_state(graph, labels, train_mask, cfg)

    assert state.support_footprint_states[2] == FOOTPRINT_KNOWN
    assert state.support_footprint_states[4] == FOOTPRINT_UNKNOWN_TARGET_CONNECTED
    assert not merge_is_purity_allowed(2, 4, state, cfg)


def test_purity_v2_diagnostics_include_known_unknown_shares():
    graph = make_target_support_graph()
    labels = np.asarray(graph.labels)
    train_mask = np.array([True, False, False, False, False])
    cfg = TaskFirstConfig(
        target_node_type=0,
        support_purity=SupportPurityConfig(zero_policy="purity_v2"),
    )
    state = build_task_first_state(graph, labels, train_mask, cfg)

    diag = purity_v2_diagnostics(state)

    assert diag["known_support_share"] > 0.0
    assert diag["unknown_structured_share"] > 0.0
    assert "zero_footprint_support_share" in diag
