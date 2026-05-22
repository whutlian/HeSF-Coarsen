import numpy as np

from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from hesf_coarsen.task_first.selection.selector import select_support_nodes
from hesf_coarsen.task_first.selection.validation_selector import build_support_block_keys, group_support_by_block


def _features():
    return {
        "support_nodes": np.array([10, 11, 12, 13], dtype=np.int64),
        "support_node_types": np.array([1, 1, 2, 2], dtype=np.int32),
        "component_matrices": {
            "relation_profile": np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
            "anchor_distribution": np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
            "class_footprint": np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
            "degree_profile": np.array([[4, 0], [3, 0], [1, 0], [1, 0]], dtype=np.float32),
        },
    }


def test_block_key_helpers_group_support_nodes_by_context():
    keys = build_support_block_keys(_features(), mode="default")
    groups = group_support_by_block(keys)

    assert len(groups) == 2
    assert any(indices.tolist() == [0, 1] for indices in groups.values())
    assert any(indices.tolist() == [2, 3] for indices in groups.values())


def test_real_validation_block_greedy_uses_injected_feedback_not_proxy_order():
    features = _features()
    # Proxy score favors nodes 10/11, but validation feedback favors block 12/13.
    importance = np.array([10.0, 9.0, 1.0, 1.0], dtype=np.float32)

    def fake_validation(selected_nodes):
        selected = {int(node) for node in selected_nodes}
        return 1.0 if 12 in selected else 0.1

    selected = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(
            selector="real_validation_block_greedy",
            candidate_pool_size=2,
            max_validation_greedy_steps=1,
            min_gain=-1.0,
        ),
        validation_evaluator=fake_validation,
    )

    assert 12 in selected["selected_support_nodes"]
    assert selected["diagnostics"]["selector_uses_true_validation_feedback"] is True
    assert selected["diagnostics"]["validation_trial_count"] > 0
    assert selected["diagnostics"]["validation_greedy_steps"] > 0
    assert selected["diagnostics"]["validation_greedy_best_gain_mean"] > 0.0
    assert selected["diagnostics"]["selector_uses_test_labels"] is False
