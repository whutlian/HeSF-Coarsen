import numpy as np

from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from hesf_coarsen.task_first.selection.selector import select_support_nodes


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


def test_real_occlusion_block_selector_scores_blocks_by_validation_drop():
    features = _features()
    importance = np.array([10.0, 9.0, 1.0, 1.0], dtype=np.float32)

    def fake_occlusion(occluded_nodes):
        occluded = {int(node) for node in occluded_nodes}
        if not occluded:
            return {"validation_macro_f1": 1.0, "validation_cross_entropy": 0.2, "margin": 0.9}
        if 12 in occluded:
            return {"validation_macro_f1": 0.4, "validation_cross_entropy": 0.9, "margin": 0.2}
        return {"validation_macro_f1": 0.9, "validation_cross_entropy": 0.3, "margin": 0.8}

    selected = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(selector="real_occlusion_block_selector", occlusion_candidate_pool_size=2),
        occlusion_evaluator=fake_occlusion,
    )

    assert 12 in selected["selected_support_nodes"]
    assert selected["diagnostics"]["occlusion_trial_count"] == 2
    assert selected["diagnostics"]["occlusion_delta_ce_max"] > 0.0
    assert selected["diagnostics"]["selector_uses_test_labels"] is False
    scores = selected["occlusion_block_scores"]
    assert len(scores) == 2
    assert scores[0]["final_block_importance"] > scores[1]["final_block_importance"]
    assert scores[0]["selected"] is True


def test_occlusion_scores_change_when_block_is_masked():
    features = _features()
    importance = np.ones(4, dtype=np.float32)
    calls = []

    def fake_occlusion(occluded_nodes):
        calls.append(tuple(int(node) for node in occluded_nodes))
        return 1.0 - 0.1 * len(occluded_nodes)

    selected = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(selector="real_occlusion_block_selector", occlusion_candidate_pool_size=2),
        occlusion_evaluator=fake_occlusion,
    )

    assert selected["diagnostics"]["occlusion_trial_count"] > 0
    assert any(len(call) > 0 for call in calls)
