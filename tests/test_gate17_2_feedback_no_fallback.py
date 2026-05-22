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


def test_real_validation_no_fallback_rejects_zero_gain_and_leaves_budget_unfilled():
    selected = select_support_nodes(
        _features(),
        np.array([10.0, 9.0, 1.0, 1.0], dtype=np.float32),
        0.5,
        SupportSelectorConfig(
            selector="real_validation_block_greedy",
            candidate_pool_size=2,
            max_validation_greedy_steps=3,
            min_gain=1.0e-4,
            allow_proxy_fill=False,
        ),
        validation_evaluator=lambda selected_nodes: 0.5,
    )

    diagnostics = selected["diagnostics"]
    assert selected["selected_support_nodes"].size == 0
    assert diagnostics["validation_trial_count"] == 2
    assert diagnostics["validation_greedy_steps"] == 0
    assert diagnostics["accepted_block_count"] == 0
    assert diagnostics["proxy_fallback_fill_count"] == 0
    assert diagnostics["validation_scores_unique_count"] == 1
    assert diagnostics["real_validation_degenerate"] is True


def test_real_occlusion_no_fallback_requires_complete_non_nan_metrics():
    def occlusion(occluded_nodes):
        count = len(occluded_nodes)
        return {
            "validation_macro_f1": 0.8 - 0.1 * count,
            "validation_cross_entropy": 0.2 + 0.1 * count,
            "margin": 0.7 - 0.1 * count,
            "teacher_kl": 0.0,
            "class_recall": 0.8 - 0.05 * count,
            "tree_tensor_l2_delta_when_occluded": 0.25 * count,
        }

    selected = select_support_nodes(
        _features(),
        np.ones(4, dtype=np.float32),
        0.5,
        SupportSelectorConfig(
            selector="real_occlusion_block_selector",
            occlusion_candidate_pool_size=2,
            allow_proxy_fill=False,
        ),
        occlusion_evaluator=occlusion,
    )

    diagnostics = selected["diagnostics"]
    assert diagnostics["occlusion_metric_complete"] is True
    assert diagnostics["occlusion_proxy_fallback_used"] is False
    assert diagnostics["occlusion_nonzero_delta_rate"] > 0.0
    assert diagnostics["occlusion_tree_delta_nonzero_rate"] > 0.0
    assert diagnostics["occlusion_delta_macro_f1_max"] > 0.0
    assert diagnostics["occlusion_tree_tensor_l2_delta_max"] > 0.0
    for row in selected["occlusion_block_scores"]:
        assert not np.isnan(row["delta_val_ce"])
        assert not np.isnan(row["delta_margin"])
        assert not np.isnan(row["delta_teacher_kl"])
        assert not np.isnan(row["tree_tensor_l2_delta_when_occluded"])

