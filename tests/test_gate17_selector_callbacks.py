import numpy as np

from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from hesf_coarsen.task_first.selection.selector import select_support_nodes


def _features():
    return {
        "support_nodes": np.array([10, 11, 12, 13, 14], dtype=np.int64),
        "support_node_types": np.array([1, 1, 2, 2, 3], dtype=np.int32),
        "component_matrices": {
            "relation_profile": np.array(
                [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            ),
            "anchor_distribution": np.array(
                [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            ),
            "class_footprint": np.array(
                [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            ),
            "degree_profile": np.array([[4, 0], [3, 0], [1, 0], [1, 0], [2, 0]], dtype=np.float32),
        },
    }


def test_real_validation_records_multistep_trial_rows_and_callback_usage():
    calls = []

    def fake_validation(selected_nodes):
        selected = {int(node) for node in selected_nodes}
        calls.append(tuple(sorted(selected)))
        return len(selected & {12, 13, 14}) + 0.01 * len(selected)

    selected = select_support_nodes(
        _features(),
        np.array([10.0, 9.0, 1.0, 1.0, 2.0], dtype=np.float32),
        0.8,
        SupportSelectorConfig(
            selector="real_validation_block_greedy",
            candidate_pool_size=2,
            max_validation_greedy_steps=3,
            min_gain=-1.0,
        ),
        validation_evaluator=fake_validation,
    )

    diagnostics = selected["diagnostics"]
    assert len(calls) > 1
    assert diagnostics["selector_family"] == "real_validation_selector"
    assert diagnostics["selector_uses_true_validation_feedback"] is True
    assert diagnostics["validation_trial_count"] > diagnostics["validation_candidate_pool_size"]
    assert diagnostics["validation_greedy_steps"] >= 2
    assert "proxy_fallback_fill_count" in diagnostics
    assert selected["validation_greedy_trials"]
    assert {"candidate_rank", "trimmed_to_budget"} <= set(selected["validation_greedy_trials"][0])


def test_proxy_true_validation_name_does_not_claim_real_feedback():
    selected = select_support_nodes(
        _features(),
        np.ones(5, dtype=np.float32),
        0.4,
        SupportSelectorConfig(selector="true_validation_block_greedy"),
    )

    assert selected["diagnostics"]["selector_uses_true_validation_feedback"] is False
    assert selected["diagnostics"]["selector_family"] == "validation_sensitivity_selector"


def test_real_occlusion_records_base_occluded_metrics_and_zero_delta_degeneracy():
    calls = []

    def constant_occlusion(occluded_nodes):
        calls.append(tuple(int(node) for node in occluded_nodes))
        return {"validation_macro_f1": 0.5}

    selected = select_support_nodes(
        _features(),
        np.ones(5, dtype=np.float32),
        0.6,
        SupportSelectorConfig(selector="real_occlusion_block_selector", occlusion_candidate_pool_size=3),
        occlusion_evaluator=constant_occlusion,
    )

    diagnostics = selected["diagnostics"]
    assert len(calls) >= 4
    assert diagnostics["occlusion_trial_count"] == 3
    assert diagnostics["occlusion_degenerate"] is True
    assert diagnostics["occlusion_proxy_fallback_used"] is True
    assert diagnostics["selector_feedback_source"] == "occlusion_proxy_fallback"
    score = selected["occlusion_block_scores"][0]
    assert {
        "base_validation_macro_f1",
        "occluded_validation_macro_f1",
        "base_val_ce",
        "occluded_val_ce",
        "tree_tensor_l2_delta_when_occluded",
    } <= set(score)
    assert np.isnan(score["base_val_ce"])
