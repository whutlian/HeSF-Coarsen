import numpy as np

from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from hesf_coarsen.task_first.selection.selector import select_support_nodes
from tests.test_gate17_true_validation_selector import _features


def test_gate17_feedback_selectors_report_no_test_label_usage():
    features = _features()
    importance = np.ones(4, dtype=np.float32)

    validation = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(selector="real_validation_block_greedy", max_validation_greedy_steps=1),
        validation_evaluator=lambda nodes: float(len(nodes)),
    )
    occlusion = select_support_nodes(
        features,
        importance,
        0.5,
        SupportSelectorConfig(selector="real_occlusion_block_selector", occlusion_candidate_pool_size=1),
        occlusion_evaluator=lambda nodes: 1.0 - 0.1 * len(nodes),
    )

    assert validation["diagnostics"]["selector_uses_test_labels"] is False
    assert validation["diagnostics"]["selection_split_source"] == "train_val_only"
    assert occlusion["diagnostics"]["selector_uses_test_labels"] is False
    assert occlusion["diagnostics"]["selection_split_source"] == "train_val_only"
