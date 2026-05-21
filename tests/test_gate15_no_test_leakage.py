import numpy as np

from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_pipeline_marks_test_labels_as_metrics_only():
    graph = make_gate15_graph()
    train_mask, val_mask, test_mask = split_masks()

    result = run_supervised_support_selection_pipeline(
        graph,
        np.asarray(graph.labels),
        train_mask,
        val_mask,
        test_mask,
        Gate15Config(
            target_node_type=0,
            selector=SupportSelectorConfig(selector="hybrid_teacher_response", support_ratios=(0.5,)),
        ),
        support_ratio=0.5,
        seed=11,
        task_epochs=1,
        task_hidden_dim=8,
        device="cpu",
    )

    assert result["selector_uses_test_labels"] is False
    assert result["teacher_uses_test_labels_for_training"] is False
    assert result["test_label_usage"] == "metrics_only"
    assert result["validation_selection_uses"] == "validation_macro_f1"
