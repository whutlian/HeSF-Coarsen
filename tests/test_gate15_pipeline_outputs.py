import numpy as np

from hesf_coarsen.task_first.selection.config import Gate15Config, SupportSelectorConfig
from hesf_coarsen.task_first.selection.pipeline import run_supervised_support_selection_pipeline
from tests.gate15_test_utils import make_gate15_graph, split_masks


def test_pipeline_reports_required_metrics_without_static_pairwise_primary():
    graph = make_gate15_graph()
    train_mask, val_mask, test_mask = split_masks()
    cfg = Gate15Config(
        target_node_type=0,
        selector=SupportSelectorConfig(
            selector="teacher_topk",
            support_ratios=(0.5,),
            background_strategy="typed_background",
        ),
    )

    result = run_supervised_support_selection_pipeline(
        graph,
        np.asarray(graph.labels),
        train_mask,
        val_mask,
        test_mask,
        cfg,
        support_ratio=0.5,
        seed=3,
        task_epochs=1,
        task_hidden_dim=8,
        device="cpu",
    )

    assert result["method"] == "HeSF-SS-teacher-topk"
    assert result["primary_method_family"] == "supervised_support_selection"
    assert result["uses_static_pairwise_coarsening_as_primary"] is False
    for key in ("macro_f1", "micro_f1", "accuracy", "validation_macro_f1", "validation_accuracy"):
        assert key in result
    assert result["selector_uses_test_labels"] is False
    assert result["target_hit"] is True
