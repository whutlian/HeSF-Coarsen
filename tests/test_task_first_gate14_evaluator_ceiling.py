from experiments.scripts.gate14_task_first_common import (
    compute_recovery_vs_ceiling,
    evaluator_status_rows,
    select_validation_best_rows,
)


def test_recovery_requires_full_graph_lite_ceiling_rows():
    rows = [{"dataset": "ACM", "seed": 1, "method": "HeSF", "macro_f1": 0.5, "accuracy": 0.6}]

    recovery = compute_recovery_vs_ceiling(rows, [])

    assert recovery[0]["recovery_status"] == "missing_full_graph_lite_ceiling"


def test_recovery_uses_compressed_metric_divided_by_ceiling():
    rows = [{"dataset": "ACM", "seed": 1, "method": "HeSF", "macro_f1": 0.5, "accuracy": 0.6}]
    ceiling = [{"dataset": "ACM", "seed": 1, "macro_f1": 1.0, "accuracy": 0.75}]

    recovery = compute_recovery_vs_ceiling(rows, ceiling)

    assert recovery[0]["recovery_vs_full_graph_lite_macro"] == 0.5
    assert recovery[0]["recovery_vs_full_graph_lite_accuracy"] == 0.8


def test_validation_selection_uses_validation_metric_not_test_metric():
    rows = [
        {"dataset": "ACM", "seed": 1, "method": "A", "validation_macro_f1": 0.1, "validation_accuracy": 0.9, "macro_f1": 0.9},
        {"dataset": "ACM", "seed": 1, "method": "B", "validation_macro_f1": 0.2, "validation_accuracy": 0.1, "macro_f1": 0.3},
    ]

    selected = select_validation_best_rows(rows)

    assert selected[0]["method"] == "B"


def test_official_evaluator_status_defaults_to_diagnostic_lite_only():
    rows = evaluator_status_rows()

    assert rows["official_hettree_status"] == "not_integrated"
    assert rows["diagnostic_scope"] == "diagnostic_lite_only"
