from experiments.scripts.gate14_task_first_common import build_ratio_matched_rows


def test_ratio_matching_separates_requested_and_realized_tables():
    hesf = [{"dataset": "ACM", "seed": 1, "method": "HeSF", "realized_support_ratio": 0.20, "macro_f1": 0.7}]
    baselines = [
        {"dataset": "ACM", "seed": 1, "method": "H6", "requested_support_ratio": 0.048, "realized_support_ratio": 0.25, "macro_f1": 0.6},
    ]

    rows = build_ratio_matched_rows(hesf, baselines, tolerance=0.025, non_comparable_gap=0.05)

    assert rows[0]["ratio_gap"] == 0.05
    assert rows[0]["comparison_status"] == "nearest_flagged"


def test_ratio_matching_marks_large_gap_non_comparable():
    hesf = [{"dataset": "ACM", "seed": 1, "method": "HeSF", "realized_support_ratio": 0.048, "macro_f1": 0.7}]
    baselines = [{"dataset": "ACM", "seed": 1, "method": "H6", "realized_support_ratio": 0.25, "macro_f1": 0.6}]

    rows = build_ratio_matched_rows(hesf, baselines, tolerance=0.025, non_comparable_gap=0.05)

    assert rows[0]["comparison_status"] == "non_comparable"
    assert rows[0]["delta_macro_f1"] == ""
