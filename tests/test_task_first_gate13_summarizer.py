from pathlib import Path

from experiments.scripts.summarize_task_first_gate13 import _mean


def test_gate13_summarizer_mean_accepts_required_method_columns():
    rows = [
        {"method": "HeSF-TC-P-response", "ratio": "0.048", "macro_f1_mean": "0.5", "accuracy_mean": "0.6"},
        {"method": "HeSF-TC-P-response", "ratio": "0.096", "macro_f1_mean": "0.7", "accuracy_mean": "0.8"},
        {"method": "HeSF-TC-P-response", "ratio": "0.20", "macro_f1_mean": "0.1", "accuracy_mean": "0.2"},
    ]

    macro, acc, count = _mean(rows, "HeSF-TC-P-response")

    assert macro == 0.6
    assert acc == 0.7
    assert count == 2
