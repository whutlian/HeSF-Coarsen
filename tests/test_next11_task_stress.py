from pathlib import Path

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_next11_hgb_task_stress import summarize_next11_hgb_task_stress


def test_task_stress_summary_outputs_required_tables_and_conclusion(tmp_path: Path):
    inp = tmp_path / "stress"
    out = tmp_path / "summary"
    write_csv(
        inp / "task_stress_runs.csv",
        [
            {"stress_block": "low_label", "stress_name": "label_0.05", "method": "HeSF-LVC-P", "dataset": "ACM", "seed": 1, "run_status": "available", "best_macro_f1": 0.71, "refined_macro_f1@5": 0.70, "projected_macro_f1": 0.6, "label_coverage_train": 1.0, "num_classes_present_train": 3},
            {"stress_block": "low_label", "stress_name": "label_0.05", "method": "flatten-sum", "dataset": "ACM", "seed": 1, "run_status": "available", "best_macro_f1": 0.70, "refined_macro_f1@5": 0.69, "projected_macro_f1": 0.59, "label_coverage_train": 1.0, "num_classes_present_train": 3},
            {"stress_block": "cross_model", "stress_name": "hgt_lite", "method": "H6-no-spec", "dataset": "ACM", "seed": 1, "run_status": "unsupported", "reason": "model cannot consume graph"},
        ],
    )

    summarize_next11_hgb_task_stress(input=inp, output=out)

    for name in [
        "task_stress_by_method_dataset.csv",
        "low_label_summary.csv",
        "early_refine_summary.csv",
        "cross_model_summary.csv",
        "relation_mask_summary.csv",
        "delta_vs_flatten_sum_by_stress.csv",
        "delta_vs_h6_by_stress.csv",
        "win_rate_by_stress.csv",
        "summary.md",
    ]:
        assert (out / name).exists(), name
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Task recovery remains competitive" in summary or "task-superiority" in summary

