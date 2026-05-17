from pathlib import Path

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_next11_guard_appendix import summarize_next11_guard_appendix


def test_guard_appendix_surfaces_threshold_and_target_failures(tmp_path: Path):
    guard = tmp_path / "guard"
    out = tmp_path / "out"
    write_csv(
        guard / "summary" / "guard_ablation_main_table.csv",
        [
            {"variant": "P_baseline", "dataset": "ACM", "seed": 1, "target_hit": "true", "DEE": 0.2, "best_macro_f1": 0.8, "onehop_high_delta_selected_share": 0.2},
            {"variant": "P_spectral_guard", "dataset": "ACM", "seed": 1, "target_hit": "true", "DEE": 0.1, "best_macro_f1": 0.797, "onehop_high_delta_selected_share": 0.1},
            {"variant": "S_baseline", "dataset": "ACM", "seed": 1, "target_hit": "true", "DEE": 0.2, "best_macro_f1": 0.8, "onehop_high_delta_selected_share": 0.2},
            {"variant": "S_spectral_guard", "dataset": "ACM", "seed": 1, "target_hit": "false", "DEE": 0.1, "best_macro_f1": 0.79, "onehop_high_delta_selected_share": 0.1},
            {"variant": "P_source_aware_auto", "dataset": "ACM", "seed": 1, "target_hit": "true", "DEE": 0.2, "best_macro_f1": 0.8, "onehop_high_delta_selected_share": 0.05},
        ],
    )

    summarize_next11_guard_appendix(guard=guard, output=out)

    summary = (out / "summary.md").read_text(encoding="utf-8").lower()
    assert "0.005" in summary
    assert "not the main method" in summary
    assert "guard is the main method" not in summary
    failures = (out / "guard_acceptance_summary.csv").read_text(encoding="utf-8")
    assert "target_hit_failures" in failures
    assert (out / "figures/guard_delta_dee.png").exists()

