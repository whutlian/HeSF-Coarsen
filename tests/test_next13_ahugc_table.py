from pathlib import Path

from experiments.scripts._common import write_csv
from experiments.scripts.next11_common import read_csv


def test_next13_ahugc_fair_table_separates_global_validation_and_oracle(tmp_path: Path):
    from experiments.scripts.summarize_next13_ahugc_fair_baseline import summarize_next13_ahugc_fair_baseline

    ahugc = tmp_path / "ahugc"
    paper = tmp_path / "paper"
    out = tmp_path / "out"
    write_csv(
        ahugc / "ahugc_style_best_overall.csv",
        [
            {
                "hash_bits": 20,
                "bucket_topk": 4,
                "assignment_source": "chebheat_sketch",
                "run_count": 15,
                "target_hit_rate": 1.0,
                "best_macro_f1_mean": 0.72364626901636,
                "best_macro_f1_std": 0.22,
                "projected_macro_f1_mean": 0.47,
                "projected_macro_f1_std": 0.15,
                "resource_logged_cumulative_dee_mean": 0.4214,
                "resource_logged_cumulative_dee_std": 0.04,
                "relation_energy_error_mean": 0.4912,
                "relation_energy_error_std": 0.02,
                "coarsening_wall_clock_sec_mean": 0.916,
                "coarsening_wall_clock_sec_std": 0.16,
                "peak_rss_gb_mean": 1.57,
                "peak_rss_gb_std": 0.09,
            }
        ],
    )
    write_csv(ahugc / "ahugc_style_best_config_by_dataset.csv", [{"dataset": "ACM", "hash_bits": 20, "bucket_topk": 4, "assignment_source": "chebheat_sketch", "best_macro_f1": 0.8}])
    write_csv(ahugc / "ahugc_style_sweep_by_config.csv", [{"hash_bits": 8, "bucket_topk": 4, "assignment_source": "raw_feature", "run_count": 15, "best_macro_f1_mean": 0.9, "best_macro_f1_std": 0.01}])
    write_csv(
        paper / "table3_external_baselines_with_ahugc_tuned_if_available.csv",
        [
            {"method": "HeSF-LVC-P", "run_count": 15, "best_mean": 0.74, "best_std": 0.1, "cumulative_dee_or_audited_dee_mean": 0.02},
            {"method": "random", "run_count": 15, "best_mean": 0.65, "best_std": 0.1, "cumulative_dee_or_audited_dee_mean": 0.49},
        ],
    )

    summarize_next13_ahugc_fair_baseline(next12_ahugc=ahugc, next12_paper=paper, output=out)

    main = read_csv(out / "external_baseline_main_table.csv")
    assert any(row["method"] == "AH-UGC-style tuned-global" for row in main)
    assert not any("oracle" in row["method"].lower() for row in main)
    assert "0.723646" in main[0]["best_macro_f1_mean_pm_std"]
    assert (out / "ahugc_oracle_appendix_only.csv").exists()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "not an official AH-UGC reproduction" in summary
    assert "HeSF-LVC is faster than AH-UGC-style" not in summary
