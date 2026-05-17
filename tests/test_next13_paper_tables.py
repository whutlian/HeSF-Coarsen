from pathlib import Path

from experiments.scripts._common import write_csv
from experiments.scripts.next11_common import read_csv


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    next12_paper = root / "paper"
    next12_ahugc = root / "ahugc"
    output = root / "out"
    write_csv(
        next12_paper / "table1_hgb_main_operator_task.csv",
        [
            {"method": "HeSF-LVC-P", "paper_final_dee": 0.02, "relation_energy_error": 0.04, "best_macro_f1": 0.74},
            {"method": "flatten-sum", "paper_final_dee": 0.18, "relation_energy_error": 0.19, "best_macro_f1": 0.73},
        ],
    )
    write_csv(
        next12_paper / "table3_external_baselines_with_ahugc_tuned_if_available.csv",
        [
            {"method": "AH-UGC-style", "best_mean": 0.68, "best_std": 0.2, "run_count": 15},
            {"method": "random", "best_mean": 0.65, "best_std": 0.1, "run_count": 15},
        ],
    )
    write_csv(
        next12_ahugc / "ahugc_style_best_overall.csv",
        [
            {
                "hash_bits": 20,
                "bucket_topk": 4,
                "assignment_source": "chebheat_sketch",
                "run_count": 15,
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
                "target_hit_rate": 1.0,
            }
        ],
    )
    write_csv(
        next12_ahugc / "ahugc_style_best_config_by_dataset.csv",
        [
            {"dataset": "ACM", "hash_bits": 20, "bucket_topk": 4, "assignment_source": "chebheat_sketch", "best_macro_f1": 0.8},
            {"dataset": "DBLP", "hash_bits": 12, "bucket_topk": 8, "assignment_source": "raw_feature", "best_macro_f1": 0.6},
        ],
    )
    write_csv(
        next12_ahugc / "ahugc_style_sweep_by_config.csv",
        [
            {"hash_bits": 8, "bucket_topk": 4, "assignment_source": "raw_feature", "run_count": 15, "best_macro_f1_mean": 0.9, "best_macro_f1_std": 0.01},
            {"hash_bits": 20, "bucket_topk": 4, "assignment_source": "chebheat_sketch", "run_count": 15, "best_macro_f1_mean": 0.72364626901636, "best_macro_f1_std": 0.22},
        ],
    )
    return next12_paper, next12_ahugc, output


def test_next13_paper_tables_use_tuned_global_ahugc_and_no_bare_dee(tmp_path: Path):
    from experiments.scripts.summarize_next13_paper_tables import summarize_next13_paper_tables

    next12_paper, next12_ahugc, output = _write_inputs(tmp_path)
    summarize_next13_paper_tables(next12_paper=next12_paper, next12_ahugc=next12_ahugc, output=output)

    table3 = read_csv(output / "table3_external_baselines_fair.csv")
    tuned = [row for row in table3 if row["method"] == "AH-UGC-style tuned-global"]
    assert len(tuned) == 1
    assert tuned[0]["result_class"] == "global_fixed"
    assert float(tuned[0]["best_macro_f1_mean"]) == 0.72364626901636
    assert float(tuned[0]["target_hit_rate"]) == 1.0
    assert tuned[0]["hash_bits"] == "20"
    assert tuned[0]["assignment_source"] == "chebheat_sketch"

    classes = {row["result_class"] for row in table3 if row["method"].startswith("AH-UGC-style")}
    assert {"global_fixed", "validation_selected_by_dataset", "oracle_appendix_only"}.issubset(classes)

    for path in output.glob("table*.csv"):
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "DEE" not in header
        assert all(not column.endswith("_DEE") for column in header)


def test_next13_paper_tables_reject_single_seed_mean_rows(tmp_path: Path):
    from experiments.scripts.summarize_next13_paper_tables import summarize_next13_paper_tables

    next12_paper, next12_ahugc, output = _write_inputs(tmp_path)
    rows = read_csv(next12_ahugc / "ahugc_style_best_overall.csv")
    rows[0]["run_count"] = 1
    write_csv(next12_ahugc / "ahugc_style_best_overall.csv", rows)

    try:
        summarize_next13_paper_tables(next12_paper=next12_paper, next12_ahugc=next12_ahugc, output=output)
    except ValueError as exc:
        assert "run_count" in str(exc)
    else:
        raise AssertionError("single-seed AH-UGC-style mean row should be rejected")
