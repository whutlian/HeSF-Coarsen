from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_typedhash_summary_renames_ahugc_and_keeps_appendix_rows_out_of_main(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next14_typedhash_baseline import summarize_next14_typedhash_baseline

    next12 = tmp_path / "next12"
    next13 = tmp_path / "next13"
    _write_csv(
        next12 / "ahugc_style_best_overall.csv",
        [
            {
                "hash_bits": 20,
                "bucket_topk": 4,
                "assignment_source": "chebheat_sketch",
                "run_count": 15,
                "best_macro_f1_mean": 0.7236,
                "best_macro_f1_std": 0.2,
                "projected_macro_f1_mean": 0.47,
                "resource_logged_cumulative_dee_mean": 0.421,
                "resource_logged_cumulative_dee_std": 0.04,
                "relation_energy_error_mean": 0.491,
                "coarsening_wall_clock_sec_mean": 0.916,
                "peak_rss_gb_mean": 1.56,
                "target_hit_rate": 1.0,
            }
        ],
    )
    _write_csv(next12 / "ahugc_style_best_config_by_dataset.csv", [{"dataset": "ACM", "assignment_source": "raw_feature", "best_macro_f1_mean": 0.9}])
    _write_csv(next12 / "ahugc_style_sweep_by_config.csv", [{"assignment_source": "chebheat_sketch", "best_macro_f1_mean": 0.95}])
    _write_csv(
        next13 / "table3_external_baselines_fair.csv",
        [
            {"method": "HeSF-LVC-P", "best_macro_f1_mean": 0.74, "resource_logged_cumulative_dee": 0.20, "coarsening_wall_clock_sec": 12.0},
            {"method": "random", "best_macro_f1_mean": 0.68, "resource_logged_cumulative_dee": 0.49, "coarsening_wall_clock_sec": 0.9},
        ],
    )

    output = tmp_path / "out"
    summarize_next14_typedhash_baseline(next12_ahugc=next12, next13_paper=next13, output=output)
    main = _read_csv(output / "typedhash_main_table.csv")
    assert any(row["method"] == "TypedHash-ChebHeat tuned-global" and row["assignment_source"] == "chebheat_sketch" for row in main)
    assert all(row["result_class"] == "global_fixed" for row in main)
    assert all(row["appendix_only"] == "False" for row in main)
    assert not any("oracle" in row["method"].lower() for row in main)
    assert not any("validation" in row["method"].lower() for row in main)
    assert all(row["best_macro_f1_mean"] not in {"", "nan", "NaN"} for row in main if row["method"].startswith("TypedHash"))
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "official AH-UGC reproduction" not in summary
    assert "not official AH-UGC" in summary


def test_typedhash_summary_writes_gap_and_quality_cost_tables(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next14_typedhash_baseline import summarize_next14_typedhash_baseline

    next12 = tmp_path / "next12"
    next13 = tmp_path / "next13"
    _write_csv(next12 / "ahugc_style_best_overall.csv", [{"hash_bits": 20, "bucket_topk": 4, "assignment_source": "chebheat_sketch", "run_count": 3, "best_macro_f1_mean": 0.72, "resource_logged_cumulative_dee_mean": 0.42, "coarsening_wall_clock_sec_mean": 1.0}])
    _write_csv(next12 / "ahugc_style_best_config_by_dataset.csv", [])
    _write_csv(next12 / "ahugc_style_sweep_by_config.csv", [])
    _write_csv(next13 / "table3_external_baselines_fair.csv", [{"method": "HeSF-LVC-P", "best_macro_f1_mean": 0.75, "resource_logged_cumulative_dee": 0.2, "coarsening_wall_clock_sec": 12.0}])
    output = tmp_path / "out"
    summarize_next14_typedhash_baseline(next12_ahugc=next12, next13_paper=next13, output=output)
    assert (output / "typedhash_gap_vs_hesf.csv").exists()
    assert (output / "typedhash_quality_cost.csv").exists()
