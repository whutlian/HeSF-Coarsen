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


def _minimal_inputs(tmp_path: Path) -> dict[str, Path]:
    next13_paper = tmp_path / "next13_paper"
    next13_ahugc = tmp_path / "next13_ahugc"
    next13_metapath = tmp_path / "next13_metapath"
    next13_structure = tmp_path / "next13_structure"
    next13_ogbn = tmp_path / "next13_ogbn"

    _write_csv(
        next13_paper / "table1_hgb_main_operator_task.csv",
        [
            {"method": "HeSF-LVC-P", "paper_final_dee": 0.02, "relation_energy_error": 0.04, "best_macro_f1": 0.74},
            {"method": "HeSF-LVC-S", "paper_final_dee": 0.01, "relation_energy_error": 0.03, "best_macro_f1": 0.73},
            {"method": "flatten-sum", "paper_final_dee": 0.18, "relation_energy_error": 0.19, "best_macro_f1": 0.74},
            {"method": "H6-no-spec", "paper_final_dee": 0.17, "relation_energy_error": 0.14, "best_macro_f1": 0.75},
        ],
    )
    _write_csv(
        next13_paper / "table2_flatten_h6_rebuttal.csv",
        [
            {"method": "flatten-sum", "paper_final_dee": 0.18, "relation_energy_error": 0.19, "best_macro_f1": 0.74},
            {"method": "H6-no-spec", "paper_final_dee": 0.17, "relation_energy_error": 0.14, "best_macro_f1": 0.75},
            {"method": "HeSF-LVC-P", "paper_final_dee": 0.02, "relation_energy_error": 0.04, "best_macro_f1": 0.74},
        ],
    )
    _write_csv(
        next13_paper / "table3_external_baselines_fair.csv",
        [
            {
                "method": "AH-UGC-style tuned-global",
                "result_class": "global_fixed",
                "appendix_only": "False",
                "hash_bits": 20,
                "bucket_topk": 4,
                "assignment_source": "chebheat_sketch",
                "best_macro_f1_mean": 0.7236,
                "best_macro_f1_std": 0.2,
                "target_hit_rate": 1.0,
                "resource_logged_cumulative_dee": 0.4214,
                "relation_energy_error": 0.4912,
                "coarsening_wall_clock_sec": 0.916,
            },
            {"method": "GraphZoom-style", "result_class": "global_fixed", "appendix_only": "False", "best_macro_f1_mean": 0.70, "resource_logged_cumulative_dee": 0.48, "coarsening_wall_clock_sec": 0.87},
        ],
    )
    _write_csv(
        next13_paper / "table4_quality_cost.csv",
        [
            {"method": "AH-UGC-style tuned-global", "best_macro_f1": "", "coarsening_wall_clock_sec": 0.916},
            {"method": "HeSF-LVC-P", "best_macro_f1": "", "coarsening_wall_clock_sec": 12.2},
        ],
    )
    _write_csv(next13_paper / "table5_claim_boundary.csv", [{"claim": "task-F1 dominance", "status": "unsupported"}])
    _write_csv(
        next13_ahugc / "external_baseline_main_table.csv",
        [
            {
                "method": "AH-UGC-style tuned-global",
                "target_hit_rate": 1.0,
                "best_macro_f1_mean_pm_std": "0.723646 +/- 0.220903",
                "projected_macro_f1_mean_pm_std": "0.471013 +/- 0.156848",
                "resource_logged_cumulative_dee_mean_pm_std": "0.421379 +/- 0.042292",
                "relation_energy_error_mean_pm_std": "0.491152 +/- 0.018718",
                "coarsening_wall_clock_sec_mean_pm_std": "0.916431 +/- 0.157854",
                "peak_rss_gb_mean_pm_std": "1.569472 +/- 0.098688",
                "notes": "protocol-matched baseline; not official AH-UGC",
            },
            {
                "method": "HeSF-LVC-P",
                "best_macro_f1_mean_pm_std": "0.745466 +/- 0.204960",
                "resource_logged_cumulative_dee_mean_pm_std": "0.201616 +/- 0.047202",
                "coarsening_wall_clock_sec_mean_pm_std": "12.283775 +/- 3.593203",
            },
        ],
    )
    _write_csv(next13_metapath / "metapath_mass_by_method.csv", [{"method": "HeSF-LVC-P", "metapath_mass_relative_error_mean": 0.68}])
    _write_csv(next13_metapath / "metapath_mass_by_dataset.csv", [{"dataset": "ACM", "method": "HeSF-LVC-P", "metapath_mass_relative_error_mean": 0.68}])
    _write_csv(next13_metapath / "metapath_mass_gap_vs_flatten_h6.csv", [{"method": "HeSF-LVC-P", "delta_vs_flatten_sum": 0.04}])
    _write_csv(next13_structure / "structure_task_by_method.csv", [{"task": "lowpass", "method": "HeSF-LVC-P", "signal_mse_mean": 0.1}])
    _write_csv(next13_structure / "structure_task_gap_vs_flatten_h6.csv", [{"task": "lowpass", "method": "HeSF-LVC-P", "delta_vs_flatten_sum_mean": 0.0}])
    _write_csv(next13_ogbn / "aggregation_backend_speedup_summary.csv", [{"backend": "A0_current_sort_reducer", "recommended": "False"}])
    return {
        "next13_paper": next13_paper,
        "next13_ahugc": next13_ahugc,
        "next13_metapath": next13_metapath,
        "next13_structure": next13_structure,
        "next13_ogbn": next13_ogbn,
    }


def test_next14_paper_tables_are_claim_safe_and_nan_free(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next14_paper_tables import summarize_next14_paper_tables

    inputs = _minimal_inputs(tmp_path)
    output = tmp_path / "out"
    summarize_next14_paper_tables(output=output, **inputs)

    for csv_path in output.glob("*_final.csv"):
        rows = _read_csv(csv_path)
        assert "DEE" not in (rows[0].keys() if rows else [])

    table3 = _read_csv(output / "table3_external_typedhash_baselines_final.csv")
    assert any(row["method"] == "TypedHash-ChebHeat tuned-global" for row in table3)
    assert "official AH-UGC reproduction" not in (output / "summary.md").read_text(encoding="utf-8")

    table4 = _read_csv(output / "table4_quality_cost_final.csv")
    assert all(row["result_class"] == "global_fixed" for row in table4)
    assert not any(row["method"].startswith("TypedHash oracle") for row in table4)
    for row in table4:
        assert row["method"]
        assert row["best_macro_f1_mean"] not in {"", "nan", "NaN"}
        assert row["coarsening_wall_clock_sec"] not in {"", "nan", "NaN"}

    assert (output / "appendix_metapath_mass_diagnostics.csv").exists()
    assert (output / "appendix_structure_critical_diagnostics.csv").exists()
    table1_text = (output / "table1_hgb_main_operator_task_final.csv").read_text(encoding="utf-8")
    assert "metapath_mass" not in table1_text


def test_next14_paper_tables_fail_clearly_when_sources_missing(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next14_paper_tables import summarize_next14_paper_tables

    missing = tmp_path / "missing"
    try:
        summarize_next14_paper_tables(
            next13_paper=missing,
            next13_ahugc=missing,
            next13_metapath=missing,
            next13_structure=missing,
            next13_ogbn=missing,
            output=tmp_path / "out",
        )
    except FileNotFoundError as exc:
        assert "missing required input sources" in str(exc)
    else:
        raise AssertionError("expected missing input sources to fail")
