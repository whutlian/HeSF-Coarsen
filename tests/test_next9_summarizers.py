import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_next9_paper_final_outputs_gap_tables_and_oracle_excludes_full_graph(tmp_path):
    from experiments.scripts.summarize_next9_hgb_paper_final import summarize_next9_hgb_paper_final

    next8 = tmp_path / "next8"
    _write_csv(
        next8 / "per_seed_table.csv",
        [
            {
                "method": "HeSF-LVC-P",
                "dataset": "ACM",
                "seed": 1,
                "target_ratio": 0.5,
                "coarse_graph_ratio": 0.5,
                "target_hit": "true",
                "dee": 0.02,
                "fse": 0.03,
                "ree_max": 0.04,
                "sipe": 0.50,
                "projected_macro_f1": 0.70,
                "refined_macro_f1@0": 0.71,
                "refined_macro_f1@1": 0.72,
                "refined_macro_f1@3": 0.73,
                "refined_macro_f1@5": 0.74,
                "best_macro_f1": 0.75,
                "refine_auc_macro_f1": 0.72,
                "runtime_sec": 10.0,
                "train_time_sec": 1.0,
                "refine_time_sec": 0.2,
                "peak_cpu_memory_gb": 2.0,
                "peak_gpu_memory_gb": 0.5,
            },
            {
                "method": "flatten-sum",
                "dataset": "ACM",
                "seed": 1,
                "target_ratio": 0.5,
                "coarse_graph_ratio": 0.5,
                "target_hit": "true",
                "dee": 0.20,
                "fse": 0.22,
                "ree_max": 0.25,
                "sipe": 0.60,
                "projected_macro_f1": 0.68,
                "refined_macro_f1@5": 0.735,
                "best_macro_f1": 0.755,
            },
            {
                "method": "H6-no-spec",
                "dataset": "ACM",
                "seed": 1,
                "target_ratio": 0.5,
                "coarse_graph_ratio": 0.5,
                "target_hit": "true",
                "dee": 0.30,
                "fse": 0.32,
                "ree_max": 0.35,
                "sipe": 0.70,
                "projected_macro_f1": 0.69,
                "refined_macro_f1@5": 0.73,
                "best_macro_f1": 0.74,
            },
            {
                "method": "full RGCN tuned",
                "dataset": "ACM",
                "seed": 1,
                "coarse_graph_ratio": 1.0,
                "best_macro_f1": 0.80,
                "refined_macro_f1@5": 0.80,
                "train_time_sec": 5.0,
            },
        ],
    )

    summarize_next9_hgb_paper_final(next8_summary_dir=next8, output=tmp_path / "next9")

    by_seed = _read_csv(tmp_path / "next9" / "final_main_table_by_seed.csv")
    p_row = next(row for row in by_seed if row["method"] == "HeSF-LVC-P")
    assert p_row["method_group"] == "ours"
    assert p_row["role"] == "default"
    assert p_row["DEE"] == "0.02"
    assert p_row["peak_rss_gb"] == "2"
    assert p_row["peak_vram_reserved_gb"] == "0.5"

    fixed = _read_csv(tmp_path / "next9" / "final_gap_vs_fixed_baselines.csv")
    flatten_task = next(
        row for row in fixed
        if row["method"] == "HeSF-LVC-P" and row["baseline"] == "flatten-sum" and row["metric"] == "best_macro_f1"
    )
    assert flatten_task["absolute_gap"] == "-0.005"
    flatten_dee = next(
        row for row in fixed
        if row["method"] == "HeSF-LVC-P" and row["baseline"] == "flatten-sum" and row["metric"] == "DEE"
    )
    assert flatten_dee["absolute_gap"] == "0.18"
    assert flatten_dee["relative_error_reduction"] == "0.9"

    oracle = _read_csv(tmp_path / "next9" / "final_gap_vs_oracle_coarse_baseline.csv")
    oracle_row = next(row for row in oracle if row["method"] == "HeSF-LVC-P" and row["metric"] == "best_macro_f1")
    assert oracle_row["oracle_coarse_baseline"] == "flatten-sum"
    assert oracle_row["oracle_coarse_baseline"] != "full RGCN tuned"

    assert (tmp_path / "next9" / "figures" / "dee_vs_best_macro_f1.png").exists()
    summary = (tmp_path / "next9" / "summary.md").read_text(encoding="utf-8")
    assert "full tuned RGCN remains stronger" in summary


def test_next9_quality_cost_handles_missing_spectral_for_full_graph_refs(tmp_path):
    from experiments.scripts.summarize_next9_quality_cost import summarize_next9_quality_cost

    paper = tmp_path / "paper"
    _write_csv(
        paper / "final_main_table_by_seed.csv",
        [
            {
                "method": "HeSF-LVC-P",
                "method_group": "ours",
                "dataset": "ACM",
                "seed": 1,
                "DEE": 0.02,
                "REEmax": 0.04,
                "best_macro_f1": 0.75,
                "total_wall_clock_sec": 10.0,
                "coarse_train_sec": 1.0,
                "peak_rss_gb": 2.0,
                "coarse_graph_ratio": 0.5,
            },
            {
                "method": "full RGCN tuned",
                "method_group": "full_graph_reference",
                "dataset": "ACM",
                "seed": 1,
                "best_macro_f1": 0.80,
                "total_wall_clock_sec": 20.0,
                "coarse_train_sec": 20.0,
                "coarse_graph_ratio": 1.0,
            },
        ],
    )

    summarize_next9_quality_cost(hgb_summary=paper, output=tmp_path / "qc")

    points = _read_csv(tmp_path / "qc" / "quality_cost_points.csv")
    assert next(row for row in points if row["method"] == "full RGCN tuned")["DEE_mean"] == ""
    dominance = _read_csv(tmp_path / "qc" / "quality_cost_dominance_table.csv")
    assert {row["method"] for row in dominance} == {"HeSF-LVC-P", "full RGCN tuned"}
    assert (tmp_path / "qc" / "figures" / "best_macro_f1_vs_total_wall_clock.png").exists()
