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


def test_next7_baseline_gap_report_joins_profiles_and_baselines(tmp_path):
    from experiments.scripts.summarize_next7_baseline_gap import summarize_next7_baseline_gap

    final_summary = tmp_path / "final_summary.csv"
    _write_csv(
        final_summary,
        [
            {
                "run_name": "h0_acm_seed1",
                "dataset": "ACM",
                "variant": "H0",
                "seed": 1,
                "target_ratio": 0.5,
                "cumulative_dee": 0.30,
                "cumulative_ree_max": 0.40,
                "cumulative_sipe": 0.70,
                "task_projected_macro_f1": 0.61,
                "task_refined_macro_f1@5": 0.64,
                "task_best_refined_macro_f1": 0.65,
                "task_refine_auc_macro_f1": 0.62,
                "task.train_time": 1.0,
                "task.refine_time": 0.2,
                "task.total_time": 1.2,
                "task_full_graph_rgcn_lite_default_macro_f1": 0.72,
                "task_full_graph_rgcn_lite_tuned_macro_f1": 0.75,
                "task_full_graph_han_small_macro_f1": 0.73,
                "task_full_graph_hgt_small_macro_f1": 0.74,
            },
            {
                "run_name": "p_acm_seed1",
                "dataset": "ACM",
                "variant": "H2",
                "experiment_block": "lambda_grid",
                "seed": 1,
                "target_ratio": 0.5,
                "lambda_spec": 0.25,
                "lambda_conv": 0.0,
                "lambda_rel": 0.0,
                "cumulative_dee": 0.12,
                "cumulative_ree_max": 0.22,
                "cumulative_sipe": 0.54,
                "task_projected_macro_f1": 0.66,
                "task_refined_macro_f1@5": 0.71,
                "task_best_refined_macro_f1": 0.76,
                "task_refine_auc_macro_f1": 0.70,
                "runtime_total_run": 9.0,
            },
            {
                "run_name": "flatten_acm_seed1",
                "dataset": "ACM",
                "variant": "H2-single-relation-sum",
                "seed": 1,
                "target_ratio": 0.5,
                "cumulative_dee": 0.20,
                "cumulative_ree_max": 0.28,
                "cumulative_sipe": 0.58,
                "task_projected_macro_f1": 0.63,
                "task_refined_macro_f1@5": 0.69,
                "task_best_refined_macro_f1": 0.70,
                "task_refine_auc_macro_f1": 0.67,
            },
        ],
    )
    baseline_summary = tmp_path / "baseline_summary.csv"
    _write_csv(
        baseline_summary,
        [
            {
                "run_name": "h2_acm_seed1",
                "dataset": "ACM",
                "variant": "H2",
                "seed": 1,
                "target_ratio": 0.5,
                "baseline": "random",
                "comparison_status": "included",
                "baseline_cumulative_dee": 0.50,
                "baseline_cumulative_ree_max": 0.55,
                "baseline_cumulative_sipe": 0.75,
                "baseline_projected_macro_f1": 0.50,
                "baseline_refined_macro_f1@5": 0.60,
                "baseline_task_best_refined_macro_f1": 0.60,
                "baseline_task_refine_auc_macro_f1": 0.55,
            },
            {
                "run_name": "h2_acm_seed1",
                "dataset": "ACM",
                "variant": "H2",
                "seed": 1,
                "target_ratio": 0.5,
                "baseline": "graphzoom_style",
                "comparison_status": "included",
                "baseline_cumulative_dee": 0.18,
                "baseline_cumulative_ree_max": 0.30,
                "baseline_cumulative_sipe": 0.62,
                "baseline_projected_macro_f1": 0.62,
                "baseline_refined_macro_f1@5": 0.68,
                "baseline_task_best_refined_macro_f1": 0.73,
                "baseline_task_refine_auc_macro_f1": 0.66,
            },
        ],
    )

    summarize_next7_baseline_gap(
        input_summaries=[final_summary],
        baseline_summaries=[baseline_summary],
        output=tmp_path / "next7",
        command_lines=["unit command"],
    )

    per_seed = _read_csv(tmp_path / "next7" / "per_seed_table.csv")
    p_row = next(row for row in per_seed if row["method"] == "HeSF-LVC-P")
    assert p_row["best_baseline_method"] == "GraphZoom-style"
    assert abs(float(p_row["delta_best_vs_best_baseline"]) - 0.03) < 1e-9
    assert abs(float(p_row["dee_reduction_vs_best_baseline"]) - (0.18 - 0.12) / 0.18) < 1e-9
    assert abs(float(p_row["delta_best_vs_full_tuned"]) - 0.01) < 1e-9

    aggregate = _read_csv(tmp_path / "next7" / "aggregate_main_table.csv")
    assert {row["method"] for row in aggregate} >= {"HeSF-LVC-P", "H0-mutual-best", "GraphZoom-style"}
    assert next(row for row in aggregate if row["method"] == "HeSF-LVC-P")["run_count"] == "1"

    full_refs = _read_csv(tmp_path / "next7" / "full_graph_reference_table.csv")
    assert full_refs[0]["dataset"] == "ACM"
    assert full_refs[0]["full_graph_rgcn_lite_tuned_macro_f1"] == "0.75"

    pareto = _read_csv(tmp_path / "next7" / "pareto_points.csv")
    assert next(row for row in pareto if row["method"] == "HeSF-LVC-P")["pareto_frontier"] == "true"

    report = (tmp_path / "next7" / "next7_hgb_baseline_gap_report.md").read_text(encoding="utf-8")
    assert "HeSF-LVC-P" in report
    assert "unit command" in (tmp_path / "next7" / "run_commands.txt").read_text(encoding="utf-8")
