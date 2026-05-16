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


def test_p3_source_aware_summary_compares_policy_metrics(tmp_path):
    from experiments.scripts.summarize_p3_source_aware import summarize_p3_source_aware

    baseline = tmp_path / "baseline"
    source_aware = tmp_path / "source_aware"
    _write_csv(
        baseline / "run_final_summary.csv",
        [
            {
                "dataset": "ACM",
                "variant": "H2",
                "lambda_spec": 0.25,
                "seed": 1,
                "cumulative_dee": 0.02,
                "cumulative_ree_max": 0.10,
                "cumulative_sipe": 0.54,
                "task_projected_macro_f1": 0.70,
                "task_refined_macro_f1@5": 0.73,
                "task_best_refined_macro_f1": 0.74,
                "target_hit": "true",
                "candidate_source_counts.onehop": 200,
                "selected_merges_by_source.onehop": 50,
            }
        ],
    )
    _write_csv(
        source_aware / "run_final_summary.csv",
        [
            {
                "dataset": "ACM",
                "variant": "H2",
                "lambda_spec": 0.25,
                "seed": 1,
                "cumulative_dee": 0.018,
                "cumulative_ree_max": 0.09,
                "cumulative_sipe": 0.53,
                "task_projected_macro_f1": 0.71,
                "task_refined_macro_f1@5": 0.735,
                "task_best_refined_macro_f1": 0.745,
                "target_hit": "true",
                "candidate_source_counts.onehop": 80,
                "selected_merges_by_source.onehop": 5,
                "source_policy_filter.onehop_rejected_by_spec": 60,
            }
        ],
    )
    _write_csv(
        baseline / "candidate_source_pareto.csv",
        [
            {
                "dataset": "ACM",
                "variant": "H2",
                "source": "onehop",
                "candidate_fraction": 0.8,
                "selected_fraction": 0.5,
                "avg_delta_spec": 16.0,
            }
        ],
    )
    _write_csv(
        source_aware / "candidate_source_pareto.csv",
        [
            {
                "dataset": "ACM",
                "variant": "H2",
                "source": "onehop",
                "candidate_fraction": 0.2,
                "selected_fraction": 0.05,
                "avg_delta_spec": 15.0,
            }
        ],
    )

    summarize_p3_source_aware(
        baseline_summary_dir=baseline,
        source_aware_summary_dir=source_aware,
        output=tmp_path / "p3",
    )

    comparison = _read_csv(tmp_path / "p3" / "source_policy_comparison.csv")
    source_row = next(row for row in comparison if row["policy"] == "source-aware")
    assert source_row["method"] == "HeSF-LVC-P"
    assert source_row["onehop_retained_mean"] == "80"
    assert source_row["onehop_rejected_by_spec_mean"] == "60"

    distribution = _read_csv(tmp_path / "p3" / "source_distribution_by_policy.csv")
    assert next(row for row in distribution if row["policy"] == "source-aware")[
        "selected_fraction_mean"
    ] == "0.05"
