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


def test_p1_flatten_sum_challenge_summarizes_checkpoints_models_and_low_label(tmp_path):
    from experiments.scripts.summarize_p1_flatten_sum_challenge import (
        summarize_p1_flatten_sum_challenge,
    )

    final_gap = tmp_path / "final_gap"
    _write_csv(
        final_gap / "per_seed_table.csv",
        [
            {
                "method": "HeSF-LVC-P",
                "dataset": "DBLP",
                "seed": 1,
                "dee": 0.1,
                "fse": 0.2,
                "ree_max": 0.3,
                "sipe": 0.4,
                "projected_macro_f1": 0.70,
                "refined_macro_f1@0": 0.71,
                "refined_macro_f1@1": 0.72,
                "refined_macro_f1@3": 0.73,
                "refined_macro_f1@5": 0.74,
                "best_macro_f1": 0.75,
            },
            {
                "method": "flatten-sum",
                "dataset": "DBLP",
                "seed": 1,
                "dee": 0.2,
                "fse": 0.3,
                "ree_max": 0.4,
                "sipe": 0.5,
                "projected_macro_f1": 0.65,
                "refined_macro_f1@0": 0.66,
                "refined_macro_f1@1": 0.67,
                "refined_macro_f1@3": 0.72,
                "refined_macro_f1@5": 0.735,
                "best_macro_f1": 0.735,
            },
        ],
    )
    cross_model = tmp_path / "cross_model.csv"
    _write_csv(
        cross_model,
        [
            {
                "run_name": "lambda_grid_DBLP_H2_r0p5_L4_ls0p25_lc0p0_lr0p0_seed1",
                "dataset": "DBLP",
                "variant": "H2",
                "coarse_model": "han_small",
                "best_refined_macro_f1": 0.76,
                "refined_original_macro_f1@5": 0.75,
            },
            {
                "run_name": "flatten_DBLP_seed1",
                "dataset": "DBLP",
                "variant": "H2-single-relation-sum",
                "coarse_model": "rgcn_lite",
                "best_refined_macro_f1": 0.735,
                "refined_original_macro_f1@5": 0.735,
            },
        ],
    )
    low_label = tmp_path / "low_label.csv"
    _write_csv(
        low_label,
        [
            {
                "run_name": "lambda_grid_DBLP_H2_r0p5_L4_ls0p25_lc0p0_lr0p0_seed1",
                "dataset": "DBLP",
                "variant": "H2",
                "train_fraction": 0.1,
                "best_refined_macro_f1": 0.61,
                "refined_original_macro_f1@5": 0.60,
            }
        ],
    )

    summarize_p1_flatten_sum_challenge(
        final_gap_dir=final_gap,
        output=tmp_path / "p1",
        cross_model_inputs=[cross_model],
        low_label_inputs=[low_label],
    )

    checkpoint = _read_csv(tmp_path / "p1" / "checkpoint_comparison.csv")
    assert next(row for row in checkpoint if row["method"] == "flatten-sum")[
        "refined_macro_f1@5_mean"
    ] == "0.735"

    failure = _read_csv(tmp_path / "p1" / "flatten_sum_failure_by_dataset.csv")
    assert failure[0]["dataset"] == "DBLP"
    assert failure[0]["delta_best_vs_HeSF-LVC-P"] == "-0.015"

    models = _read_csv(tmp_path / "p1" / "cross_model_transfer.csv")
    assert next(row for row in models if row["coarse_model"] == "han_small")["method"] == "HeSF-LVC-P"

    low_rows = _read_csv(tmp_path / "p1" / "low_label_transfer.csv")
    assert low_rows[0]["train_fraction"] == "0.1"

    report = (tmp_path / "p1" / "p1_flatten_sum_challenge_report.md").read_text(encoding="utf-8")
    assert "operator-preserving coarsening" in report
