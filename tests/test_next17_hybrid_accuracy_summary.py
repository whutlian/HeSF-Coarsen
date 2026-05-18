from experiments.scripts.summarize_next17_hybrid_accuracy import aggregate_rows, summarize_block


def test_next17_summary_aggregates_fixed_config_rows(tmp_path):
    rows = [
        {
            "dataset": "ACM",
            "target_ratio": "0.024",
            "ratio_label": "2.4%",
            "variant": "A1_target_preserve",
            "model_name": "sehgnn_lite",
            "eval_mode": "full_target_inference",
            "macro_f1": "0.4",
            "accuracy": "0.5",
            "run_status": "success",
        },
        {
            "dataset": "ACM",
            "target_ratio": "0.024",
            "ratio_label": "2.4%",
            "variant": "A1_target_preserve",
            "model_name": "sehgnn_lite",
            "eval_mode": "full_target_inference",
            "macro_f1": "0.6",
            "accuracy": "0.7",
            "run_status": "success",
        },
    ]

    aggregate = aggregate_rows(rows, ["dataset", "target_ratio", "variant", "model_name", "eval_mode"])
    summarize_block(tmp_path, rows, title="Block")

    assert aggregate[0]["macro_f1_mean"] == 0.5
    assert aggregate[0]["accuracy_mean"] == 0.6
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "by_dataset.csv").exists()
    assert (tmp_path / "by_ratio.csv").exists()
    assert (tmp_path / "summary.md").exists()
