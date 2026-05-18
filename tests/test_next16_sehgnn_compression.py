from pathlib import Path

from experiments.scripts.summarize_next16_sehgnn_compression import summarize_next16_sehgnn_compression


def test_next16_summarizer_aggregates_sehgnn_runs(tmp_path):
    raw = tmp_path / "sehgnn_runs.csv"
    raw.write_text(
        "\n".join(
            [
                "dataset,method,target_ratio,seed,run_status,skipped,macro_f1,micro_f1,accuracy,actual_ratio,train_time,feature_time,total_time",
                "ACM,HeSF-LVC-P,0.012,1,success,false,0.40,0.50,0.50,0.013,1.0,0.2,1.2",
                "ACM,HeSF-LVC-P,0.012,2,success,false,0.60,0.70,0.70,0.012,2.0,0.4,2.4",
            ]
        ),
        encoding="utf-8",
    )

    result = summarize_next16_sehgnn_compression(input=tmp_path, output=tmp_path / "summary")

    assert result["raw_rows"] == 2
    by_dataset = (tmp_path / "summary" / "sehgnn_by_dataset_ratio_method.csv").read_text(encoding="utf-8")
    assert "0.5" in by_dataset
    assert (tmp_path / "summary" / "summary.md").exists()
