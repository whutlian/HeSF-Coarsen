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


def test_next15_summarizer_groups_hettree_metrics_by_dataset_ratio_method(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next15_hettree_compression import summarize_next15_hettree_compression

    raw = tmp_path / "raw"
    _write_csv(
        raw / "hettree_runs.csv",
        [
            {
                "dataset": "ACM",
                "method": "HeSF-LVC-P",
                "target_ratio": 0.012,
                "seed": 1,
                "skipped": False,
                "macro_f1": 0.4,
                "micro_f1": 0.5,
                "accuracy": 0.5,
                "actual_ratio": 0.013,
                "run_status": "success",
            },
            {
                "dataset": "ACM",
                "method": "HeSF-LVC-P",
                "target_ratio": 0.012,
                "seed": 2,
                "skipped": False,
                "macro_f1": 0.6,
                "micro_f1": 0.7,
                "accuracy": 0.7,
                "actual_ratio": 0.014,
                "run_status": "success",
            },
            {
                "dataset": "DBLP",
                "method": "HeSF-LVC-S",
                "target_ratio": 0.024,
                "seed": 1,
                "skipped": True,
                "macro_f1": "",
                "micro_f1": "",
                "accuracy": "",
                "actual_ratio": "",
                "run_status": "oom",
            },
        ],
    )

    output = tmp_path / "summary"
    summarize_next15_hettree_compression(input=raw, output=output)

    by_dataset = _read_csv(output / "hettree_by_dataset_ratio_method.csv")
    acm = next(row for row in by_dataset if row["dataset"] == "ACM")
    assert acm["run_count"] == "2"
    assert acm["failed_count"] == "0"
    assert float(acm["macro_f1_mean"]) == 0.5
    assert float(acm["accuracy_mean"]) == 0.6

    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "HETTREE" in summary
    assert "1.2%" in summary
