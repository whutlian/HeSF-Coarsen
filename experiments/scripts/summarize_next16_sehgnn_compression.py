from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv


METRICS = [
    "macro_f1",
    "micro_f1",
    "accuracy",
    "actual_ratio",
    "train_time",
    "feature_time",
    "total_time",
    "peak_vram_allocated_mb",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def _is_failed(row: Mapping[str, Any]) -> bool:
    status = str(row.get("run_status", row.get("status", ""))).strip().lower()
    skipped = str(row.get("skipped", "")).strip().lower()
    return status not in {"success", "available", ""} or skipped in {"true", "1", "yes"}


def _ratio_label(value: Any) -> str:
    ratio = _as_float(value)
    if ratio is None:
        return str(value)
    return f"{ratio * 100:.1f}%"


def _mean_std(values: Sequence[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    arr = [float(value) for value in values]
    mean = float(sum(arr) / len(arr))
    if len(arr) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in arr) / (len(arr) - 1)
    return mean, float(var**0.5)


def _aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        row: dict[str, Any] = {key: value for key, value in zip(keys, group_key)}
        row["ratio_label"] = _ratio_label(row.get("target_ratio", row.get("compression_ratio", "")))
        row["run_count"] = len(group_rows)
        row["failed_count"] = sum(1 for item in group_rows if _is_failed(item))
        for metric in METRICS:
            values = [
                value
                for value in (_as_float(item.get(metric, "")) for item in group_rows if not _is_failed(item))
                if value is not None
            ]
            mean, std = _mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def summarize_next16_sehgnn_compression(*, input: str | Path, output: str | Path) -> dict[str, int]:
    input = Path(input)
    output = Path(output)
    raw_path = input / "sehgnn_runs.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing required input: {raw_path}")
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(raw_path)
    write_csv(output / "sehgnn_runs.csv", rows)

    by_dataset = _aggregate(rows, ["dataset", "method", "target_ratio"])
    by_method = _aggregate(rows, ["method", "target_ratio"])
    write_csv(output / "sehgnn_by_dataset_ratio_method.csv", by_dataset)
    write_csv(output / "sehgnn_by_ratio_method.csv", by_method)

    display_rows = [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "ratio": row.get("ratio_label", ""),
            "macro_f1": row.get("macro_f1_mean", ""),
            "micro_f1": row.get("micro_f1_mean", ""),
            "accuracy": row.get("accuracy_mean", ""),
            "actual_ratio": row.get("actual_ratio_mean", ""),
            "n": row.get("run_count", ""),
            "failed": row.get("failed_count", ""),
        }
        for row in by_dataset[:24]
    ]
    summary = [
        "# Next16 SeHGNN Compression Evaluation",
        "",
        "This table reports a local SeHGNN-style downstream evaluation on existing coarse HGB graphs.",
        "The evaluator follows the official ICT-GIMLab/SeHGNN architecture pattern: per-metapath projection and transformer semantic fusion.",
        "",
        "Ratios are interpreted as whole-graph node compression targets. Metrics: macro-F1, micro-F1, accuracy.",
        "",
        markdown_table(display_rows, ["dataset", "method", "ratio", "macro_f1", "micro_f1", "accuracy", "actual_ratio", "n", "failed"]),
        "",
        "Raw and aggregate CSV files are in this directory.",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "raw_rows": len(rows),
        "by_dataset_rows": len(by_dataset),
        "by_method_rows": len(by_method),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next16_sehgnn_compression(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
