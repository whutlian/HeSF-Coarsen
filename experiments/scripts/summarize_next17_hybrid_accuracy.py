from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv


METRICS = ["macro_f1", "micro_f1", "accuracy", "primary_task_metric"]


def _as_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _failed(row: Mapping[str, Any]) -> bool:
    status = str(row.get("run_status", "")).lower()
    return status not in {"", "success", "available"}


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        item: dict[str, Any] = {key: value for key, value in zip(keys, group_key)}
        item["run_count"] = len(group_rows)
        item["failed_count"] = sum(1 for row in group_rows if _failed(row))
        for metric in METRICS:
            values = [
                value
                for value in (_as_float(row.get(metric, "")) for row in group_rows if not _failed(row))
                if value is not None
            ]
            item[f"{metric}_mean"] = float(mean(values)) if values else ""
            item[f"{metric}_std"] = _std(values) if values else ""
        out.append(item)
    return out


def summarize_block(output: str | Path, rows: Sequence[Mapping[str, Any]], *, title: str) -> dict[str, int]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    write_csv(output / "runs.csv", rows)
    by_dataset = aggregate_rows(rows, ["dataset", "variant", "model_name", "eval_mode"])
    by_ratio = aggregate_rows(rows, ["target_ratio", "variant", "model_name", "eval_mode"])
    by_dataset_ratio = aggregate_rows(
        rows,
        ["dataset", "target_ratio", "variant", "model_name", "eval_mode"],
    )
    write_csv(output / "by_dataset.csv", by_dataset)
    write_csv(output / "by_ratio.csv", by_ratio)
    write_csv(output / "by_dataset_ratio.csv", by_dataset_ratio)
    display = by_dataset_ratio[:40]
    summary = [
        f"# {title}",
        "",
        markdown_table(
            display,
            [
                "dataset",
                "target_ratio",
                "variant",
                "model_name",
                "eval_mode",
                "macro_f1_mean",
                "accuracy_mean",
                "run_count",
                "failed_count",
            ],
        ),
        "",
        "Rows are fixed global configs aggregated over seeds.",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"rows": len(rows), "by_dataset": len(by_dataset), "by_ratio": len(by_ratio)}


def summarize_next17_hybrid_accuracy(*, input: str | Path, output: str | Path | None = None) -> dict[str, int]:
    input = Path(input)
    output = Path(output) if output is not None else input
    rows = read_csv(input / "runs.csv")
    return summarize_block(output, rows, title="Next17 Hybrid Accuracy")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summarize_next17_hybrid_accuracy(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
