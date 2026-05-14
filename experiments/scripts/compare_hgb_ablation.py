from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv


def _float_values(rows: Iterable[Mapping[str, str]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(metric, "")
        if raw in {"", None}:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def compare_hgb_ablation(
    *,
    summary: str | Path,
    output: str | Path,
    group_by: list[str],
    metrics: list[str],
) -> None:
    with Path(summary).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("row_type", "final") not in {"", "final"}:
            continue
        key = tuple(str(row.get(column, "")) for column in group_by)
        groups[key].append(row)

    output_rows: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items()):
        out: dict[str, object] = {column: value for column, value in zip(group_by, key)}
        out["run_count"] = int(len(group_rows))
        for metric in metrics:
            values = _float_values(group_rows, metric)
            if not values:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_median"] = ""
                out[f"{metric}_min"] = ""
                out[f"{metric}_max"] = ""
                continue
            out[f"{metric}_mean"] = float(sum(values) / len(values))
            out[f"{metric}_median"] = float(median(values))
            out[f"{metric}_min"] = float(min(values))
            out[f"{metric}_max"] = float(max(values))
        output_rows.append(out)

    write_csv(output, output_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare grouped HGB ablation summaries.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", nargs="+", default=["dataset", "variant"])
    parser.add_argument("--metrics", nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compare_hgb_ablation(
        summary=args.summary,
        output=args.output,
        group_by=list(args.group_by),
        metrics=list(args.metrics),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
