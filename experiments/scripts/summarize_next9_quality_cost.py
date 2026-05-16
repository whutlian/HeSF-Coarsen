from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_next9_hgb_paper_final import _plot_scatter


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in (_as_float(value, None) for value in values) if value is not None]
    return None if not clean else float(mean(clean))


def _std(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in (_as_float(value, None) for value in values) if value is not None]
    return None if len(clean) <= 1 else float(pstdev(clean))


def _aggregate(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    metrics = [
        "DEE",
        "REEmax",
        "SIPE",
        "best_macro_f1",
        "refined_macro_f1@5",
        "total_wall_clock_sec",
        "peak_rss_gb",
        "coarse_train_sec",
        "coarse_graph_ratio",
    ]
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        out["run_count"] = len(group)
        for metric in metrics:
            values = [row.get(metric, "") for row in group]
            out[f"{metric}_mean"] = _fmt(_mean(values))
            out[f"{metric}_std"] = _fmt(_std(values) or 0.0)
        output.append(out)
    return output


def _dominance_rows(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in points:
        dominated_by = []
        f1 = _as_float(row.get("best_macro_f1_mean"), None)
        cost = _as_float(row.get("total_wall_clock_sec_mean"), None)
        memory = _as_float(row.get("peak_rss_gb_mean"), None)
        for other in points:
            if other is row:
                continue
            other_f1 = _as_float(other.get("best_macro_f1_mean"), None)
            other_cost = _as_float(other.get("total_wall_clock_sec_mean"), None)
            other_memory = _as_float(other.get("peak_rss_gb_mean"), None)
            if f1 is None or other_f1 is None:
                continue
            cost_ok = cost is None or other_cost is None or other_cost <= cost
            memory_ok = memory is None or other_memory is None or other_memory <= memory
            strict = other_f1 > f1 or (cost is not None and other_cost is not None and other_cost < cost)
            if other_f1 >= f1 and cost_ok and memory_ok and strict:
                dominated_by.append(str(other.get("method", "")))
        out = dict(row)
        out["dominated"] = "true" if dominated_by else "false"
        out["dominated_by"] = ";".join(dominated_by)
        output.append(out)
    return output


def summarize_next9_quality_cost(
    *,
    hgb_summary: str | Path,
    output: str | Path,
    ogbn_summary: str | Path | None = None,
    command_lines: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    hgb_summary = Path(hgb_summary)
    output = Path(output)
    rows = _read_csv(hgb_summary / "final_main_table_by_seed.csv")
    points = _aggregate(rows, ("method", "method_group"))
    by_dataset = _aggregate(rows, ("dataset", "method", "method_group"))
    dominance = _dominance_rows(points)

    write_csv(output / "quality_cost_points.csv", points)
    write_csv(output / "quality_cost_by_dataset.csv", by_dataset)
    write_csv(output / "quality_cost_dominance_table.csv", dominance)
    if command_lines:
        (output / "run_commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    figure_dir = output / "figures"
    _plot_scatter(points, "total_wall_clock_sec_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_total_wall_clock.png")
    _plot_scatter(points, "peak_rss_gb_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_peak_rss.png")
    _plot_scatter(points, "coarse_train_sec_mean", "best_macro_f1_mean", figure_dir / "best_macro_f1_vs_train_time.png")
    _plot_scatter(points, "DEE_mean", "best_macro_f1_mean", figure_dir / "dee_vs_best_macro_f1.png")
    _plot_scatter(points, "REEmax_mean", "best_macro_f1_mean", figure_dir / "ree_max_vs_best_macro_f1.png")
    _plot_scatter(points, "coarse_graph_ratio_mean", "best_macro_f1_mean", figure_dir / "coarse_ratio_vs_best_macro_f1.png")

    summary = [
        "# Next9 Quality-cost Pareto",
        "",
        "full tuned RGCN: best task when highest, no compression.",
        "flatten-sum/H6: task competitive, high operator distortion.",
        "P/S: compressed, low operator distortion, competitive task.",
        "",
        markdown_table(points, ["method", "best_macro_f1_mean", "total_wall_clock_sec_mean", "peak_rss_gb_mean", "DEE_mean", "coarse_graph_ratio_mean"]),
        "",
    ]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"points": points, "by_dataset": by_dataset, "dominance": dominance}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hgb-summary", required=True, type=Path)
    parser.add_argument("--ogbn-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--command-lines", nargs="*", default=[])
    args = parser.parse_args(argv)
    summarize_next9_quality_cost(
        hgb_summary=args.hgb_summary,
        ogbn_summary=args.ogbn_summary,
        output=args.output,
        command_lines=args.command_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
