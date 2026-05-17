from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, read_csv, write_png


def summarize_next11_ogbn_aggregation_optimizer(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    runs = read_csv(input / "aggregation_optimizer_runs.csv")
    stages = read_csv(input / "aggregation_optimizer_stage_breakdown.csv")
    by_relation = read_csv(input / "aggregation_optimizer_by_relation.csv")
    checks = read_csv(input / "aggregation_optimizer_correctness_checks.csv")
    baseline = {
        (row.get("size", ""), row.get("method", "")): row
        for row in runs
        if row.get("aggregation_variant") == "A0_current_sort_reducer" and row.get("run_status") == "available"
    }
    speedups = []
    for row in runs:
        ref = baseline.get((row.get("size", ""), row.get("method", "")))
        ref_sec = as_float(ref.get("aggregation_total_sec") if ref else None, None)
        sec = as_float(row.get("aggregation_total_sec"), None)
        speedup = ref_sec / sec if ref_sec and sec else ""
        rss_ref = as_float(ref.get("peak_rss_gb") if ref else None, None)
        rss = as_float(row.get("peak_rss_gb"), None)
        rss_increase = (rss - rss_ref) / rss_ref if rss_ref and rss is not None else ""
        correctness = str(row.get("correctness_passed", "")).lower() == "true"
        speedups.append(
            {
                "size": row.get("size", ""),
                "method": row.get("method", ""),
                "aggregation_variant": row.get("aggregation_variant", ""),
                "run_status": row.get("run_status", ""),
                "aggregation_total_sec": row.get("aggregation_total_sec", ""),
                "speedup_vs_a0": speedup,
                "peak_rss_increase_vs_a0": rss_increase,
                "correctness_passed": row.get("correctness_passed", ""),
                "recommended": bool(str(row.get("size", "")) == "full-local" and as_float(speedup, 0.0) >= 1.25 and correctness and as_float(rss_increase, 1.0) <= 0.2),
                "reason": row.get("reason", ""),
            }
        )
    write_csv(output / "aggregation_optimizer_runs.csv", runs)
    write_csv(output / "aggregation_optimizer_stage_breakdown.csv", stages)
    write_csv(output / "aggregation_optimizer_by_relation.csv", by_relation)
    write_csv(output / "aggregation_optimizer_correctness_checks.csv", checks)
    write_csv(output / "aggregation_optimizer_speedup_summary.csv", speedups)
    write_png(output / "figures" / "aggregation_stage_stacked_bar.png", stages, "aggregation_total_sec", "sort_sec")
    write_png(output / "figures" / "full_local_speedup.png", speedups, "speedup_vs_a0", "peak_rss_increase_vs_a0")
    write_png(output / "figures" / "per_relation_edges_per_sec.png", by_relation, "input_edges", "edges_per_sec")
    recommended = [row for row in speedups if row.get("recommended")]
    lines = [
        "# Next11 OGBN Aggregation Optimizer",
        "",
        "OGBN-MAG remains system/protocol evidence only; no task-quality claim is made.",
        (
            f"Recommended variants: {', '.join(str(row['aggregation_variant']) for row in recommended)}"
            if recommended
            else "No optimizer variant satisfies full-local speedup >= 1.25x with correctness and RSS constraints; keep A0 as default and report the dominant stage."
        ),
        "",
        markdown_table(speedups, ["size", "method", "aggregation_variant", "run_status", "speedup_vs_a0", "peak_rss_increase_vs_a0", "recommended", "reason"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"speedups": speedups}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next11_ogbn_aggregation_optimizer(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

