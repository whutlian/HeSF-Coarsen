from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, read_csv, write_png


def summarize_next13_ogbn_aggregation_backend(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    runs = read_csv(input / "aggregation_backend_runs.csv")
    timing = read_csv(input / "aggregation_backend_exclusive_timing.csv")
    rels = read_csv(input / "aggregation_backend_by_relation.csv")
    checks = read_csv(input / "aggregation_backend_correctness_checks.csv")
    baseline = {
        (row.get("size", ""), row.get("method", "")): row
        for row in runs
        if row.get("backend") == "A0_current_sort_reducer" and row.get("run_status") == "available"
    }
    speedups = []
    for row in runs:
        ref = baseline.get((row.get("size", ""), row.get("method", "")))
        sec = as_float(row.get("aggregation_total_sec"), None)
        ref_sec = as_float(ref.get("aggregation_total_sec") if ref else None, None)
        speedup = ref_sec / sec if ref_sec and sec else ""
        rss = as_float(row.get("peak_rss_gb"), None)
        ref_rss = as_float(ref.get("peak_rss_gb") if ref else None, None)
        rss_delta = rss - ref_rss if rss is not None and ref_rss is not None else ""
        recommended = bool(
            row.get("backend") != "A0_current_sort_reducer"
            and row.get("size") == "full-local"
            and as_float(speedup, 0.0) >= 1.25
            and str(row.get("correctness_passed", "")).lower() == "true"
            and str(row.get("edge_weight_preservation_checks", "")).lower() == "passed"
            and as_float(rss_delta, 999.0) <= 0.2
        )
        speedups.append({**row, "speedup_vs_a0": speedup, "rss_delta_gb_vs_a0": rss_delta, "recommended": recommended})
    write_csv(output / "aggregation_backend_runs.csv", runs)
    write_csv(output / "aggregation_backend_speedup_summary.csv", speedups)
    write_csv(output / "aggregation_backend_exclusive_timing.csv", timing)
    write_csv(output / "aggregation_backend_by_relation.csv", rels)
    write_csv(output / "aggregation_backend_correctness_checks.csv", checks)
    write_png(output / "figures" / "aggregation_backend_speedup.png", speedups, "size", "speedup_vs_a0")
    write_png(output / "figures" / "exclusive_timing_residual.png", timing, "backend", "exclusive_timing_residual_sec")
    accepted = any(bool(row.get("recommended")) for row in speedups)
    lines = [
        "# Next13 OGBN Aggregation Backend",
        "",
        "OGBN-MAG is system/profiling evidence only; no task-quality claim is made.",
        "A new backend is adopted only if full-local speedup >= 1.25x with correctness and RSS checks.",
        "Result: " + ("new backend meets adoption rule." if accepted else "no new backend meets adoption rule; keep A0 default."),
        "",
        markdown_table(speedups, ["size", "method", "backend", "run_status", "aggregation_total_sec", "speedup_vs_a0", "rss_delta_gb_vs_a0", "correctness_passed", "edge_weight_preservation_checks", "recommended"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"accepted": accepted}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next13_ogbn_aggregation_backend(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
