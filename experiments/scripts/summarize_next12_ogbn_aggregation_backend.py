from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, read_csv, write_png


def summarize_next12_ogbn_aggregation_backend(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    runs = read_csv(input / "aggregation_backend_runs.csv")
    stages = read_csv(input / "aggregation_backend_stage_breakdown.csv")
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
        ref_sec = as_float(ref.get("aggregation_total_sec") if ref else None, None)
        sec = as_float(row.get("aggregation_total_sec"), None)
        speedup = ref_sec / sec if ref_sec and sec else ""
        rss_ref = as_float(ref.get("peak_rss_gb") if ref else None, None)
        rss = as_float(row.get("peak_rss_gb"), None)
        rss_delta_gb = (rss - rss_ref) if rss_ref is not None and rss is not None else ""
        ok = str(row.get("correctness_passed", "")).lower() == "true"
        weights = str(row.get("edge_weight_preservation_checks", "")).lower() == "passed"
        recommended = bool(row.get("backend") == "A3_packed_key_sort" and row.get("size") == "full-local" and as_float(speedup, 0.0) >= 1.25 and ok and weights and as_float(rss_delta_gb, 999.0) <= 0.2)
        speedups.append({**row, "speedup_vs_a0": speedup, "rss_delta_gb_vs_a0": rss_delta_gb, "recommended": recommended})
    write_csv(output / "aggregation_backend_runs.csv", runs)
    write_csv(output / "aggregation_backend_stage_breakdown.csv", stages)
    write_csv(output / "aggregation_backend_by_relation.csv", rels)
    write_csv(output / "aggregation_backend_correctness_checks.csv", checks)
    write_csv(output / "aggregation_backend_speedup_summary.csv", speedups)
    write_png(output / "figures" / "a3_speedup_vs_a0.png", speedups, "size", "speedup_vs_a0")
    write_png(output / "figures" / "a3_rss_delta_vs_a0.png", speedups, "size", "rss_delta_gb_vs_a0")
    accepted = any(row.get("recommended") for row in speedups)
    lines = [
        "# Next12 OGBN Aggregation Backend",
        "",
        "OGBN-MAG remains system/profiling evidence only; no task-quality claim is made.",
        (
            "A3 packed-key sort meets the full-local adoption rule and can replace A0 for the accepted method rows."
            if accepted
            else "A3 packed-key sort does not meet the full-local >=1.25x speedup plus correctness/RSS rule; keep A0 default."
        ),
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
    summarize_next12_ogbn_aggregation_backend(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
