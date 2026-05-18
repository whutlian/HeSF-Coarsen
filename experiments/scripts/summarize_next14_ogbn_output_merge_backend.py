from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import as_float, read_csv, write_png


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "passed"}


def summarize_next14_ogbn_output_merge_backend(*, input: str | Path, output: str | Path) -> dict[str, Any]:
    input = Path(input)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    runs = read_csv(input / "aggregation_backend_runs.csv")
    timing = read_csv(input / "aggregation_backend_exclusive_timing.csv")
    rels = read_csv(input / "aggregation_backend_by_relation.csv")
    diagnostics = read_csv(input / "aggregation_output_merge_diagnostics.csv")
    checks = read_csv(input / "edge_weight_preservation_checks.csv")
    baseline = {
        (row.get("size", ""), row.get("method", "")): row
        for row in runs
        if row.get("backend") == "A0_current_sort_reducer" and row.get("run_status") == "available"
    }
    speedups: list[dict[str, Any]] = []
    for row in runs:
        ref = baseline.get((row.get("size", ""), row.get("method", "")), {})
        sec = as_float(row.get("aggregation_total_sec"), None)
        ref_sec = as_float(ref.get("aggregation_total_sec"), None)
        speedup = ref_sec / sec if ref_sec and sec else ""
        rss = as_float(row.get("peak_rss_gb"), None)
        ref_rss = as_float(ref.get("peak_rss_gb"), None)
        rss_delta = rss - ref_rss if rss is not None and ref_rss is not None else ""
        residual_frac = as_float(row.get("exclusive_timing_residual_frac"), as_float(row.get("exclusive_timing_residual_sec"), 0.0) or 0.0)
        speedups.append({**row, "speedup_vs_a0": speedup, "rss_delta_gb_vs_a0": rss_delta, "exclusive_timing_residual_frac": residual_frac, "recommended": False})

    full_by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in speedups:
        if row.get("size") == "full-local" and row.get("backend") != "A0_current_sort_reducer":
            full_by_backend.setdefault(str(row.get("backend", "")), []).append(row)
    recommended_backends = set()
    for backend, rows in full_by_backend.items():
        methods = {row.get("method") for row in rows}
        if not {"HeSF-LVC-P", "HeSF-LVC-S"}.issubset(methods):
            continue
        ok = True
        for row in rows:
            ok = ok and as_float(row.get("speedup_vs_a0"), 0.0) >= 1.25
            ok = ok and _truth(row.get("correctness_passed"))
            ok = ok and str(row.get("edge_weight_preservation_checks", "")).lower() == "passed"
            ok = ok and as_float(row.get("rss_delta_gb_vs_a0"), 999.0) <= 0.2
            ok = ok and as_float(row.get("exclusive_timing_residual_frac"), 999.0) <= 0.05
        if ok:
            recommended_backends.add(backend)
    for row in speedups:
        row["recommended"] = row.get("backend") in recommended_backends

    write_csv(output / "aggregation_backend_runs.csv", runs)
    write_csv(output / "aggregation_backend_speedup_summary.csv", speedups)
    write_csv(output / "aggregation_backend_exclusive_timing.csv", timing)
    write_csv(output / "aggregation_backend_by_relation.csv", rels)
    write_csv(output / "aggregation_output_merge_diagnostics.csv", diagnostics)
    write_csv(output / "edge_weight_preservation_checks.csv", checks)
    write_png(output / "figures" / "aggregation_backend_speedup.png", speedups, "backend", "speedup_vs_a0")
    write_png(output / "figures" / "output_write_bytes_per_sec.png", diagnostics, "backend", "output_write_bytes_per_sec")
    accepted = bool(recommended_backends)
    lines = [
        "# Next14 OGBN Output/Merge Backend",
        "",
        "OGBN-MAG is system/profiling evidence only. A0 remains default unless both full-local P and S meet adoption criteria.",
        "Result: " + (f"recommended backend(s): {', '.join(sorted(recommended_backends))}." if accepted else "no A6/A7/A8 backend meets adoption criteria; keep A0 default."),
        "",
        markdown_table(speedups, ["size", "method", "backend", "run_status", "aggregation_total_sec", "speedup_vs_a0", "rss_delta_gb_vs_a0", "exclusive_timing_residual_frac", "correctness_passed", "edge_weight_preservation_checks", "recommended"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"accepted": accepted, "rows": len(runs)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next14_ogbn_output_merge_backend(input=args.input, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
