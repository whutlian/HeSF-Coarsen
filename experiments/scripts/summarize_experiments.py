from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import (
    diagnostics_row,
    discover_run_dirs,
    disk_usage_bytes,
    markdown_table,
    read_json,
    write_csv,
)


def summarize_experiments(inputs: Iterable[str | Path], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    resource_rows: list[dict] = []
    quality_rows: list[dict] = []
    failure_rows: list[dict] = []

    for run_dir in discover_run_dirs(inputs):
        metadata_path = run_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        diagnostics_paths = sorted(run_dir.glob("level_*/diagnostics.json"))
        base = {
            "run_name": metadata.get("run_name", run_dir.name),
            "status": metadata.get("status", "success" if diagnostics_paths else "unknown"),
            "dataset": metadata.get("dataset", ""),
            "run_dir": str(run_dir),
            "failure_reason": metadata.get("failure_reason", ""),
        }
        if diagnostics_paths:
            for diagnostics_path in diagnostics_paths:
                row = {**base, **diagnostics_row(run_dir, diagnostics_path)}
                all_rows.append(row)
                resource_rows.append(
                    {
                        "run_name": base["run_name"],
                        "level": row.get("level", ""),
                        "disk_usage_bytes": disk_usage_bytes(run_dir),
                        "runtime_total": sum(
                            float(row.get(f"runtime_by_stage.{key}", 0) or 0)
                            for key in (
                                "sketch",
                                "candidates",
                                "scoring",
                                "matching_and_aggregation",
                                "spectral_diagnostics",
                            )
                        ),
                    }
                )
                quality_rows.append(
                    {
                        "run_name": base["run_name"],
                        "level": row.get("level", ""),
                        "compression_ratio": row.get("compression_ratio", ""),
                        "matched_pairs": row.get("matched_pairs", ""),
                        "singleton_ratio": row.get("singleton_ratio", ""),
                        "candidate_count_mean": row.get("candidate_count_mean", ""),
                        "spectral_sketch_dirichlet_energy_relative_error": row.get(
                            "spectral.sketch_dirichlet_energy_relative_error",
                            "",
                        ),
                        "spectral_relation_weighted_fused_energy_relative_error": row.get(
                            "spectral.relation_weighted_fused_energy_relative_error",
                            "",
                        ),
                        "spectral_relation_energy_relative_error_max": row.get(
                            "spectral.relation_energy_relative_error_max",
                            "",
                        ),
                        "spectral_chebheat_sketch_inner_product_relative_error": row.get(
                            "spectral.chebheat_sketch_inner_product_relative_error",
                            "",
                        ),
                    }
                )
        else:
            all_rows.append(base)
        if base["status"] == "failed":
            failure_rows.append(base)

    write_csv(output / "all_runs.csv", all_rows)
    write_csv(output / "resource_summary.csv", resource_rows)
    write_csv(output / "quality_summary.csv", quality_rows)
    write_csv(output / "failures.csv", failure_rows)
    report_rows = all_rows[:20]
    report = [
        "# Experiment Summary",
        "",
        f"Runs: {len(all_rows)}",
        f"Failures: {len(failure_rows)}",
        "",
        "## Completed Runs",
        "",
        markdown_table(report_rows, ["run_name", "status", "dataset", "level", "compression_ratio", "failure_reason"]),
        "",
        "## Failed Runs",
        "",
        markdown_table(failure_rows[:20], ["run_name", "status", "dataset", "failure_reason"]),
        "",
        "## Correctness Invariants",
        "",
        "Invariant fields are preserved in `all_runs.csv` when emitted by each runner.",
        "",
        "## Compression Ratios",
        "",
        "See `quality_summary.csv`.",
        "",
        "## Candidate Source Distribution",
        "",
        "Flattened diagnostics fields are preserved in `all_runs.csv`.",
        "",
        "## Runtime Breakdown",
        "",
        "Standard stage fields map to `runtime_by_stage.sketch`, `runtime_by_stage.candidates`, `runtime_by_stage.scoring`, `runtime_by_stage.matching_and_aggregation`, and derived totals.",
        "",
        "## Memory And Disk Footprint",
        "",
        "See `resource_summary.csv` for artifact disk usage and runtime totals.",
        "",
        "## Spectral Diagnostics",
        "",
        "Sketch-based diagnostics are optional and are recorded when present in run artifacts.",
        "",
        "## Bottleneck Analysis",
        "",
        "Compare resource and runtime summaries across presets before scaling full graph runs.",
        "",
        "## Recommended Next Engineering Fixes",
        "",
        "- Promote any failing run in `failures.csv` to a focused regression test.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize experiment run directories.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--inputs", nargs="+", type=Path, dest="input_flags", help="Run roots to summarize; accepted for plan command compatibility.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = [*args.inputs, *(args.input_flags or [])]
    if not inputs:
        parser.error("at least one input root is required via positional inputs or --inputs")
    summarize_experiments(inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
