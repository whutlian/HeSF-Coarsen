from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _target_ratio_from_name(run_name: str) -> float | None:
    match = re.search(r"_r([0-9]+(?:p[0-9]+)?)", run_name)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def _quality_row(base: Mapping[str, Any], row: Mapping[str, Any], *, row_type: str) -> dict[str, Any]:
    return {
        "run_name": base["run_name"],
        "dataset": base.get("dataset", ""),
        "variant": base.get("variant", row.get("variant", "")),
        "row_type": row_type,
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
        "spectral_fused_sketch_energy_relative_error": row.get(
            "spectral.fused_sketch_energy_relative_error",
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
        "final_cumulative_ratio": row.get("final_cumulative_ratio", ""),
        "target_abs_error": row.get("target_abs_error", ""),
        "target_hit": row.get("target_hit", ""),
    }


def _final_cumulative_row(base: Mapping[str, Any], level_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(level_rows, key=lambda row: _as_int(row.get("level"), 0) or 0)
    first = ordered[0]
    last = ordered[-1]
    initial_nodes = _as_int(
        _first(
            first,
            (
                "target_control.original_nodes",
                "original_nodes",
            ),
        ),
        0,
    ) or 0
    final_nodes = _as_int(last.get("coarse_nodes"), 0) or 0
    final_ratio = float(final_nodes / max(initial_nodes, 1))
    run_name = str(base["run_name"])
    target_ratio = _as_float(
        _first(
            last,
            (
                "config.coarsening.target_ratio",
                "target_control.target_ratio",
                "target_ratio",
            ),
            "",
        ),
        None,
    )
    if target_ratio is None:
        target_ratio = _target_ratio_from_name(run_name)
    target_ratio = float(target_ratio if target_ratio is not None else final_ratio)
    target_abs_error = abs(final_ratio - target_ratio)
    hit_tolerance = _as_float(
        _first(last, ("config.coarsening.target_hit_tolerance", "target_hit_tolerance"), ""),
        0.05,
    )
    target_hit = bool(target_abs_error <= float(hit_tolerance or 0.05))

    def level_error(row: Mapping[str, Any]) -> float:
        coarse_nodes = _as_int(row.get("coarse_nodes"), final_nodes) or final_nodes
        return abs(float(coarse_nodes / max(initial_nodes, 1)) - target_ratio)

    best = min(ordered, key=level_error)
    max_levels = _as_int(
        _first(last, ("config.coarsening.max_levels", "target_control.max_levels"), ""),
        len(ordered),
    ) or len(ordered)
    stopped_by = "target_hit" if target_hit else "max_levels"
    if len(ordered) < max_levels and not target_hit:
        input_nodes = _as_int(last.get("original_nodes"), final_nodes) or final_nodes
        stopped_by = "no_decrease" if final_nodes >= input_nodes else "no_more_levels"

    final = {**base, **last}
    final.update(
        {
            "row_type": "final",
            "level": "final",
            "level_row_count": int(len(ordered)),
            "initial_nodes": int(initial_nodes),
            "final_nodes": int(final_nodes),
            "final_cumulative_ratio": final_ratio,
            "target_ratio": target_ratio,
            "target_abs_error": target_abs_error,
            "target_hit": "true" if target_hit else "false",
            "best_level": str(best.get("level", "")),
            "stopped_by": stopped_by,
            "cumulative_dee": last.get("spectral.sketch_dirichlet_energy_relative_error", ""),
            "cumulative_fwe_weighted": last.get(
                "spectral.relation_weighted_fused_energy_relative_error",
                "",
            ),
            "cumulative_fse_unweighted": last.get("spectral.fused_sketch_energy_relative_error", ""),
            "cumulative_ree_max": last.get("spectral.relation_energy_relative_error_max", ""),
            "cumulative_sipe": last.get(
                "spectral.chebheat_sketch_inner_product_relative_error",
                "",
            ),
        }
    )
    return final


def summarize_experiments(inputs: Iterable[str | Path], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    resource_rows: list[dict] = []
    quality_rows: list[dict] = []
    final_rows: list[dict] = []
    failure_rows: list[dict] = []

    run_dirs = discover_run_dirs(inputs)
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        diagnostics_paths = sorted(run_dir.glob("level_*/diagnostics.json"))
        base = {
            "run_name": metadata.get("run_name", run_dir.name),
            "status": metadata.get("status", "success" if diagnostics_paths else "unknown"),
            "dataset": metadata.get("dataset", ""),
            "variant": metadata.get("variant", ""),
            "run_dir": str(run_dir),
            "failure_reason": metadata.get("failure_reason", ""),
        }
        if diagnostics_paths:
            level_rows: list[dict] = []
            for diagnostics_path in diagnostics_paths:
                row = {**base, **diagnostics_row(run_dir, diagnostics_path), "row_type": "level"}
                level_rows.append(row)
                all_rows.append(row)
                resource_rows.append(
                    {
                        "run_name": base["run_name"],
                        "row_type": "level",
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
                quality_rows.append(_quality_row(base, row, row_type="level"))
            final_row = _final_cumulative_row(base, level_rows)
            final_rows.append(final_row)
            all_rows.append(final_row)
            quality_rows.append(_quality_row(base, final_row, row_type="final"))
        else:
            all_rows.append({**base, "row_type": "run"})
        if base["status"] == "failed":
            failure_rows.append(base)

    for final_row in final_rows:
        final_row["run_count_unique"] = int(len(final_rows))

    write_csv(output / "all_runs.csv", all_rows)
    write_csv(output / "final_summary.csv", final_rows)
    write_csv(output / "resource_summary.csv", resource_rows)
    write_csv(output / "quality_summary.csv", quality_rows)
    write_csv(output / "failures.csv", failure_rows)
    report_rows = all_rows[:20]
    report = [
        "# Experiment Summary",
        "",
        f"Unique runs: {len(final_rows)}",
        f"Level rows: {sum(int(row.get('level_row_count', 0) or 0) for row in final_rows)}",
        f"Rows: {len(all_rows)}",
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
