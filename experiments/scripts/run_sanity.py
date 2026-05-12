from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_command_metadata, write_config_snapshot
from hesf_coarsen.coarsen.multilevel import run_multilevel_coarsening
from hesf_coarsen.config import DEFAULT_CONFIG
from hesf_coarsen.eval.invariants import validate_level_invariants
from hesf_coarsen.io.edge_list import generate_synthetic_graph, save_graph


def _sanity_config(output: Path, max_levels: int) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config["coarsening"] = dict(
        DEFAULT_CONFIG["coarsening"],
        target_ratio=0.45,
        max_levels=max_levels,
        per_level_ratio=0.7,
    )
    config["sketch"] = dict(DEFAULT_CONFIG["sketch"], dim=8, order=2, dtype="float32")
    config["candidates"] = dict(
        DEFAULT_CONFIG["candidates"],
        total_budget_K=8,
        twohop_budget_K2=4,
        per_middle_pair_cap=16,
        bucket_pair_cap=16,
        simhash_bits=4,
    )
    config["output"] = {"dir": str(output)}
    return config


def run_sanity(output: str | Path, python: str = "python") -> int:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = generate_synthetic_graph(num_users=16, num_items=10, num_tags=6, seed=20260512)
    save_graph(graph, output / "input_graph")

    rows: list[dict[str, object]] = []
    core_keys = [
        "schema_type_violations",
        "invalid_assignment_count",
        "relation_schema_violations",
        "diagnostics_missing_count",
    ]
    for max_levels in (1, 2):
        run_name = f"sanity_{max_levels}_level"
        run_dir = output / run_name
        config = _sanity_config(run_dir, max_levels=max_levels)
        write_config_snapshot(run_dir / "config.yaml", config)
        write_command_metadata(
            run_dir,
            run_name=run_name,
            command=[python, "experiments/scripts/run_sanity.py", "--output", str(output)],
            dataset="synthetic_tiny",
            max_levels=max_levels,
            status="running",
        )
        try:
            current_input = graph
            results = run_multilevel_coarsening(graph, config)
            for result in results:
                level_dir = run_dir / f"level_{result.level}"
                invariants = validate_level_invariants(
                    original=current_input,
                    coarse=result.graph,
                    assignment=result.assignment,
                    diagnostics_path=level_dir / "diagnostics.json",
                )
                row = {
                    "run_name": run_name,
                    "status": "success",
                    "level": result.level,
                    "input_nodes": current_input.num_nodes,
                    "coarse_nodes": result.graph.num_nodes,
                    **invariants,
                }
                rows.append(row)
                current_input = result.graph
            write_command_metadata(
                run_dir,
                run_name=run_name,
                command=[python, "experiments/scripts/run_sanity.py", "--output", str(output)],
                dataset="synthetic_tiny",
                max_levels=max_levels,
                status="success",
                levels=len(results),
            )
        except Exception as exc:
            rows.append({"run_name": run_name, "status": "failed", "failure_reason": str(exc)})
            write_command_metadata(
                run_dir,
                run_name=run_name,
                dataset="synthetic_tiny",
                max_levels=max_levels,
                status="failed",
                failure_reason=str(exc),
            )

    summary_path = output / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Sanity Report", "", markdown_table(rows, fieldnames[: min(len(fieldnames), 8)])]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 1 if any(int(row.get(key, 0) or 0) != 0 for row in rows for key in core_keys) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tiny synthetic sanity experiments.")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/sanity"))
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_sanity(output=args.output, python=args.python)


if __name__ == "__main__":
    raise SystemExit(main())
