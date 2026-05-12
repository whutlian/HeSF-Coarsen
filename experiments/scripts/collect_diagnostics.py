from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import diagnostics_row, discover_run_dirs, write_csv


def collect_diagnostics(inputs: list[str | Path], output: str | Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in discover_run_dirs(inputs):
        for diagnostics in sorted(run_dir.glob("level_*/diagnostics.json")):
            rows.append(diagnostics_row(run_dir, diagnostics))
    write_csv(output, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect level diagnostics into one CSV.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path, help="Run roots to scan; accepted for plan command compatibility.")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/diagnostics.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = [*args.inputs, *(args.runs or [])]
    if not inputs:
        parser.error("at least one run root is required via positional inputs or --runs")
    collect_diagnostics(inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
