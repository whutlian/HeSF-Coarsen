from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.gate13_task_first_common import (
    DEFAULT_SEEDS,
    aggregate_rows,
    run_full_graph_ceiling_row,
    run_parallel,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate13 full graph lite ceiling.")
    parser.add_argument("--datasets", nargs="+", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=["hettree_lite"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def _worker(args: argparse.Namespace, dataset: str, seed: int, model: str) -> dict:
    try:
        return run_full_graph_ceiling_row(args, dataset, int(seed), model)
    except Exception as exc:
        return {"dataset": dataset, "seed": int(seed), "model": model, "status": "failed", "error": repr(exc)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [(dataset, seed, model) for dataset in args.datasets for seed in args.seeds for model in args.models]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "full_graph_lite_ceiling_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("model")), int(row.get("seed", 0))))
    write_csv(args.output / "full_graph_lite_ceiling_runs.csv", rows)
    by_dataset = aggregate_rows(rows, ["dataset", "model"], ["macro_f1", "micro_f1", "accuracy"])
    write_csv(args.output / "full_graph_lite_ceiling_by_dataset.csv", by_dataset)
    summary = aggregate_rows(rows, ["model"], ["macro_f1", "micro_f1", "accuracy"])
    write_csv(args.output / "full_graph_lite_ceiling_summary.csv", summary)
    write_summary_md(
        args.output / "full_graph_lite_ceiling_summary.md",
        "Gate13 Full Graph Lite Ceiling",
        by_dataset,
        ["dataset", "model", "runs", "macro_f1_mean", "accuracy_mean"],
    )
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "failed": len(failures), "success": len(rows) - len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
