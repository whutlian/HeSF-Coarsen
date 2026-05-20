from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.gate13_task_first_common import (
    DEFAULT_METRICS,
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    load_hgb_graph,
    run_parallel,
    run_support_baseline,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 support-only baselines."))
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["flatten-sum-support-only", "H6-no-spec-support-only", "TypedHash-ChebHeat-support-only", "random-support-only", "sketch-support-only-basic"],
    )
    return parser


def _worker(args: argparse.Namespace, dataset: str, baseline: str, ratio: float, seed: int) -> dict:
    row = {"dataset": dataset, "method": baseline, "support_ratio": float(ratio), "ratio": float(ratio), "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        coarse, assignment, diag = run_support_baseline(
            original,
            baseline=baseline,
            ratio=float(ratio),
            seed=int(seed),
            candidate_k=int(args.candidate_k),
        )
        row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
        add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
        row["macro_f1"] = row.get("task.macro_f1")
        row["micro_f1"] = row.get("task.micro_f1")
        row["accuracy"] = row.get("task.accuracy")
        row["status"] = "success"
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "oom_or_runtime_error" if "out of memory" in message.lower() else "failed"
        row["error"] = message
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [(dataset, baseline, ratio, seed) for dataset in args.datasets for baseline in args.baselines for ratio in args.ratios for seed in args.seeds]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "support_only_baseline_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), int(row.get("seed", 0))))
    write_csv(args.output / "support_only_baseline_runs.csv", rows)
    by_dataset = aggregate_rows(rows, ["dataset", "method", "ratio"], DEFAULT_METRICS)
    write_csv(args.output / "support_only_baseline_by_dataset.csv", by_dataset)
    write_summary_md(args.output / "support_only_baseline_summary.md", "Gate13 Support-Only Baselines", by_dataset, ["dataset", "method", "ratio", "runs", "task.macro_f1_mean", "task.accuracy_mean", "realized_support_ratio_mean"])
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
