from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv, write_json
from experiments.scripts.gate14_task_first_common import (
    BASELINE_METHODS,
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    build_ratio_matched_rows,
    load_hgb_graph,
    run_parallel,
    run_support_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate14 ratio-matched support baselines."))
    parser.set_defaults(ratios=[0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--baselines", nargs="+", default=list(BASELINE_METHODS))
    return parser


def _worker(args: argparse.Namespace, dataset: str, method: str, ratio: float, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset, "method": method, "ratio": float(ratio), "requested_support_ratio": float(ratio), "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        coarse, assignment, diag = run_support_baseline(
            original,
            baseline=method,
            ratio=float(ratio),
            seed=int(seed),
            candidate_k=int(args.candidate_k),
        )
        row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
        add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
        row["macro_f1"] = row.get("task.macro_f1")
        row["micro_f1"] = row.get("task.micro_f1")
        row["accuracy"] = row.get("task.accuracy")
        row["validation_macro_f1"] = row.get("task.validation_macro_f1")
        row["validation_accuracy"] = row.get("task.validation_accuracy")
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
    combos = [(dataset, method, ratio, seed) for dataset in args.datasets for method in args.baselines for ratio in args.ratios for seed in args.seeds]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "baseline_requested_ratio_table.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), int(row.get("seed", 0))))
    write_csv(args.output / "baseline_requested_ratio_table.csv", rows)
    realized = aggregate_rows(rows, ["dataset", "method", "requested_support_ratio"], ("realized_support_ratio", "realized_full_ratio", "macro_f1", "accuracy", "validation_macro_f1", "validation_accuracy"))
    write_csv(args.output / "baseline_realized_ratio_table.csv", realized)
    # This stage has only baselines; final Gate14 will rebuild this table with HeSF rows.
    write_csv(args.output / "baseline_nearest_ratio_matched_table.csv", build_ratio_matched_rows([], rows))
    report_rows = [
        row for row in realized
        if abs(float(row.get("realized_support_ratio_mean", 0.0) or 0.0) - float(row.get("requested_support_ratio", 0.0) or 0.0)) > 0.025
    ]
    report = "# Baseline Ratio Mismatch Report\n\n"
    report += markdown_table(report_rows, ["dataset", "method", "requested_support_ratio", "realized_support_ratio_mean", "macro_f1_mean", "accuracy_mean"])
    (args.output / "baseline_ratio_mismatch_report.md").write_text(report + "\n", encoding="utf-8")
    (args.output / "ratio_matched_baseline_summary.md").write_text(report + "\n", encoding="utf-8")
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
