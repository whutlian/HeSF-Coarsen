from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv, write_json
from experiments.scripts.gate14_task_first_common import add_common_args, aggregate_rows, run_parallel
from experiments.scripts.run_task_first_gate14_final import _worker


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate14 coverage/purity/stateful repair ablations."))
    parser.set_defaults(ratios=[0.12, 0.20, 0.30])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "HeSF-TC-P-response-static",
            "HeSF-TC-no-coverage",
            "HeSF-TC-coverage-v2",
            "HeSF-TC-purity-v2",
            "HeSF-TC-coverage-v2-purity-v2",
            "HeSF-TC-stateful-v1",
            "HeSF-TC-stateful-v1-coverage-v2",
            "HeSF-TC-stateful-v1-purity-v2",
            "HeSF-TC-stateful-v1-coverage-v2-purity-v2",
        ],
    )
    parser.add_argument("--candidate-source", default="hybrid_task_aware")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [(dataset, method, ratio, seed) for dataset in args.datasets for method in args.methods for ratio in args.ratios for seed in args.seeds]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "repair_ablation_runs.csv")
    write_csv(args.output / "repair_ablation_runs.csv", rows)
    summary = aggregate_rows(rows, ["method", "ratio"], ("macro_f1", "accuracy", "coverage_v2_error_last", "purity_v2_error_last", "stateful_signature_drift_last"))
    write_csv(args.output / "repair_ablation_summary.csv", summary)
    (args.output / "coverage_v2_summary.md").write_text(
        "# Coverage V2 Summary\n\n" + markdown_table([row for row in summary if "coverage" in row.get("method", "")], ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "coverage_v2_error_last_mean"]) + "\n",
        encoding="utf-8",
    )
    (args.output / "purity_v2_summary.md").write_text(
        "# Purity V2 Summary\n\n" + markdown_table([row for row in summary if "purity" in row.get("method", "")], ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "purity_v2_error_last_mean"]) + "\n",
        encoding="utf-8",
    )
    (args.output / "stateful_matching_summary.md").write_text(
        "# Stateful Matching Summary\n\n" + markdown_table([row for row in summary if "stateful" in row.get("method", "")], ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "stateful_signature_drift_last_mean"]) + "\n",
        encoding="utf-8",
    )
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
