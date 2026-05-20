from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.gate13_task_first_common import (
    add_common_args,
    add_task_and_optional_spectral,
    load_hgb_graph,
    run_multilevel_task_first,
    run_parallel,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 ratio-budget sanity."))
    parser.set_defaults(ratios=[0.012, 0.024, 0.048, 0.096, 0.20, 0.50])
    parser.add_argument("--method", default="HeSF-TC-P")
    parser.add_argument("--candidate-source", default="target_response_knn")
    parser.add_argument("--pair-delta-mode", default="response_signature")
    return parser


def _worker(args: argparse.Namespace, dataset: str, ratio: float, seed: int) -> dict:
    row = {"dataset": dataset, "method": args.method, "ratio": float(ratio), "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        coarse, assignment, diag = run_multilevel_task_first(
            original,
            method=str(args.method),
            ratio=float(ratio),
            ratio_mode=str(args.ratio_mode),
            seed=int(seed),
            max_levels=int(args.max_levels),
            per_level_ratio=float(args.per_level_ratio),
            candidate_k=int(args.candidate_k),
            candidate_source=str(args.candidate_source),
            pair_delta_mode=str(args.pair_delta_mode),
            coverage_mode="combined",
            purity_policy="unknown_blocks_known",
        )
        row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
        row["level_count"] = len(diag.get("levels", []))
        add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
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
    combos = [(dataset, ratio, seed) for dataset in args.datasets for ratio in args.ratios for seed in args.seeds]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "ratio_budget_sanity_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), float(row.get("ratio", 0)), int(row.get("seed", 0))))
    write_csv(args.output / "ratio_budget_sanity_runs.csv", rows)
    level_rows = []
    for row in rows:
        for key in row:
            if key.endswith("_last") or key in {"dataset", "ratio", "seed", "stop_reason", "floor_reason", "realized_support_ratio"}:
                pass
        level_rows.append({
            "dataset": row.get("dataset"),
            "ratio": row.get("ratio"),
            "seed": row.get("seed"),
            "requested_support_ratio": row.get("requested_ratio"),
            "realized_support_ratio": row.get("realized_support_ratio"),
            "desired_final_support_nodes": row.get("desired_final_support_nodes"),
            "selected_support_merges_last": row.get("selected_support_merges_last"),
            "candidate_pair_count_last": row.get("candidate_pair_count_last"),
            "eligible_candidate_pair_count_last": row.get("eligible_candidate_pair_count_last"),
            "stop_reason": row.get("stop_reason"),
            "floor_reason": row.get("floor_reason"),
        })
    write_csv(args.output / "ratio_budget_by_level.csv", level_rows)
    write_summary_md(args.output / "ratio_budget_summary.md", "Gate13 Ratio Budget Sanity", level_rows, ["dataset", "ratio", "seed", "realized_support_ratio", "stop_reason", "floor_reason"])
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
