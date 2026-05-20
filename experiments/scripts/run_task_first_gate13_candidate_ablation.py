from __future__ import annotations

import argparse
import json
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
    run_multilevel_task_first,
    run_parallel,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 candidate-source ablation."))
    parser.add_argument("--methods", nargs="+", default=["HeSF-TC-P", "HeSF-TC-S"])
    parser.add_argument(
        "--candidate-sources",
        nargs="+",
        default=["random_support", "sketch", "target_anchor_co_support", "class_footprint_knn", "target_response_knn"],
    )
    parser.add_argument("--pair-delta-mode", default="local_surrogate")
    return parser


def _worker(args: argparse.Namespace, dataset: str, method: str, ratio: float, source: str, seed: int) -> dict:
    row = {"dataset": dataset, "method": method, "ratio": float(ratio), "candidate_source": source, "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        coarse, assignment, diag = run_multilevel_task_first(
            original,
            method=method,
            ratio=float(ratio),
            ratio_mode=str(args.ratio_mode),
            seed=int(seed),
            max_levels=int(args.max_levels),
            per_level_ratio=float(args.per_level_ratio),
            candidate_k=int(args.candidate_k),
            candidate_source=source,
            pair_delta_mode=str(args.pair_delta_mode),
            coverage_mode="old_common_anchor_only",
            purity_policy="zero_as_no_conflict",
        )
        row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
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
    combos = [
        (dataset, method, ratio, source, seed)
        for dataset in args.datasets
        for method in args.methods
        for ratio in args.ratios
        for source in args.candidate_sources
        for seed in args.seeds
    ]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "candidate_source_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), str(row.get("candidate_source")), int(row.get("seed", 0))))
    write_csv(args.output / "candidate_source_runs.csv", rows)
    by_dataset = aggregate_rows(rows, ["dataset", "method", "ratio", "candidate_source"], DEFAULT_METRICS)
    write_csv(args.output / "candidate_source_by_dataset.csv", by_dataset)
    write_summary_md(args.output / "candidate_source_summary.md", "Gate13 Candidate Source Ablation", by_dataset, ["dataset", "method", "ratio", "candidate_source", "runs", "task.macro_f1_mean", "task.accuracy_mean", "realized_support_ratio_mean"])
    dist_rows = []
    selected_rows = []
    for row in rows:
        dist_rows.append({key: row.get(key) for key in ("dataset", "method", "ratio", "candidate_source", "seed", "candidate_source_counts_last", "candidate_candidate_pairs_retained_last")})
        selected_rows.append({key: row.get(key) for key in ("dataset", "method", "ratio", "candidate_source", "seed", "selected_merges_by_source_last")})
    write_csv(args.output / "candidate_source_distribution.csv", dist_rows)
    write_csv(args.output / "selected_merge_source_distribution.csv", selected_rows)
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
