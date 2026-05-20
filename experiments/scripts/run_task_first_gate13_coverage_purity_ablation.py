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
    run_multilevel_task_first,
    run_parallel,
    write_summary_md,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 coverage and purity ablation."))
    parser.add_argument("--methods", nargs="+", default=["HeSF-TC-P", "HeSF-TC-S"])
    parser.add_argument("--candidate-source", default="target_response_knn")
    parser.add_argument("--pair-delta-mode", default="response_signature")
    parser.add_argument("--coverage-modes", nargs="+", default=["old_common_anchor_only", "cross_anchor_collision", "class_context_collision", "combined"])
    parser.add_argument("--purity-policies", nargs="+", default=["zero_as_no_conflict", "unknown_blocks_known", "unknown_propagated", "unknown_only_merge"])
    parser.add_argument("--js-thresholds", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.50])
    return parser


def _worker(args: argparse.Namespace, dataset: str, method: str, ratio: float, coverage: str, purity: str, threshold: float, seed: int) -> dict:
    row = {"dataset": dataset, "method": method, "ratio": float(ratio), "coverage_mode": coverage, "purity_policy": purity, "js_threshold": float(threshold), "seed": int(seed), "status": "running"}
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
            candidate_source=str(args.candidate_source),
            pair_delta_mode=str(args.pair_delta_mode),
            coverage_mode=coverage,
            purity_policy=purity,
            js_threshold=float(threshold),
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
    combo_set = set()
    for dataset in args.datasets:
        for method in args.methods:
            for ratio in args.ratios:
                for seed in args.seeds:
                    for coverage in args.coverage_modes:
                        combo_set.add((dataset, method, ratio, coverage, "unknown_blocks_known", 0.35, seed))
                    for purity in args.purity_policies:
                        for threshold in args.js_thresholds:
                            combo_set.add((dataset, method, ratio, "combined", purity, float(threshold), seed))
    combos = sorted(combo_set, key=lambda item: (str(item[0]), str(item[1]), float(item[2]), str(item[3]), str(item[4]), float(item[5]), int(item[6])))
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "coverage_purity_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), str(row.get("coverage_mode")), str(row.get("purity_policy")), float(row.get("js_threshold", 0)), int(row.get("seed", 0))))
    write_csv(args.output / "coverage_purity_runs.csv", rows)
    coverage_by = aggregate_rows(rows, ["dataset", "method", "ratio", "coverage_mode"], DEFAULT_METRICS)
    purity_by = aggregate_rows(rows, ["dataset", "method", "ratio", "purity_policy", "js_threshold"], DEFAULT_METRICS)
    write_csv(args.output / "coverage_ablation_by_dataset.csv", coverage_by)
    write_csv(args.output / "purity_ablation_by_dataset.csv", purity_by)
    diag_cols = ["dataset", "method", "ratio", "coverage_mode", "purity_policy", "js_threshold", "seed", "coverage_same_anchor_loss_mean_last", "coverage_cross_anchor_collision_loss_mean_last", "coverage_class_context_collision_loss_mean_last", "selected_known_unknown_count_last", "selected_unknown_unknown_count_last"]
    write_csv(args.output / "coverage_collision_diagnostics.csv", [{key: row.get(key) for key in diag_cols} for row in rows])
    write_csv(args.output / "purity_policy_diagnostics.csv", [{key: row.get(key) for key in diag_cols + ["known_footprint_count_last", "unknown_target_connected_count_last", "unknown_isolated_count_last"]} for row in rows])
    write_summary_md(args.output / "coverage_summary.md", "Gate13 Coverage Ablation", coverage_by, ["dataset", "method", "ratio", "coverage_mode", "runs", "task.macro_f1_mean", "task.accuracy_mean"])
    write_summary_md(args.output / "purity_summary.md", "Gate13 Purity Ablation", purity_by, ["dataset", "method", "ratio", "purity_policy", "js_threshold", "runs", "task.macro_f1_mean", "task.accuracy_mean"])
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
