from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv, write_json
from experiments.scripts.gate14_task_first_common import (
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    load_hgb_graph,
    run_multilevel_task_first,
    run_parallel,
)


SOURCES = (
    "random_support",
    "graph_sketch",
    "target_anchor_co_support",
    "class_footprint_knn",
    "target_response_signature_knn",
    "relation_response_knn",
    "hybrid_task_aware",
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate14 candidate-source diagnostics."))
    parser.set_defaults(ratios=[0.20])
    parser.add_argument("--methods", nargs="+", default=["HeSF-TC-P-response-static", "HeSF-TC-S-response-static", "HeSF-TC-stateful-v1"])
    parser.add_argument("--candidate-sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--skip-task-eval", action="store_true")
    return parser


def _worker(args: argparse.Namespace, dataset: str, method: str, source: str, ratio: float, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset, "method": method, "candidate_source": source, "ratio": float(ratio), "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        pair_delta = "stateful_signature" if "stateful" in method else "response_signature"
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
            pair_delta_mode=pair_delta,
            coverage_mode="coverage_v2" if "coverage-v2" in method else "combined",
            purity_policy="purity_v2" if "purity-v2" in method else "unknown_blocks_known",
        )
        row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
        if not bool(args.skip_task_eval):
            add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
            row["macro_f1"] = row.get("task.macro_f1")
            row["accuracy"] = row.get("task.accuracy")
            row["validation_macro_f1"] = row.get("task.validation_macro_f1")
            row["validation_accuracy"] = row.get("task.validation_accuracy")
        row["generated_candidates"] = row.get("candidate_pair_count_last", row.get("candidate_candidate_pairs_emitted_last", ""))
        row["retained_candidates"] = row.get("candidate_pair_count_last", "")
        row["scored_candidates"] = row.get("eligible_candidate_pair_count_last", "")
        selected_merges = row.get("selected_support_merges", row.get("selected_support_merges_last", ""))
        row["selected_merges"] = selected_merges
        row["selected_share"] = float(selected_merges or 0) / max(float(row.get("eligible_candidate_pair_count_last", 0) or 0), 1.0)
        row["avg_score"] = row.get("selected_norm_score_task_first_p50_last", "")
        row["avg_delta_target_spec"] = row.get("selected_norm_delta_target_spec_p50_last", "")
        row["avg_delta_rel_response"] = row.get("selected_norm_delta_rel_response_p50_last", "")
        row["avg_coverage_v2_collision"] = row.get("coverage_v2_error_last", "")
        row["avg_purity_v2_collision"] = row.get("purity_v2_error_last", "")
        row["candidate_generation_sec"] = row.get("candidate_total_sec_last", "")
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
    combos = [(dataset, method, source, ratio, seed) for dataset in args.datasets for method in args.methods for source in args.candidate_sources for ratio in args.ratios for seed in args.seeds]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "candidate_source_runs.csv")
    write_csv(args.output / "candidate_source_runs.csv", rows)
    diag_cols = [
        "dataset", "method", "candidate_source", "ratio", "seed", "generated_candidates", "retained_candidates",
        "scored_candidates", "selected_merges", "selected_share", "avg_score", "avg_delta_target_spec",
        "avg_delta_rel_response", "avg_coverage_v2_collision", "avg_purity_v2_collision", "candidate_generation_sec",
    ]
    write_csv(args.output / "gate14_candidate_source_diagnostics.csv", [{key: row.get(key) for key in diag_cols} for row in rows])
    summary = aggregate_rows(rows, ["method", "candidate_source", "ratio"], ("macro_f1", "accuracy", "selected_share"))
    write_csv(args.output / "candidate_source_summary.csv", summary)
    (args.output / "candidate_source_summary.md").write_text(
        "# Candidate Source Summary\n\n" + markdown_table(summary, ["method", "candidate_source", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "selected_share_mean"]) + "\n",
        encoding="utf-8",
    )
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
