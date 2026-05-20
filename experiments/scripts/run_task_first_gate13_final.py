from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.gate13_task_first_common import (
    DEFAULT_METRICS,
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    load_hgb_graph,
    run_full_graph_ceiling_row,
    run_multilevel_task_first,
    run_parallel,
    run_support_baseline,
    write_summary_md,
)
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.hettree_task import infer_target_node_type


BASELINES = {
    "flatten-sum-support-only",
    "H6-no-spec-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
    "sketch-support-only-basic",
}


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate13 final table."))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "HeSF-TC-P-response",
            "HeSF-TC-S-response",
            "HeSF-TC-no-target-spec",
            "HeSF-TC-no-coverage",
            "HeSF-TC-no-purity",
            "flatten-sum-support-only",
            "H6-no-spec-support-only",
            "TypedHash-ChebHeat-support-only",
            "random-support-only",
            "full-graph-hettree-lite-ceiling",
            "A0-current-all-type-coarse-transfer-reference",
        ],
    )
    return parser


def _worker(args: argparse.Namespace, dataset: str, method: str, ratio: float, seed: int) -> dict:
    row = {"dataset": dataset, "method": method, "ratio": float(ratio), "seed": int(seed), "status": "running"}
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        if method == "full-graph-hettree-lite-ceiling":
            row.update(run_full_graph_ceiling_row(args, dataset, int(seed), "hettree_lite"))
            row["ratio"] = float(ratio)
            row["method"] = method
            row["realized_support_ratio"] = 1.0
            row["realized_full_ratio"] = 1.0
            row["target_hit"] = True
            row["selected_support_merges"] = 0
            row["num_levels"] = 0
            row["task.macro_f1"] = row.get("macro_f1")
            row["task.micro_f1"] = row.get("micro_f1")
            row["task.accuracy"] = row.get("accuracy")
        elif method == "A0-current-all-type-coarse-transfer-reference":
            target_type = infer_target_node_type(original)
            support = int(np.sum(original.node_type != int(target_type)))
            target_count = int(original.num_nodes - support)
            desired_support = max(0, int(np.ceil(support * float(ratio) - 1.0e-12)))
            full_ratio = float((target_count + desired_support) / max(original.num_nodes, 1))
            coarse, assignment, diag = coarsen_type_isolated_lsh(
                original,
                target_ratio=full_ratio,
                seed=int(seed),
                hash_bits=20,
                bucket_topk=4,
                assignment_source="chebheat_sketch",
            )
            row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
            row["realized_support_ratio"] = ""
            row["realized_full_ratio"] = float(coarse.num_nodes / max(original.num_nodes, 1))
            row["target_hit"] = False
            row["selected_support_merges"] = ""
            row["num_levels"] = 1
            add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment.assignment, seed=int(seed), args=args)
            row["status"] = "success"
        elif method in BASELINES:
            coarse, assignment, diag = run_support_baseline(
                original,
                baseline=method,
                ratio=float(ratio),
                seed=int(seed),
                candidate_k=int(args.candidate_k),
            )
            row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
            add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
            row["status"] = "success"
        else:
            coarse, assignment, diag = run_multilevel_task_first(
                original,
                method=method,
                ratio=float(ratio),
                ratio_mode=str(args.ratio_mode),
                seed=int(seed),
                max_levels=int(args.max_levels),
                per_level_ratio=float(args.per_level_ratio),
                candidate_k=int(args.candidate_k),
                candidate_source="target_response_knn",
                pair_delta_mode="response_signature",
                coverage_mode="combined" if method != "HeSF-TC-no-coverage" else "old_common_anchor_only",
                purity_policy="unknown_blocks_known",
            )
            row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
            add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
            row["status"] = "success"
        row["macro_f1"] = row.get("task.macro_f1", row.get("macro_f1"))
        row["micro_f1"] = row.get("task.micro_f1", row.get("micro_f1"))
        row["accuracy"] = row.get("task.accuracy", row.get("accuracy"))
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "oom_or_runtime_error" if "out of memory" in message.lower() else "failed"
        row["error"] = message
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def _gap_tables(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple, dict[str, list[dict]]] = {}
    for row in rows:
        if row.get("status") == "success":
            grouped.setdefault((row.get("dataset"), row.get("ratio"), row.get("seed")), {}).setdefault(str(row.get("method")), []).append(row)
    gaps = []
    wins = []
    recovery = []
    for key, by_method in grouped.items():
        ceiling = by_method.get("full-graph-hettree-lite-ceiling", [{}])[0]
        ceiling_macro = float(ceiling.get("macro_f1", ceiling.get("task.macro_f1", 0)) or 0)
        ceiling_acc = float(ceiling.get("accuracy", ceiling.get("task.accuracy", 0)) or 0)
        baselines = [name for name in BASELINES if name in by_method]
        hesf = [name for name in by_method if str(name).startswith("HeSF-TC")]
        for method in hesf:
            row = by_method[method][0]
            macro = float(row.get("macro_f1", 0) or 0)
            acc = float(row.get("accuracy", 0) or 0)
            recovery.append({"dataset": key[0], "ratio": key[1], "seed": key[2], "method": method, "recovery_vs_full_lite_macro": macro / ceiling_macro if ceiling_macro else "", "recovery_vs_full_lite_accuracy": acc / ceiling_acc if ceiling_acc else ""})
            for baseline in baselines:
                brow = by_method[baseline][0]
                bmacro = float(brow.get("macro_f1", 0) or 0)
                bacc = float(brow.get("accuracy", 0) or 0)
                gaps.append({"dataset": key[0], "ratio": key[1], "seed": key[2], "method": method, "baseline": baseline, "delta_macro_f1": macro - bmacro, "delta_accuracy": acc - bacc})
                wins.append({"dataset": key[0], "ratio": key[1], "seed": key[2], "method": method, "baseline": baseline, "macro_win": macro > bmacro, "accuracy_win": acc > bacc})
    return gaps, wins, recovery


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    combos = [(dataset, method, ratio, seed) for dataset in args.datasets for method in args.methods for ratio in args.ratios for seed in args.seeds]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    rows = run_parallel(combos, _worker, args, args.output / "gate13_final_runs.csv")
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), int(row.get("seed", 0))))
    write_csv(args.output / "gate13_final_runs.csv", rows)
    by_dataset = aggregate_rows(rows, ["dataset", "method", "ratio"], DEFAULT_METRICS + ("macro_f1", "accuracy"))
    by_method = aggregate_rows(rows, ["method", "ratio"], DEFAULT_METRICS + ("macro_f1", "accuracy"))
    write_csv(args.output / "gate13_final_by_dataset.csv", by_dataset)
    write_csv(args.output / "gate13_final_by_method.csv", by_method)
    gaps, wins, recovery = _gap_tables(rows)
    write_csv(args.output / "gate13_final_gap_vs_baselines.csv", gaps)
    write_csv(args.output / "gate13_final_win_rate.csv", wins)
    write_csv(args.output / "gate13_final_recovery_vs_ceiling.csv", recovery)
    diag_cols = ["dataset", "method", "ratio", "seed", "target_spec_error_last", "relation_response_error_last", "support_coverage_error_last", "support_purity_error_last", "selected_known_unknown_count_last", "selected_unknown_unknown_count_last", "coverage_cross_anchor_collision_loss_mean_last"]
    write_csv(args.output / "gate13_final_selected_merge_diagnostics.csv", [{key: row.get(key) for key in diag_cols} for row in rows])
    write_summary_md(args.output / "gate13_final_summary.md", "Gate13 Final HGB Table", by_method, ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "realized_support_ratio_mean"])
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
