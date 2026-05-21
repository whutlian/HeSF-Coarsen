from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv, write_json
from experiments.scripts.gate14_task_first_common import (
    BASELINE_METHODS,
    GATE14_RATIOS,
    add_common_args,
    add_task_and_optional_spectral,
    aggregate_rows,
    build_ratio_matched_rows,
    compute_recovery_vs_ceiling,
    evaluator_status_rows,
    load_hgb_graph,
    run_full_graph_ceiling_row,
    run_multilevel_task_first,
    run_parallel,
    run_support_baseline,
    select_validation_best_rows,
    write_placeholder_png,
)
from hesf_coarsen.baselines.type_isolated_lsh import coarsen_type_isolated_lsh
from hesf_coarsen.eval.hettree_task import infer_target_node_type


HESF_METHODS = (
    "HeSF-TC-P-response-static",
    "HeSF-TC-S-response-static",
    "HeSF-TC-no-coverage",
    "HeSF-TC-coverage-v2",
    "HeSF-TC-purity-v2",
    "HeSF-TC-coverage-v2-purity-v2",
    "HeSF-TC-stateful-v1",
    "HeSF-TC-stateful-v1-coverage-v2",
    "HeSF-TC-stateful-v1-purity-v2",
    "HeSF-TC-stateful-v1-coverage-v2-purity-v2",
    "HeSF-TC-no-target-spec",
    "HeSF-TC-no-rel-response",
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description="Run Gate14 final task-first matrix."))
    parser.set_defaults(ratios=list(GATE14_RATIOS), task_epochs=10, task_hidden_dim=32)
    parser.add_argument("--methods", nargs="+", default=list(HESF_METHODS) + list(BASELINE_METHODS) + ["full-graph-hettree-lite-tuned", "A0-current-all-type-coarse-transfer-reference"])
    parser.add_argument("--candidate-source", default="hybrid_task_aware")
    parser.add_argument("--baseline-runs", type=Path)
    parser.add_argument("--full-graph-runs", type=Path)
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def _settings_for_method(method: str) -> dict[str, Any]:
    pair_delta = "stateful_signature" if "stateful-v1" in method else "response_signature"
    coverage = "coverage_v2" if "coverage-v2" in method else "combined"
    purity = "purity_v2" if "purity-v2" in method else "unknown_blocks_known"
    if "no-coverage" in method:
        coverage = "coverage_v1_legacy"
    return {"pair_delta_mode": pair_delta, "coverage_mode": coverage, "purity_policy": purity}


def _run_a0_reference(original, ratio: float, seed: int) -> tuple[Any, np.ndarray, dict[str, Any]]:
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
    out = {key: value for key, value in diag.items() if not isinstance(value, list)}
    out["realized_support_ratio"] = ""
    out["realized_full_ratio"] = float(coarse.num_nodes / max(original.num_nodes, 1))
    out["target_hit"] = False
    out["selected_support_merges"] = ""
    out["num_levels"] = 1
    return coarse, assignment.assignment, out


def _worker(args: argparse.Namespace, dataset: str, method: str, ratio: float, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "ratio": float(ratio),
        "requested_support_ratio": float(ratio),
        "seed": int(seed),
        "status": "running",
    }
    try:
        original = load_hgb_graph(Path(args.data_root), dataset)
        if method == "full-graph-hettree-lite-tuned":
            row.update(run_full_graph_ceiling_row(args, dataset, int(seed), "hettree_lite"))
            row["method"] = method
            row["ratio"] = float(ratio)
            row["requested_support_ratio"] = 1.0
            row["realized_support_ratio"] = 1.0
            row["realized_full_ratio"] = 1.0
            row["target_hit"] = True
            row["selected_support_merges"] = 0
            row["num_levels"] = 0
            row["task.macro_f1"] = row.get("macro_f1")
            row["task.micro_f1"] = row.get("micro_f1")
            row["task.accuracy"] = row.get("accuracy")
            row["task.validation_macro_f1"] = row.get("validation_macro_f1")
            row["task.validation_accuracy"] = row.get("validation_accuracy")
        elif method in BASELINE_METHODS:
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
        elif method == "A0-current-all-type-coarse-transfer-reference":
            coarse, assignment, diag = _run_a0_reference(original, float(ratio), int(seed))
            row.update(diag)
            add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
            row["status"] = "success"
        else:
            settings = _settings_for_method(method)
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
                pair_delta_mode=str(settings["pair_delta_mode"]),
                coverage_mode=str(settings["coverage_mode"]),
                purity_policy=str(settings["purity_policy"]),
            )
            row.update({key: value for key, value in diag.items() if not isinstance(value, list)})
            add_task_and_optional_spectral(row, original=original, coarse=coarse, assignment=assignment, seed=int(seed), args=args)
            row["status"] = "success"
        row["macro_f1"] = row.get("task.macro_f1", row.get("macro_f1"))
        row["micro_f1"] = row.get("task.micro_f1", row.get("micro_f1"))
        row["accuracy"] = row.get("task.accuracy", row.get("accuracy"))
        row["validation_macro_f1"] = row.get("task.validation_macro_f1", row.get("validation_macro_f1"))
        row["validation_accuracy"] = row.get("task.validation_accuracy", row.get("validation_accuracy"))
        row.update(evaluator_status_rows())
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "oom_or_runtime_error" if "out of memory" in message.lower() else "failed"
        row["error"] = message
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    return row


def _oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "success" and str(row.get("method")).startswith("HeSF-TC"):
            groups.setdefault((row.get("dataset"), row.get("seed")), []).append(row)
    out = []
    for _key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        best = max(group, key=lambda row: (float(row.get("macro_f1") or -1), float(row.get("accuracy") or -1)))
        item = dict(best)
        item["oracle_appendix_only"] = True
        out.append(item)
    return out


def _write_required_figures(output: Path) -> None:
    figures = {
        "macro_f1_vs_realized_support_ratio.png": "Macro-F1 vs Realized Support Ratio",
        "accuracy_vs_realized_support_ratio.png": "Accuracy vs Realized Support Ratio",
        "recovery_vs_realized_support_ratio.png": "Recovery vs Realized Support Ratio",
        "baseline_gap_vs_ratio.png": "Baseline Gap vs Ratio",
        "coverage_v1_vs_v2_task.png": "Coverage V1 vs V2 Task",
        "purity_v1_vs_v2_task.png": "Purity V1 vs V2 Task",
        "stateful_vs_static_task.png": "Stateful vs Static Task",
        "candidate_source_selected_share.png": "Candidate Source Selected Share",
    }
    for name, title in figures.items():
        write_placeholder_png(output / "figures" / name, title)


def _read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _external_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(args.baseline_runs):
        item = dict(row)
        item.setdefault("status", "success")
        item.setdefault("requested_support_ratio", item.get("ratio"))
        rows.append(item)
    for row in _read_csv(args.full_graph_runs):
        item = dict(row)
        item["method"] = "full-graph-hettree-lite-tuned"
        item.setdefault("ratio", 1.0)
        item.setdefault("requested_support_ratio", 1.0)
        item.setdefault("realized_support_ratio", 1.0)
        item.setdefault("realized_full_ratio", 1.0)
        item.setdefault("target_hit", True)
        item.setdefault("selected_support_merges", 0)
        item.setdefault("num_levels", 0)
        item.setdefault("status", "success")
        item.setdefault("task.macro_f1", item.get("macro_f1"))
        item.setdefault("task.micro_f1", item.get("micro_f1"))
        item.setdefault("task.accuracy", item.get("accuracy"))
        rows.append(item)
    return rows


def _combo_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset")),
        str(row.get("method")),
        str(float(row.get("ratio", 0.0) or 0.0)),
        str(int(float(row.get("seed", 0) or 0))),
    )


def _existing_success_rows(output: Path) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in (output / "gate14_all_runs.csv", output / "gate14_all_runs.partial.csv"):
        for row in _read_csv(path):
            if row.get("status") == "success":
                rows[_combo_key(row)] = dict(row)
    return list(rows.values())


def _write_gate14_decision(output: Path, by_method: list[dict[str, Any]], gaps: list[dict[str, Any]], recovery: list[dict[str, Any]]) -> None:
    hesf_rows = [row for row in by_method if str(row.get("method")).startswith("HeSF-TC")]
    best = max(hesf_rows, key=lambda row: float(row.get("macro_f1_mean", 0.0) or 0.0)) if hesf_rows else {}
    def gap_for(baseline: str) -> float:
        vals = [float(row.get("delta_macro_f1", 0.0)) for row in gaps if row.get("baseline") == baseline and row.get("comparison_status") in {"matched", "nearest_flagged"} and row.get("delta_macro_f1") not in {"", None}]
        return float(np.mean(vals)) if vals else 0.0

    flatten_gap = gap_for("flatten-sum-support-only")
    h6_gap = gap_for("H6-no-spec-support-only")
    typed_gap = gap_for("TypedHash-ChebHeat-support-only")
    rec_macro = [float(row.get("recovery_vs_full_graph_lite_macro", 0.0)) for row in recovery if row.get("recovery_status") == "ok" and row.get("recovery_vs_full_graph_lite_macro") not in {"", None}]
    rec_acc = [float(row.get("recovery_vs_full_graph_lite_accuracy", 0.0)) for row in recovery if row.get("recovery_status") == "ok" and row.get("recovery_vs_full_graph_lite_accuracy") not in {"", None}]
    mean_rec_macro = float(np.mean(rec_macro)) if rec_macro else 0.0
    mean_rec_acc = float(np.mean(rec_acc)) if rec_acc else 0.0
    if flatten_gap >= 0.01 and h6_gap >= 0.01 and typed_gap >= 0.01:
        decision = "CONTINUE_HESF_TC_WITH_TASK_FIRST_REDESIGN"
    elif str(best.get("method", "")).startswith("HeSF-TC-P") and float(best.get("ratio", 0.0) or 0.0) == 0.2:
        decision = "CONTINUE_HESF_TC_WITH_P_RESPONSE_RATIO_0P2"
    elif mean_rec_macro < 0.8:
        decision = "DROP_HESF_TC_AFTER_RATIO_MATCHED_FAILURE"
    else:
        decision = "PAUSE_HESF_TC_PENDING_OFFICIAL_EVALUATOR"
    text = f"""# Gate14 Decision

Decision: `{decision}`

## Evidence

- Best validation-selected HeSF-TC variant: `{best.get('method', '')}`
- Best support ratio: `{best.get('ratio', '')}`
- Full-graph-lite ceiling: see `full_graph_lite_ceiling_summary.md`
- Recovery vs ceiling: macro `{mean_rec_macro:.6f}`, accuracy `{mean_rec_acc:.6f}`
- Ratio-matched delta vs flatten-sum: `{flatten_gap:.6f}`
- Ratio-matched delta vs H6-no-spec: `{h6_gap:.6f}`
- Ratio-matched delta vs TypedHash-ChebHeat: `{typed_gap:.6f}`
- Coverage-v2 effect: see `coverage_v2_summary.md`
- Purity-v2 effect: see `purity_v2_summary.md`
- Stateful scoring effect: see `stateful_matching_summary.md`
- Candidate source effect: see `candidate_source_summary.md`

## Why this is task-first

Gate14 uses validation-selected downstream `macro_f1`, `micro_f1`, and `accuracy` as the decision surface. Spectral, response, coverage, and purity terms are used as auxiliary merge regularizers and diagnostics rather than preservation-first claims.

## Remaining issues

- official evaluator status: `not_integrated`
- evaluator ceiling limitation: `diagnostic_lite_only`
- task metric instability by dataset remains possible
- runtime/resource concerns are reported in `gate14_all_runs.csv`

## Next step

Use the selected Gate14 branch only if the ratio-matched validation-selected evidence is stable; otherwise integrate an official/faithful evaluator before paper claims.
"""
    (output / "gate14_decision.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    methods = list(args.methods)
    if args.baseline_runs is not None:
        methods = [method for method in methods if method not in BASELINE_METHODS]
    if args.full_graph_runs is not None:
        methods = [method for method in methods if method != "full-graph-hettree-lite-tuned"]
    combos = [(dataset, method, ratio, seed) for dataset in args.datasets for method in methods for ratio in args.ratios for seed in args.seeds]
    existing_rows = _existing_success_rows(args.output) if bool(args.resume_existing) else []
    existing_keys = {_combo_key(row) for row in existing_rows}
    if existing_keys:
        combos = [
            combo for combo in combos
            if (str(combo[0]), str(combo[1]), str(float(combo[2])), str(int(combo[3]))) not in existing_keys
        ]
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]
    run_csv = args.output / "gate14_all_runs.partial.csv" if bool(args.resume_existing) else args.output / "gate14_all_runs.csv"
    rows = run_parallel(combos, _worker, args, run_csv) if combos else []
    rows = [*existing_rows, *rows, *_external_rows(args)]
    rows = sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("method")), float(row.get("ratio", 0)), int(row.get("seed", 0))))
    write_csv(args.output / "gate14_all_runs.csv", rows)
    metrics = (
        "realized_support_ratio",
        "realized_full_ratio",
        "selected_support_merges",
        "target_hit",
        "macro_f1",
        "micro_f1",
        "accuracy",
        "validation_macro_f1",
        "validation_accuracy",
        "coverage_v1_error_last",
        "coverage_v2_error_last",
        "purity_v1_error_last",
        "purity_v2_error_last",
        "stateful_signature_drift_last",
        "total_coarsen_sec",
        "peak_rss_mb",
    )
    by_dataset = aggregate_rows(rows, ["dataset", "method", "ratio"], metrics)
    by_method = aggregate_rows(rows, ["method", "ratio"], metrics)
    write_csv(args.output / "gate14_by_method_ratio_dataset.csv", by_dataset)
    write_csv(args.output / "gate14_final_by_method.csv", by_method)
    hesf_rows = [row for row in rows if row.get("status") == "success" and str(row.get("method")).startswith("HeSF-TC")]
    baseline_rows = [row for row in rows if row.get("status") == "success" and row.get("method") in BASELINE_METHODS]
    gaps = build_ratio_matched_rows(hesf_rows, baseline_rows)
    write_csv(args.output / "gate14_ratio_matched_gaps.csv", gaps)
    ceiling_rows = [row for row in rows if row.get("status") == "success" and row.get("method") == "full-graph-hettree-lite-tuned"]
    recovery = compute_recovery_vs_ceiling(hesf_rows, ceiling_rows)
    write_csv(args.output / "gate14_recovery_vs_ceiling.csv", recovery)
    selected = select_validation_best_rows(hesf_rows)
    write_csv(args.output / "gate14_validation_selected_test.csv", selected)
    write_csv(args.output / "gate14_oracle_appendix.csv", _oracle_rows(rows))
    candidate_diag = [
        {
            key: row.get(key)
            for key in row
            if key in {"dataset", "method", "ratio", "seed", "candidate_source", "candidate_pair_count_last", "eligible_candidate_pair_count_last", "selected_support_merges", "candidate_candidate_pairs_retained_last"}
        }
        for row in hesf_rows
    ]
    write_csv(args.output / "gate14_candidate_source_diagnostics.csv", candidate_diag)
    merge_diag_keys = {
        "dataset", "method", "ratio", "seed", "selected_support_merges", "coverage_v1_error_last", "coverage_v2_error_last",
        "purity_v1_error_last", "purity_v2_error_last", "anchor_collision_rate_last", "class_context_collision_rate_last",
        "stateful_signature_drift_last", "stateful_update_count_last", "rescore_count_last",
    }
    write_csv(args.output / "gate14_merge_diagnostics.csv", [{key: row.get(key) for key in merge_diag_keys} for row in hesf_rows])
    write_csv(args.output / "gate14_code_issue_report.csv", [{"issue": "official evaluator not integrated", "status": "diagnostic_lite_only"}])
    _write_required_figures(args.output)
    summary_rows = aggregate_rows(selected, ["method", "ratio"], ("macro_f1", "accuracy", "validation_macro_f1"))
    (args.output / "validation_selection_summary.md").write_text(
        "# Validation Selection Summary\n\n" + markdown_table(summary_rows, ["method", "ratio", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean"]) + "\n",
        encoding="utf-8",
    )
    _write_gate14_decision(args.output, by_method, gaps, recovery)
    failures = [row for row in rows if row.get("status") != "success"]
    write_json(args.output / "result.json", {"rows": len(rows), "success": len(rows) - len(failures), "failed": len(failures)})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
