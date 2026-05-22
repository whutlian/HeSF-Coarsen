from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_gate17 import _bool, _float, _mean, _no_test_leakage, _success, read_csv, validation_selected


GATE17_4_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
STRONG_BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only"}
RANK_ELIGIBLE_METHODS = {
    "HeSF-SS-real-validation-neutral-fill",
    "HeSF-SS-real-occlusion-neutral-fill",
    "HeSF-SS-H6-cluster-validation-neutral-fill",
    "HeSF-SS-H6-cluster-occlusion-neutral-fill",
    "HeSF-SS-lossy-prototype-fixed-saturation",
}
DIAGNOSTIC_METHODS = {
    "HeSF-SS-real-validation-no-fallback",
    "HeSF-SS-real-occlusion-selection-only",
    "HeSF-SS-full-residual-prototype-upperbound",
    "HeSF-SS-H6-selected-set-control",
    "HeSF-SS-H6-equivalence-control",
}
DECISION_RATIOS = {0.30, 0.70}


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _find_raw_rows(input_dir: Path) -> list[dict[str, Any]]:
    _raise_csv_field_limit()
    for path in [input_dir / "gate17_4_raw_rows.csv", input_dir / "main" / "gate17_4_raw_rows.csv"]:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def _is_decision_ratio(row: Mapping[str, Any]) -> bool:
    return any(abs(_float(row.get("requested_support_ratio")) - ratio) <= 1.0e-9 for ratio in DECISION_RATIOS)


def _row_no_leakage(row: Mapping[str, Any]) -> bool:
    return not _bool(row.get("selector_uses_test_labels")) and not _bool(row.get("teacher_uses_test_labels_for_training")) and _bool(row.get("no_test_leakage", True))


def _eligible(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    return bool(
        _success(row)
        and method in RANK_ELIGIBLE_METHODS
        and _bool(row.get("eligible_for_main_decision", False))
        and _bool(row.get("node_budget_exact_match", False))
        and _bool(row.get("represented_context_exact_or_bounded", False))
        and not _bool(row.get("full_residual_upperbound", False))
        and _row_no_leakage(row)
        and str(row.get("primary_eval_mode", "")) == "compressed_projected"
    )


def _best_strong_baseline(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        item
        for item in rows
        if _success(item)
        and str(item.get("method")) in STRONG_BASELINES
        and str(item.get("dataset")) == str(row.get("dataset"))
        and str(item.get("seed")) == str(row.get("seed"))
        and abs(_float(item.get("requested_support_ratio")) - _float(row.get("requested_support_ratio"))) <= 1.0e-9
        and _bool(item.get("node_budget_exact_match", item.get("support_budget_exact_match", True)))
    ]
    return max(candidates, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))), default=None)


def exact_budget_paired_gaps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _eligible(row):
            continue
        baseline = _best_strong_baseline(row, rows)
        if baseline is None:
            continue
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "requested_support_ratio": row.get("requested_support_ratio"),
                "method": row.get("method"),
                "best_baseline_method": baseline.get("method"),
                "method_macro_f1": _float(row.get("macro_f1")),
                "baseline_macro_f1": _float(baseline.get("macro_f1")),
                "delta_macro_f1": float(round(_float(row.get("macro_f1")) - _float(baseline.get("macro_f1")), 12)),
                "method_accuracy": _float(row.get("accuracy")),
                "baseline_accuracy": _float(baseline.get("accuracy")),
                "delta_accuracy": float(round(_float(row.get("accuracy")) - _float(baseline.get("accuracy")), 12)),
                "decision_ratio": bool(_is_decision_ratio(row)),
            }
        )
    return sorted(out, key=lambda item: (str(item.get("dataset")), str(item.get("method")), _float(item.get("requested_support_ratio"))))


def _group_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(round(_mean(_float(row.get(key)) for row in rows), 12))


def _best_method(gaps: Sequence[Mapping[str, Any]]) -> str | None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gaps:
        if bool(row.get("decision_ratio")):
            groups[str(row.get("method"))].append(row)
    if not groups:
        return None
    return max(
        groups,
        key=lambda method: (
            _mean(_float(row.get("delta_macro_f1")) for row in groups[method]),
            _mean(_float(row.get("delta_accuracy")) for row in groups[method]),
            method,
        ),
    )


def _aggregate_selected_by_method(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row.get("method"))].append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "runs": int(len(group)),
                "macro_f1_mean": _mean(_float(row.get("macro_f1")) for row in group),
                "accuracy_mean": _mean(_float(row.get("accuracy")) for row in group),
                "validation_macro_f1_mean": _mean(_float(row.get("validation_macro_f1")) for row in group),
                "eligible_for_main_decision_mean": _mean(1.0 if _eligible(row) else 0.0 for row in group),
            }
        )
    return sorted(out, key=lambda row: (-float(row["macro_f1_mean"]), -float(row["accuracy_mean"]), str(row["method"])))


def _acm_saturated(rows: Sequence[Mapping[str, Any]]) -> bool:
    acm = [row for row in rows if str(row.get("dataset", "")).upper() == "ACM" and str(row.get("method", "")).startswith("HeSF-SS")]
    return bool(acm) and all(_bool(row.get("candidate_full_equivalent", row.get("candidate_allclose_to_full", row.get("allclose_to_full")))) for row in acm if _float(row.get("requested_support_ratio")) <= 0.10)


def _h6_equivalence_summary(input_dir: Path) -> dict[str, Any]:
    rows = read_csv(input_dir / "diagnostics" / "gate17_4_h6_equivalence.csv")
    construction = [row for row in rows if str(row.get("mode")) == "construction"]
    selected = [row for row in rows if str(row.get("mode")) == "selected_set"]
    if not construction:
        return {
            "h6_equivalence_control_pass": False,
            "h6_equivalence_macro_gap_max": None,
            "h6_equivalence_tree_l2_max": None,
            "h6_construction_gap_detected": None,
            "h6_selected_set_control_reported": bool(selected),
        }
    pass_all = all(_bool(row.get("construction_equivalence_pass")) for row in construction)
    return {
        "h6_equivalence_control_pass": bool(pass_all),
        "h6_equivalence_macro_gap_max": float(round(max(abs(_float(row.get("macro_gap_vs_h6"))) for row in construction), 12)),
        "h6_equivalence_tree_l2_max": float(round(max(_float(row.get("tree_l2_delta_vs_h6")) for row in construction), 12)),
        "h6_construction_gap_detected": bool(not pass_all),
        "h6_selected_set_control_reported": bool(selected),
    }


def _write_reports(output_dir: Path, result: Mapping[str, Any], selected_by_method: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> None:
    decision_lines = [
        "# Gate17.4 Decision",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Decision Reasons",
        "",
        *([f"- `{reason}`" for reason in result.get("main_failure_reasons", [])] or ["- none"]),
    ]
    (output_dir / "gate17_4_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    report_lines = [
        "# Gate17.4 Final Report",
        "",
        "## 1. Executive decision",
        "",
        f"- decision: `{result['decision']}`",
        f"- best_eligible_method: `{result.get('best_eligible_method')}`",
        f"- gate18_allowed: `{result.get('gate18_allowed')}`",
        "",
        "## 2. Gate17.3 recap and why Gate17.4 was needed",
        "",
        "Gate17.4 fixes best-eligible DBLP gap reporting and separates H6 selected-set overlap from H6 construction equivalence.",
        "",
        "## 3. Code audit summary",
        "",
        "See `outputs/gate17_4_code_audit/` for the pre-implementation audit tables.",
        "",
        "## 4. H6-equivalence control results",
        "",
        f"- h6_equivalence_control_pass: `{result.get('h6_equivalence_control_pass')}`",
        f"- h6_equivalence_macro_gap_max: `{result.get('h6_equivalence_macro_gap_max')}`",
        f"- h6_equivalence_tree_l2_max: `{result.get('h6_equivalence_tree_l2_max')}`",
        "",
        "## 5. Best eligible real-feedback results",
        "",
        f"- overall exact delta macro: `{result.get('best_eligible_overall_exact_delta_macro')}`",
        f"- overall exact delta accuracy: `{result.get('best_eligible_overall_exact_delta_accuracy')}`",
        f"- DBLP exact delta macro: `{result.get('best_eligible_dblp_exact_delta_macro')}`",
        "",
        "## 6. DBLP exact-budget comparison",
        "",
        f"- DBLP ratio 0.3 gap: `{result.get('best_eligible_dblp_ratio_0_3_gap')}`",
        f"- DBLP ratio 0.7 gap: `{result.get('best_eligible_dblp_ratio_0_7_gap')}`",
        "",
        "## 7. ACM saturation note",
        "",
        "ACM support saturation is reported as sanity evidence only and is not used as success evidence.",
        "",
        "## 8. IMDB diagnostic note",
        "",
        f"- IMDB exact delta macro: `{result.get('best_eligible_imdb_exact_delta_macro')}`",
        "",
        "## 9. Budget and leakage checks",
        "",
        f"- no_test_leakage: `{result.get('no_test_leakage')}`",
        f"- candidate_node_budget_pass: `{result.get('candidate_node_budget_pass')}`",
        f"- candidate_represented_context_budget_pass: `{result.get('candidate_represented_context_budget_pass')}`",
        "",
        "## 10. Feedback signal checks",
        "",
        f"- real_feedback_signal_positive: `{result.get('real_feedback_signal_positive')}`",
        "",
        "## 11. Prototype diagnostics if any",
        "",
        f"- prototype_diagnostic_only: `{result.get('prototype_diagnostic_only')}`",
        "",
        "## 12. Decision against Gate18 / for Gate18",
        "",
        "Gate18 is allowed only when the strict Gate17.4 decision rules all pass.",
        "",
        "## 13. Next recommended action",
        "",
        "If H6 construction equivalence fails, fix graph construction/aggregation. If it passes but DBLP gap remains, continue H6-cluster feedback diagnostics.",
        "",
        "## Validation-selected methods",
        "",
        markdown_table(selected_by_method, ["method", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean", "eligible_for_main_decision_mean"]),
        "",
        "## Exact-budget paired gaps",
        "",
        markdown_table(gaps[:40], ["dataset", "seed", "method", "best_baseline_method", "requested_support_ratio", "delta_macro_f1", "delta_accuracy"]),
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _find_raw_rows(input_dir)
    selected = validation_selected([dict(row) for row in rows])
    selected_by_method = _aggregate_selected_by_method(selected)
    gaps = exact_budget_paired_gaps([dict(row) for row in rows])
    write_csv(output_dir / "gate17_4_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_4_by_dataset_selected.csv", selected)
    write_csv(output_dir / "gate17_4_exact_budget_paired_gaps.csv", gaps)

    best_method = _best_method(gaps)
    best_gaps = [row for row in gaps if str(row.get("method")) == str(best_method) and bool(row.get("decision_ratio"))]
    best_rows = [row for row in rows if str(row.get("method")) == str(best_method) and _eligible(row) and _is_decision_ratio(row)]
    dblp_best = [row for row in best_gaps if str(row.get("dataset", "")).upper() == "DBLP"]
    acm_best = [row for row in best_gaps if str(row.get("dataset", "")).upper() == "ACM"]
    imdb_best = [row for row in best_gaps if str(row.get("dataset", "")).upper() == "IMDB"]
    ratio_gap = {
        f"{_float(row.get('requested_support_ratio')):.1f}": _float(row.get("delta_macro_f1"))
        for row in dblp_best
    }
    h6 = _h6_equivalence_summary(input_dir)
    no_leakage = bool(_no_test_leakage(rows) and all(_row_no_leakage(row) for row in rows if str(row.get("method", "")).startswith("HeSF-SS")))
    candidate_rows = [row for row in rows if _eligible(row)]
    candidate_node_budget_pass = bool(candidate_rows) and all(_bool(row.get("node_budget_exact_match")) for row in candidate_rows)
    candidate_context_pass = bool(candidate_rows) and all(_bool(row.get("represented_context_exact_or_bounded")) for row in candidate_rows)
    feedback_rows = [row for row in rows if str(row.get("dataset", "")).upper() == "DBLP" and str(row.get("method", "")).startswith("HeSF-SS")]
    feedback_signal = any(_bool(row.get("validation_signal_pass")) or _bool(row.get("occlusion_task_signal_pass")) for row in feedback_rows)
    overall_macro = _group_mean(best_gaps, "delta_macro_f1")
    overall_accuracy = _group_mean(best_gaps, "delta_accuracy")
    dblp_macro = _group_mean(dblp_best, "delta_macro_f1")
    main_reasons: list[str] = []
    main_reasons.append("PASS_H6_EQUIVALENCE_CONTROL" if h6["h6_equivalence_control_pass"] else "FAIL_H6_EQUIVALENCE_CONTROL")
    if best_method is None:
        main_reasons.append("FAIL_BEST_ELIGIBLE_BUDGET")
    else:
        main_reasons.append("PASS_BEST_ELIGIBLE_BUDGET")
    main_reasons.append("PASS_CANDIDATE_REPRESENTED_CONTEXT_BUDGET" if candidate_context_pass else "FAIL_CANDIDATE_REPRESENTED_CONTEXT_BUDGET")
    main_reasons.append("ACM_SUPPORT_SATURATED_NOT_USED_FOR_RANKING")
    if any("prototype" in str(row.get("method", "")).lower() for row in rows):
        main_reasons.append("PROTOTYPE_DIAGNOSTIC_ONLY_DUE_TO_SATURATION")
    db_gap_ok = bool(
        best_method is not None
        and ratio_gap.get("0.3") is not None
        and ratio_gap.get("0.7") is not None
        and ratio_gap.get("0.3", -1.0) >= 0.0
        and ratio_gap.get("0.7", -1.0) >= 0.0
    )
    main_reasons.append("PASS_DBLP_REAL_FEEDBACK_NEAR_OR_BEATS_H6" if db_gap_ok else "FAIL_DBLP_REAL_FEEDBACK_GAP")
    gate18 = bool(
        h6["h6_equivalence_control_pass"]
        and db_gap_ok
        and overall_macro is not None
        and overall_macro > 0.0
        and overall_accuracy is not None
        and overall_accuracy >= 0.0
        and no_leakage
        and candidate_node_budget_pass
        and candidate_context_pass
        and best_method != "HeSF-SS-full-residual-prototype-upperbound"
    )
    if gate18:
        decision = "PASS_GATE17_4_READY_FOR_GATE18"
    elif not h6["h6_equivalence_control_pass"]:
        decision = "FAIL_H6_EQUIVALENCE_CONTROL"
    else:
        decision = "CONTINUE_GATE17_X_H6_EQUIVALENCE_READY"
    result: dict[str, Any] = {
        "stage": "Gate17.4",
        "decision": decision,
        "gate18_allowed": bool(gate18),
        "best_eligible_method": best_method,
        "best_eligible_overall_exact_delta_macro": overall_macro,
        "best_eligible_overall_exact_delta_accuracy": overall_accuracy,
        "best_eligible_dblp_exact_delta_macro": dblp_macro,
        "best_eligible_dblp_ratio_0_3_gap": ratio_gap.get("0.3"),
        "best_eligible_dblp_ratio_0_7_gap": ratio_gap.get("0.7"),
        "best_eligible_imdb_exact_delta_macro": _group_mean(imdb_best, "delta_macro_f1"),
        "best_eligible_acm_exact_delta_macro": _group_mean(acm_best, "delta_macro_f1"),
        "best_eligible_validation_macro_f1_mean": _group_mean(best_rows, "validation_macro_f1"),
        "best_eligible_test_macro_f1_mean": _group_mean(best_rows, "macro_f1"),
        **h6,
        "acm_support_saturated": _acm_saturated(rows),
        "dblp_primary_decision_dataset": True,
        "imdb_weak_evaluator_diagnostic": True,
        "acm_support_saturation_sanity": True,
        "no_test_leakage": bool(no_leakage),
        "primary_eval_mode": "compressed_projected",
        "eligible_method_count": int(len({str(row.get("method")) for row in candidate_rows if _is_decision_ratio(row)})),
        "main_failure_reasons": main_reasons,
        "candidate_node_budget_pass": bool(candidate_node_budget_pass),
        "candidate_represented_context_budget_pass": bool(candidate_context_pass),
        "real_feedback_signal_positive": bool(feedback_signal),
        "prototype_diagnostic_only": True,
        "typedhash_note": "TypedHash skipped in Gate17.4 for speed; H6/flatten used as strong baselines.",
        "rows": int(len(rows)),
        "success": int(sum(1 for row in rows if _success(row))),
        "failed": int(sum(1 for row in rows if not _success(row))),
        "dataset_seed_map": dict(GATE17_4_SINGLE_SEED_BY_DATASET),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_reports(output_dir, result, selected_by_method, gaps)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17.4 H6 equivalence outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17_4_h6_equivalence"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
