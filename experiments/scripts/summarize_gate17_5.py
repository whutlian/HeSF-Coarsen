from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_gate17 import (
    _bool,
    _float,
    _mean,
    _no_test_leakage,
    _success,
    assert_dataset_integrity,
    normalize_header,
    read_csv,
    validation_selected,
)


GATE17_5_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
STRONG_BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only"}
DIAGNOSTIC_METHODS = {
    "HeSF-SS-real-occlusion-neutral-fill",
    "HeSF-SS-H6-equivalence-control",
    "HeSF-SS-H6-selected-set-control",
    "HeSF-SS-full-residual-prototype-upperbound",
    "HeSF-SS-lossy-prototype-fixed-saturation",
}
DECISION_RATIOS = {0.30, 0.70}


def _find_raw_rows(input_dir: Path) -> list[dict[str, Any]]:
    for path in [input_dir / "gate17_5_raw_rows.csv", input_dir / "main" / "gate17_5_raw_rows.csv"]:
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
        and method.startswith("HeSF-SS")
        and method not in DIAGNOSTIC_METHODS
        and _bool(row.get("eligible_for_main_decision", False))
        and str(row.get("primary_eval_mode", "")) == "compressed_projected"
        and _bool(row.get("node_budget_exact_match", False))
        and _bool(row.get("represented_context_exact_or_bounded", False))
        and _row_no_leakage(row)
        and not _bool(row.get("full_residual_upperbound", False))
        and not _bool(row.get("diagnostic_only", False))
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


def _h6_equivalence_summary(input_dir: Path) -> dict[str, Any]:
    rows = read_csv(input_dir / "diagnostics" / "gate17_5_h6_equivalence.csv")
    construction = [row for row in rows if str(row.get("mode")) == "construction"]
    if not construction:
        return {"h6_equivalence_control_pass": False, "h6_equivalence_macro_gap_max": None, "h6_equivalence_tree_l2_max": None}
    return {
        "h6_equivalence_control_pass": bool(all(_bool(row.get("construction_equivalence_pass")) for row in construction)),
        "h6_equivalence_macro_gap_max": float(round(max(abs(_float(row.get("macro_gap_vs_h6"))) for row in construction), 12)),
        "h6_equivalence_tree_l2_max": float(round(max(_float(row.get("tree_l2_delta_vs_h6")) for row in construction), 12)),
    }


def _acm_saturated(rows: Sequence[Mapping[str, Any]]) -> bool:
    acm = [row for row in rows if str(row.get("dataset", "")).upper() == "ACM" and str(row.get("method", "")).startswith("HeSF-SS")]
    return bool(acm) and any(_bool(row.get("candidate_full_equivalent", row.get("allclose_to_full"))) for row in acm)


def _raw_header_name(input_dir: Path) -> str:
    for path in [input_dir / "gate17_5_raw_rows.csv", input_dir / "main" / "gate17_5_raw_rows.csv"]:
        if path.exists():
            first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            return first.split(",", 1)[0]
    return ""


def _write_header_check(input_dir: Path, output_dir: Path, rows: Sequence[dict[str, Any]], gaps: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> bool:
    raw = _raw_header_name(input_dir)
    groups = {(str(row.get("dataset")), str(row.get("seed")), str(row.get("method"))) for row in rows if _success(row)}
    exact_nulls = sum(1 for row in gaps if not str(row.get("dataset", "")).strip())
    unique = sorted({str(row.get("dataset")) for row in rows})
    passed = bool("dataset" in rows[0] and exact_nulls == 0 and len(selected) == len(groups))
    write_csv(
        output_dir / "diagnostics" / "gate17_5_header_normalization_check.csv",
        [
            {
                "raw_header_name": raw,
                "normalized_header_name": normalize_header(raw),
                "dataset_key_present": "dataset" in rows[0],
                "unique_datasets": ",".join(unique),
                "validation_selected_expected_rows": int(len(groups)),
                "validation_selected_actual_rows": int(len(selected)),
                "exact_gap_dataset_null_count": int(exact_nulls),
                "pass": bool(passed),
            }
        ],
    )
    return passed


def _write_reports(output_dir: Path, result: Mapping[str, Any], selected_by_method: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> None:
    decision_lines = [
        "# Gate17.5 Decision",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Decision Reasons",
        "",
        *([f"- `{reason}`" for reason in result.get("main_failure_reasons", [])] or ["- none"]),
    ]
    (output_dir / "gate17_5_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    report_lines = [
        "# Gate17.5 Final Report",
        "",
        "## Executive Decision",
        "",
        f"- decision: `{result['decision']}`",
        f"- gate18_allowed: `{result['gate18_allowed']}`",
        f"- best_eligible_method: `{result.get('best_eligible_method')}`",
        "",
        "## Corrected Summary And Header Check",
        "",
        f"- header_normalization_pass: `{result.get('header_normalization_pass')}`",
        f"- primary_eval_mode: `{result.get('primary_eval_mode')}`",
        "",
        "## DBLP Exact-Budget Gaps",
        "",
        f"- ratio 0.30 macro gap: `{result.get('best_eligible_dblp_ratio_0_3_gap_macro')}`",
        f"- ratio 0.70 macro gap: `{result.get('best_eligible_dblp_ratio_0_7_gap_macro')}`",
        f"- missing exact ratios: `{result.get('best_eligible_dblp_missing_exact_ratios')}`",
        "",
        "## H6 Equivalence",
        "",
        f"- h6_equivalence_control_pass: `{result.get('h6_equivalence_control_pass')}`",
        "",
        "## ACM / IMDB Roles",
        "",
        f"- acm_support_saturated: `{result.get('acm_support_saturated')}`",
        f"- acm_used_for_success_evidence: `{result.get('acm_used_for_success_evidence')}`",
        f"- imdb_weak_evaluator_diagnostic: `{result.get('imdb_weak_evaluator_diagnostic')}`",
        "",
        "## Validation-Selected Methods",
        "",
        markdown_table(selected_by_method, ["method", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean", "eligible_for_main_decision_mean"]),
        "",
        "## Exact-Budget Paired Gaps",
        "",
        markdown_table(gaps[:40], ["dataset", "seed", "method", "best_baseline_method", "requested_support_ratio", "delta_macro_f1", "delta_accuracy"]),
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    rows = _find_raw_rows(input_dir)
    assert_dataset_integrity(rows, expected=set(GATE17_5_SINGLE_SEED_BY_DATASET))
    selected = validation_selected([dict(row) for row in rows])
    selected_by_method = _aggregate_selected_by_method(selected)
    gaps = exact_budget_paired_gaps([dict(row) for row in rows])
    write_csv(output_dir / "gate17_5_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_5_by_dataset_selected.csv", selected)
    write_csv(output_dir / "gate17_5_exact_budget_paired_gaps.csv", gaps)

    header_pass = _write_header_check(input_dir, output_dir, rows, gaps, selected)
    best_method = _best_method(gaps)
    best_gaps = [row for row in gaps if str(row.get("method")) == str(best_method) and bool(row.get("decision_ratio"))]
    best_rows = [row for row in rows if str(row.get("method")) == str(best_method) and _eligible(row) and _is_decision_ratio(row)]
    dblp_best = [row for row in best_gaps if str(row.get("dataset", "")).upper() == "DBLP"]
    dblp_by_ratio = {f"{_float(row.get('requested_support_ratio')):.1f}": row for row in dblp_best}
    missing = [ratio for ratio in sorted(DECISION_RATIOS) if f"{ratio:.1f}" not in dblp_by_ratio]
    h6 = _h6_equivalence_summary(input_dir)
    no_leakage = bool(_no_test_leakage(rows) and all(_row_no_leakage(row) for row in rows if str(row.get("method", "")).startswith("HeSF-SS")))
    overall_macro = _group_mean(best_gaps, "delta_macro_f1")
    overall_accuracy = _group_mean(best_gaps, "delta_accuracy")
    dblp_accuracy_gaps = [_float(row.get("delta_accuracy")) for row in dblp_best]
    dblp_accuracy_close = bool(dblp_accuracy_gaps and min(dblp_accuracy_gaps) >= -0.01)
    reasons: list[str] = []
    reasons.append("HEADER_NORMALIZATION_PASS" if header_pass else "HEADER_NORMALIZATION_FAIL")
    reasons.append("H6_CONSTRUCTION_EQUIVALENCE_PASS" if h6["h6_equivalence_control_pass"] else "H6_CONSTRUCTION_EQUIVALENCE_FAIL")
    if best_method:
        reasons.append("REAL_VALIDATION_NEUTRAL_FILL_LEADS" if best_method == "HeSF-SS-real-validation-neutral-fill" else f"BEST_ELIGIBLE_{best_method}")
    if dblp_by_ratio.get("0.3") and _float(dblp_by_ratio["0.3"].get("delta_macro_f1")) >= 0.0:
        reasons.append("DBLP_0_3_MACRO_PASS")
    if 0.70 in missing:
        reasons.append("DBLP_0_7_EXACT_BUDGET_MISSING")
    elif dblp_by_ratio.get("0.7") and _float(dblp_by_ratio["0.7"].get("delta_macro_f1")) >= 0.0:
        reasons.append("DBLP_0_7_MACRO_PASS")
    if overall_accuracy is not None and overall_accuracy < 0.0:
        reasons.append("ACCURACY_DELTA_NEGATIVE")
    gate18 = bool(
        header_pass
        and h6["h6_equivalence_control_pass"]
        and best_method is not None
        and best_method not in DIAGNOSTIC_METHODS
        and not missing
        and _float(dblp_by_ratio.get("0.3", {}).get("delta_macro_f1"), -1.0) >= 0.0
        and _float(dblp_by_ratio.get("0.7", {}).get("delta_macro_f1"), -1.0) >= 0.0
        and overall_macro is not None
        and overall_macro > 0.0
        and (overall_accuracy is not None and (overall_accuracy >= 0.0 or dblp_accuracy_close))
        and no_leakage
    )
    if gate18:
        decision = "PASS_GATE17_5_READY_FOR_GATE18"
    elif missing:
        decision = "CONTINUE_GATE17_5_SUMMARY_FIXED"
    else:
        decision = "CONTINUE_GATE17_X_H6_CLUSTER_GATING"
    if not gate18:
        reasons.append("GATE18_BLOCKED")
    result: dict[str, Any] = {
        "stage": "Gate17.5",
        "decision": decision,
        "gate18_allowed": bool(gate18),
        "header_normalization_pass": bool(header_pass),
        "best_eligible_method": best_method,
        "best_eligible_overall_exact_delta_macro": overall_macro,
        "best_eligible_overall_exact_delta_accuracy": overall_accuracy,
        "best_eligible_dblp_ratio_0_3_gap_macro": None if "0.3" not in dblp_by_ratio else _float(dblp_by_ratio["0.3"].get("delta_macro_f1")),
        "best_eligible_dblp_ratio_0_3_gap_accuracy": None if "0.3" not in dblp_by_ratio else _float(dblp_by_ratio["0.3"].get("delta_accuracy")),
        "best_eligible_dblp_ratio_0_7_gap_macro": None if "0.7" not in dblp_by_ratio else _float(dblp_by_ratio["0.7"].get("delta_macro_f1")),
        "best_eligible_dblp_ratio_0_7_gap_accuracy": None if "0.7" not in dblp_by_ratio else _float(dblp_by_ratio["0.7"].get("delta_accuracy")),
        "best_eligible_dblp_exact_rows": int(len(dblp_best)),
        "best_eligible_dblp_missing_exact_ratios": [float(value) for value in missing],
        "best_eligible_validation_macro_f1_mean": _group_mean(best_rows, "validation_macro_f1"),
        "best_eligible_test_macro_f1_mean": _group_mean(best_rows, "macro_f1"),
        **h6,
        "acm_support_saturated": _acm_saturated(rows),
        "acm_used_for_success_evidence": False,
        "dblp_primary_decision_dataset": True,
        "imdb_weak_evaluator_diagnostic": True,
        "no_test_leakage": bool(no_leakage),
        "primary_eval_mode": "compressed_projected",
        "eligible_method_count": int(len({str(row.get("method")) for row in rows if _eligible(row)})),
        "main_failure_reasons": reasons,
        "typedhash_note": "TypedHash skipped for Gate17.5 speed; Gate18 requires TypedHash.",
        "rows": int(len(rows)),
        "success": int(sum(1 for row in rows if _success(row))),
        "failed": int(sum(1 for row in rows if not _success(row))),
        "dataset_seed_map": dict(GATE17_5_SINGLE_SEED_BY_DATASET),
    }
    (output_dir / "gate17_5_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_reports(output_dir, result, selected_by_method, gaps)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17.5 H6 cluster gating outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17_5_h6_cluster_gating"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
