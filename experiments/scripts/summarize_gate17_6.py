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


GATE17_6_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
STRONG_BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only"}
DIAGNOSTIC_METHODS = {
    "HeSF-SS-real-occlusion-neutral-fill",
    "HeSF-SS-H6-cluster-validation-coverage-gated",
    "HeSF-SS-H6-equivalence-control",
    "HeSF-SS-H6-selected-set-control",
    "HeSF-SS-full-residual-prototype-upperbound",
}
H6_FILL_ONLY_METHOD = "HeSF-SS-H6-fill-only"
DECISION_RATIOS = (0.30, 0.70)


def _find_raw_rows(input_dir: Path) -> list[dict[str, Any]]:
    for path in [input_dir / "gate17_6_raw_rows.csv", input_dir / "main" / "gate17_6_raw_rows.csv"]:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def _ratio(row: Mapping[str, Any]) -> float:
    return _float(row.get("requested_support_ratio", row.get("support_ratio")))


def _same_ratio(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(_ratio(left) - _ratio(right)) <= 1.0e-9


def _is_decision_ratio(row: Mapping[str, Any]) -> bool:
    value = _ratio(row)
    return any(abs(value - ratio) <= 1.0e-9 for ratio in DECISION_RATIOS)


def _row_no_leakage(row: Mapping[str, Any]) -> bool:
    return (
        not _bool(row.get("selector_uses_test_labels"))
        and not _bool(row.get("teacher_uses_test_labels_for_training"))
        and _bool(row.get("no_test_leakage", True))
    )


def _bool_default(value: Any, default: bool) -> bool:
    if value in {"", None}:
        return bool(default)
    return _bool(value)


def _typedhash_included(rows: Sequence[Mapping[str, Any]]) -> bool:
    typed = [row for row in rows if _success(row) and str(row.get("method")) == "TypedHash-ChebHeat-support-only"]
    ratios = {round(_ratio(row), 2) for row in typed}
    return bool({0.30, 0.70}.issubset(ratios))


def _eligible(row: Mapping[str, Any], *, typedhash_included: bool) -> bool:
    method = str(row.get("method", ""))
    exact = _bool(row.get("node_budget_exact_match", row.get("support_budget_exact_match", False)))
    effective_pass = _bool_default(row.get("effective_support_node_budget_pass"), exact)
    represented_pass = _bool_default(row.get("represented_context_budget_pass"), _bool(row.get("represented_context_exact_or_bounded", False)))
    return bool(
        _success(row)
        and typedhash_included
        and method.startswith("HeSF-SS-")
        and method not in DIAGNOSTIC_METHODS
        and method != H6_FILL_ONLY_METHOD
        and method != "HeSF-SS-full-residual-prototype-upperbound"
        and _bool(row.get("eligible_for_main_decision", False))
        and str(row.get("primary_eval_mode", "")) == "compressed_projected"
        and exact
        and effective_pass
        and represented_pass
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
        and _same_ratio(item, row)
        and _bool(item.get("node_budget_exact_match", item.get("support_budget_exact_match", True)))
    ]
    return max(candidates, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))), default=None)


def best_strong_baseline_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(str(row.get("dataset")), str(row.get("seed")), _ratio(row)) for row in rows if _success(row)})
    out: list[dict[str, Any]] = []
    for dataset, seed, ratio in keys:
        bucket = [
            row
            for row in rows
            if _success(row)
            and str(row.get("method")) in STRONG_BASELINES
            and str(row.get("dataset")) == dataset
            and str(row.get("seed")) == seed
            and abs(_ratio(row) - ratio) <= 1.0e-9
        ]
        best = max(bucket, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))), default=None)
        if best is not None:
            out.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "requested_support_ratio": ratio,
                    "best_strong_baseline_method": best.get("method"),
                    "best_strong_baseline_macro_f1": _float(best.get("macro_f1")),
                    "best_strong_baseline_accuracy": _float(best.get("accuracy")),
                    "typedhash_is_best_strong": str(best.get("method")) == "TypedHash-ChebHeat-support-only",
                }
            )
    return out


def exact_budget_paired_gaps(rows: Sequence[dict[str, Any]], *, typedhash_included: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _eligible(row, typedhash_included=typedhash_included):
            continue
        baseline = _best_strong_baseline(row, rows)
        if baseline is None:
            continue
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "requested_support_ratio": row.get("requested_support_ratio", row.get("support_ratio", "")),
                "requested_support_count": row.get("requested_support_count", ""),
                "method": row.get("method"),
                "best_baseline_method": baseline.get("method"),
                "method_macro_f1": _float(row.get("macro_f1")),
                "baseline_macro_f1": _float(baseline.get("macro_f1")),
                "delta_macro_f1": float(round(_float(row.get("macro_f1")) - _float(baseline.get("macro_f1")), 12)),
                "method_accuracy": _float(row.get("accuracy")),
                "baseline_accuracy": _float(baseline.get("accuracy")),
                "delta_accuracy": float(round(_float(row.get("accuracy")) - _float(baseline.get("accuracy")), 12)),
                "validation_macro_f1": _float(row.get("validation_macro_f1")),
                "validation_accuracy": _float(row.get("validation_accuracy")),
                "primary_eval_mode": row.get("primary_eval_mode", ""),
                "decision_ratio": bool(_is_decision_ratio(row)),
            }
        )
    return sorted(out, key=lambda item: (str(item.get("dataset")), str(item.get("method")), _float(item.get("requested_support_ratio"))))


def _group_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(round(_mean(_float(row.get(key)) for row in rows), 12))


def _method_groups(gaps: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gaps:
        if bool(row.get("decision_ratio")):
            groups[str(row.get("method"))].append(row)
    return groups


def _best_method(gaps: Sequence[Mapping[str, Any]]) -> str | None:
    groups = _method_groups(gaps)
    if not groups:
        return None

    def key(method: str) -> tuple[bool, bool, float, float, float, str]:
        group = groups[method]
        dblp = [row for row in group if str(row.get("dataset", "")).upper() == "DBLP"]
        by_ratio = {f"{_float(row.get('requested_support_ratio')):.1f}": row for row in dblp}
        return (
            _float(by_ratio.get("0.3", {}).get("delta_macro_f1"), -1.0) >= 0.0,
            _float(by_ratio.get("0.7", {}).get("delta_macro_f1"), -1.0) >= 0.0,
            _mean(_float(row.get("delta_accuracy")) for row in dblp) if dblp else -1.0e9,
            _mean(_float(row.get("delta_macro_f1")) for row in group),
            _mean(_float(row.get("validation_macro_f1")) for row in group),
            method,
        )

    return max(groups, key=key)


def _aggregate_selected_by_method(selected: Sequence[Mapping[str, Any]], *, typedhash_included: bool) -> list[dict[str, Any]]:
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
                "eligible_for_main_decision_mean": _mean(1.0 if _eligible(row, typedhash_included=typedhash_included) else 0.0 for row in group),
            }
        )
    return sorted(out, key=lambda row: (-float(row["macro_f1_mean"]), -float(row["accuracy_mean"]), str(row["method"])))


def _acm_saturated(rows: Sequence[Mapping[str, Any]]) -> bool:
    acm = [row for row in rows if str(row.get("dataset", "")).upper() == "ACM" and str(row.get("method", "")).startswith("HeSF-SS")]
    return bool(acm) and any(_bool(row.get("candidate_full_equivalent", row.get("allclose_to_full", False))) for row in acm)


def _raw_header_name(input_dir: Path) -> str:
    for path in [input_dir / "gate17_6_raw_rows.csv", input_dir / "main" / "gate17_6_raw_rows.csv"]:
        if path.exists():
            first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            return first.split(",", 1)[0]
    return ""


def _write_header_check(
    input_dir: Path,
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> bool:
    raw = _raw_header_name(input_dir)
    groups = {(str(row.get("dataset")), str(row.get("seed")), str(row.get("method"))) for row in rows if _success(row)}
    exact_nulls = sum(1 for row in gaps if not str(row.get("dataset", "")).strip())
    unique = sorted({str(row.get("dataset")) for row in rows})
    passed = bool(rows and "dataset" in rows[0] and exact_nulls == 0 and len(selected) == len(groups))
    write_csv(
        output_dir / "diagnostics" / "gate17_6_header_normalization_check.csv",
        [
            {
                "raw_header_name": raw,
                "normalized_header_name": normalize_header(raw),
                "dataset_key_present": bool(rows and "dataset" in rows[0]),
                "unique_datasets": ",".join(unique),
                "validation_selected_expected_rows": int(len(groups)),
                "validation_selected_actual_rows": int(len(selected)),
                "exact_gap_dataset_null_count": int(exact_nulls),
                "pass": bool(passed),
            }
        ],
    )
    return passed


def _h6_fill_only_beats_validation(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if not _success(row) or str(row.get("method")) != H6_FILL_ONLY_METHOD:
            continue
        competitors = [
            item
            for item in rows
            if _success(item)
            and str(item.get("method", "")).startswith("HeSF-SS-validation-H6-fill")
            and str(item.get("dataset")) == str(row.get("dataset"))
            and str(item.get("seed")) == str(row.get("seed"))
            and _same_ratio(item, row)
        ]
        if competitors and _float(row.get("macro_f1")) > max(_float(item.get("macro_f1")) for item in competitors):
            return True
    return False


def _random_fill_beats_h6_fill(rows: Sequence[Mapping[str, Any]]) -> bool:
    for random_row in rows:
        if not _success(random_row) or str(random_row.get("method")) != "HeSF-SS-random-fill-after-validation":
            continue
        h6_rows = [
            row
            for row in rows
            if _success(row)
            and str(row.get("method")) == "HeSF-SS-validation-H6-fill"
            and str(row.get("dataset")) == str(random_row.get("dataset"))
            and str(row.get("seed")) == str(random_row.get("seed"))
            and _same_ratio(row, random_row)
        ]
        if h6_rows and _float(random_row.get("macro_f1")) > max(_float(row.get("macro_f1")) for row in h6_rows):
            return True
    return False


def _h6_fill_dependency(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _success(row)
        and str(row.get("method", "")).startswith("HeSF-SS-validation-H6-fill")
        and _float(row.get("h6_fill_support_count")) > 0.0
        for row in rows
    )


def _all_hesf_lose_to_typedhash_on_dblp(gaps: Sequence[Mapping[str, Any]]) -> bool:
    dblp = [row for row in gaps if str(row.get("dataset", "")).upper() == "DBLP"]
    if not dblp:
        return False
    typedhash_baseline_rows = [row for row in dblp if str(row.get("best_baseline_method")) == "TypedHash-ChebHeat-support-only"]
    return bool(typedhash_baseline_rows) and all(_float(row.get("delta_macro_f1")) < -0.03 for row in dblp)


def _calibration_destroyed_macro(rows: Sequence[Mapping[str, Any]], best_method: str | None) -> bool:
    if best_method is None:
        return False
    acc_rows = [row for row in rows if str(row.get("method", "")).startswith("HeSF-SS-validation-H6-fill-acc") and _success(row)]
    uncal = [row for row in rows if str(row.get("method")) == "HeSF-SS-validation-H6-fill" and _success(row)]
    return bool(acc_rows and uncal) and all(_float(row.get("macro_f1")) <= 0.0 for row in acc_rows) and _mean(_float(row.get("accuracy")) for row in uncal) < 0.0


def _write_class_shift_report(output_dir: Path, rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> None:
    per_class = read_csv(output_dir / "diagnostics" / "gate17_6_per_class_metrics.csv")
    best = str(result.get("best_eligible_method", ""))
    best_rows = [row for row in per_class if str(row.get("method")) == best]
    recall_up = sorted(
        [row for row in best_rows if _float(row.get("delta_recall_vs_best_strong")) > 0.0],
        key=lambda row: -_float(row.get("delta_recall_vs_best_strong")),
    )[:5]
    precision_down = sorted(
        [row for row in best_rows if _float(row.get("delta_precision_vs_best_strong")) < 0.0],
        key=lambda row: _float(row.get("delta_precision_vs_best_strong")),
    )[:5]
    lines = [
        "# Gate17.6 Class Shift Report",
        "",
        "1. Class recall increases driving macro-F1:",
        markdown_table(recall_up, ["dataset", "requested_support_ratio", "class_id", "delta_recall_vs_best_strong", "delta_f1_vs_best_strong"]) if recall_up else "No positive per-class recall deltas were available.",
        "",
        "2. Precision or majority accuracy drops driving accuracy gap:",
        markdown_table(precision_down, ["dataset", "requested_support_ratio", "class_id", "delta_precision_vs_best_strong", "delta_f1_vs_best_strong"]) if precision_down else "No negative precision deltas were available.",
        "",
        "3. Accuracy-calibrated variants:",
        f"Best eligible method: `{best}`; DBLP accuracy gaps: 0.30=`{result.get('best_eligible_dblp_ratio_0_3_gap_accuracy')}`, 0.70=`{result.get('best_eligible_dblp_ratio_0_7_gap_accuracy')}`.",
        "",
        "4. H6-fill-only distribution:",
        f"h6_fill_only_beats_validation_flag=`{result.get('h6_fill_only_beats_validation_flag')}`.",
        "",
        "5. Random fill versus H6 fill:",
        f"random_fill_beats_h6_fill_flag=`{result.get('random_fill_beats_h6_fill_flag')}`.",
    ]
    (output_dir / "diagnostics" / "gate17_6_class_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(output_dir: Path, result: Mapping[str, Any], selected_by_method: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> None:
    decision_lines = [
        "# Gate17.6 Decision",
        "",
        f"Decision: `{result['decision']}`",
        f"Gate18 allowed: `{result['gate18_allowed']}`",
        "",
        "## Blockers And Flags",
        "",
        *([f"- `{reason}`" for reason in result.get("main_failure_reasons", [])] or ["- none"]),
    ]
    (output_dir / "gate17_6_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    report_lines = [
        "# Gate17.6 Final Report",
        "",
        "## Executive Decision",
        "",
        f"- decision: `{result['decision']}`",
        f"- gate18_allowed: `{result['gate18_allowed']}`",
        f"- best_eligible_method: `{result.get('best_eligible_method')}`",
        f"- typedhash_included: `{result.get('typedhash_included')}`",
        "",
        "## DBLP Exact-Budget Gaps",
        "",
        f"- ratio 0.30 macro gap: `{result.get('best_eligible_dblp_ratio_0_3_gap_macro')}`",
        f"- ratio 0.30 accuracy gap: `{result.get('best_eligible_dblp_ratio_0_3_gap_accuracy')}`",
        f"- ratio 0.70 macro gap: `{result.get('best_eligible_dblp_ratio_0_7_gap_macro')}`",
        f"- ratio 0.70 accuracy gap: `{result.get('best_eligible_dblp_ratio_0_7_gap_accuracy')}`",
        "",
        "## Flags",
        "",
        f"- macro_pass: `{result.get('macro_pass')}`",
        f"- accuracy_pass: `{result.get('accuracy_pass')}`",
        f"- accuracy_blocker: `{result.get('accuracy_blocker')}`",
        f"- h6_fill_dependency_flag: `{result.get('h6_fill_dependency_flag')}`",
        f"- h6_fill_only_beats_validation_flag: `{result.get('h6_fill_only_beats_validation_flag')}`",
        f"- random_fill_beats_h6_fill_flag: `{result.get('random_fill_beats_h6_fill_flag')}`",
        "",
        "## Validation-Selected Methods",
        "",
        markdown_table(selected_by_method, ["method", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean", "eligible_for_main_decision_mean"]),
        "",
        "## Exact-Budget Paired Gaps",
        "",
        markdown_table(gaps[:80], ["dataset", "seed", "method", "best_baseline_method", "requested_support_ratio", "delta_macro_f1", "delta_accuracy"]),
    ]
    (output_dir / "gate17_6_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    rows = _find_raw_rows(input_dir)
    present_datasets = {str(row.get("dataset")) for row in rows}
    expected_datasets = set(GATE17_6_SINGLE_SEED_BY_DATASET)
    assert_dataset_integrity(rows, expected=expected_datasets if expected_datasets.issubset(present_datasets) else None)
    typedhash = _typedhash_included(rows)
    selected = validation_selected([dict(row) for row in rows])
    selected_by_method = _aggregate_selected_by_method(selected, typedhash_included=typedhash)
    gaps = exact_budget_paired_gaps([dict(row) for row in rows], typedhash_included=typedhash)
    strong_rows = best_strong_baseline_rows([dict(row) for row in rows])
    write_csv(output_dir / "gate17_6_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_6_by_dataset_selected.csv", selected)
    write_csv(output_dir / "gate17_6_exact_budget_paired_gaps.csv", gaps)
    write_csv(output_dir / "diagnostics" / "gate17_6_typedhash_baseline_check.csv", strong_rows)

    header_pass = _write_header_check(input_dir, output_dir, rows, gaps, selected)
    best_method = _best_method(gaps)
    best_gaps = [row for row in gaps if str(row.get("method")) == str(best_method) and bool(row.get("decision_ratio"))]
    best_rows = [row for row in rows if str(row.get("method")) == str(best_method) and _eligible(row, typedhash_included=typedhash) and _is_decision_ratio(row)]
    dblp_best = [row for row in best_gaps if str(row.get("dataset", "")).upper() == "DBLP"]
    dblp_by_ratio = {f"{_float(row.get('requested_support_ratio')):.1f}": row for row in dblp_best}
    imdb_best = [row for row in best_gaps if str(row.get("dataset", "")).upper() == "IMDB"]
    overall_macro = _group_mean(best_gaps, "delta_macro_f1")
    overall_accuracy = _group_mean(best_gaps, "delta_accuracy")
    dblp_03_macro = None if "0.3" not in dblp_by_ratio else _float(dblp_by_ratio["0.3"].get("delta_macro_f1"))
    dblp_03_acc = None if "0.3" not in dblp_by_ratio else _float(dblp_by_ratio["0.3"].get("delta_accuracy"))
    dblp_07_macro = None if "0.7" not in dblp_by_ratio else _float(dblp_by_ratio["0.7"].get("delta_macro_f1"))
    dblp_07_acc = None if "0.7" not in dblp_by_ratio else _float(dblp_by_ratio["0.7"].get("delta_accuracy"))
    macro_pass = bool(dblp_03_macro is not None and dblp_07_macro is not None and dblp_03_macro >= 0.0 and dblp_07_macro >= 0.0 and overall_macro is not None and overall_macro > 0.0)
    accuracy_pass = bool(dblp_03_acc is not None and dblp_07_acc is not None and dblp_03_acc >= -0.005 and dblp_07_acc >= -0.005 and overall_accuracy is not None and overall_accuracy >= 0.0)
    near_closed_accuracy = bool(dblp_03_acc is not None and dblp_07_acc is not None and dblp_03_acc >= -0.005 and dblp_07_acc >= -0.005)
    no_leakage = bool(_no_test_leakage(rows) and all(_row_no_leakage(row) for row in rows if str(row.get("method", "")).startswith("HeSF-SS")))
    h6_fill_only_flag = _h6_fill_only_beats_validation(rows)
    random_fill_flag = _random_fill_beats_h6_fill(rows)
    h6_dependency_flag = _h6_fill_dependency(rows)
    typedhash_best_count = sum(1 for row in strong_rows if _bool(row.get("typedhash_is_best_strong")))
    stop_reasons: list[str] = []
    if h6_fill_only_flag:
        stop_reasons.append("H6_FILL_ONLY_BEATS_VALIDATION_H6_VARIANTS")
    if _all_hesf_lose_to_typedhash_on_dblp(gaps):
        stop_reasons.append("TYPEDHASH_STRONG_BASELINE_ALL_HESF_DBLP_MACRO_LOSS_GT_0_03")
    if _calibration_destroyed_macro(rows, best_method):
        stop_reasons.append("ACCURACY_CALIBRATION_DESTROYS_MACRO")

    reasons: list[str] = []
    reasons.append("HEADER_NORMALIZATION_PASS" if header_pass else "HEADER_NORMALIZATION_FAIL")
    reasons.append("TYPEDHASH_INCLUDED" if typedhash else "TYPEDHASH_MISSING")
    if macro_pass:
        reasons.append("DBLP_MACRO_PASS_BOTH_RATIOS")
    else:
        reasons.append("DBLP_MACRO_BLOCKER")
    if accuracy_pass:
        reasons.append("ACCURACY_PASS")
    else:
        reasons.append("ACCURACY_BLOCKER")
    reasons.extend(stop_reasons)
    gate18 = bool(
        typedhash
        and no_leakage
        and best_method is not None
        and best_method != H6_FILL_ONLY_METHOD
        and macro_pass
        and (accuracy_pass or (near_closed_accuracy and output_dir.joinpath("diagnostics", "gate17_6_class_shift_report.md").exists()))
        and str(best_rows[0].get("primary_eval_mode", "compressed_projected")) == "compressed_projected" if best_rows else False
    )
    if gate18:
        decision = "ENTER_GATE18_MULTI_SEED"
    elif stop_reasons:
        decision = "STOP_VALIDATION_FILL_SUBLINE"
    else:
        decision = "CONTINUE_GATE17_X_ACCURACY_CALIBRATION"

    result: dict[str, Any] = {
        "stage": "Gate17.6",
        "decision": decision,
        "gate18_allowed": bool(gate18),
        "best_eligible_method": best_method,
        "typedhash_included": bool(typedhash),
        "typedhash_best_strong_baseline_count": int(typedhash_best_count),
        "best_strong_baseline_by_dataset_ratio": strong_rows,
        "best_eligible_overall_exact_delta_macro": overall_macro,
        "best_eligible_overall_exact_delta_accuracy": overall_accuracy,
        "best_eligible_dblp_ratio_0_3_gap_macro": dblp_03_macro,
        "best_eligible_dblp_ratio_0_3_gap_accuracy": dblp_03_acc,
        "best_eligible_dblp_ratio_0_7_gap_macro": dblp_07_macro,
        "best_eligible_dblp_ratio_0_7_gap_accuracy": dblp_07_acc,
        "best_eligible_imdb_mean_gap_macro": _group_mean(imdb_best, "delta_macro_f1"),
        "best_eligible_imdb_mean_gap_accuracy": _group_mean(imdb_best, "delta_accuracy"),
        "best_eligible_validation_macro_f1_mean": _group_mean(best_rows, "validation_macro_f1"),
        "acm_support_saturated": _acm_saturated(rows),
        "acm_used_for_success_evidence": False,
        "accuracy_blocker": not bool(accuracy_pass),
        "macro_pass": bool(macro_pass),
        "accuracy_pass": bool(accuracy_pass),
        "h6_fill_dependency_flag": bool(h6_dependency_flag),
        "h6_fill_only_beats_validation_flag": bool(h6_fill_only_flag),
        "random_fill_beats_h6_fill_flag": bool(random_fill_flag),
        "primary_eval_mode": "compressed_projected",
        "no_test_leakage": bool(no_leakage),
        "header_normalization_pass": bool(header_pass),
        "eligible_method_count": int(len({str(row.get("method")) for row in rows if _eligible(row, typedhash_included=typedhash)})),
        "main_failure_reasons": reasons,
        "rows": int(len(rows)),
        "success": int(sum(1 for row in rows if _success(row))),
        "failed": int(sum(1 for row in rows if not _success(row))),
        "dataset_seed_map": dict(GATE17_6_SINGLE_SEED_BY_DATASET),
    }
    (output_dir / "gate17_6_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_reports(output_dir, result, selected_by_method, gaps)
    _write_class_shift_report(output_dir, rows, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17.6 accuracy-calibrated H6 fill outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17_6_accuracy_calibrated_h6_fill"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
