from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.summarize_gate17 import _bool, _float, _mean, _no_test_leakage, _success, assert_dataset_integrity, read_csv, validation_selected


GATE17_PREFIX = "HeSF-SS"
STRONG_BASELINES = {"H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only"}
GATE17_3_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
FAILURE_PRIORITY = [
    "FAIL_NO_TEST_LEAKAGE_CHECK",
    "FAIL_NODE_BUDGET",
    "FAIL_REPRESENTED_CONTEXT_BUDGET",
    "FAIL_FULL_RESIDUAL_SHORTCUT_ONLY",
    "FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT",
    "FAIL_REAL_FEEDBACK_TASK_DEGENERATE",
    "FAIL_DBLP_PROTOTYPE_SATURATION",
    "FAIL_DBLP_GAP_VS_STRONG_BASELINE",
    "FAIL_ALL_METHODS_TIED",
]


def _find_raw_rows(input_dir: Path) -> list[dict[str, Any]]:
    for path in [input_dir / "gate17_3_raw_rows.csv", input_dir / "main" / "gate17_3_raw_rows.csv"]:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def _gate(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")).startswith(GATE17_PREFIX)


def _diagnostic(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    return "upperbound" in method.lower() or "full-residual" in method.lower() or method == "HeSF-SS-real-validation-no-fallback"


def _eligible(row: Mapping[str, Any]) -> bool:
    requested = _float(row.get("requested_support_ratio"))
    represented = _float(row.get("represented_context_ratio"), requested)
    return bool(
        _success(row)
        and _gate(row)
        and not _diagnostic(row)
        and _bool(row.get("eligible_for_main_decision", True))
        and _bool(row.get("node_budget_exact_match", row.get("support_budget_exact_match", False)))
        and represented <= requested + 0.10 + 1.0e-12
        and not _bool(row.get("selector_uses_test_labels"))
        and not _bool(row.get("teacher_uses_test_labels_for_training"))
    )


def _nunique(values: Sequence[Any]) -> int:
    normalized: set[str] = set()
    for value in values:
        try:
            normalized.add(f"{float(value):.12g}")
        except (TypeError, ValueError):
            normalized.add(str(value))
    return int(len(normalized))


def _all_methods_tied(rows: Sequence[Mapping[str, Any]]) -> bool:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success(row):
            groups[(str(row.get("dataset")), str(row.get("seed")))].append(row)
    return bool(groups) and all(
        len({str(row.get("method")) for row in group}) > 1
        and _nunique([row.get("macro_f1") for row in group]) <= 1
        and _nunique([row.get("validation_macro_f1") for row in group]) <= 1
        for group in groups.values()
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


def _best_strong_baseline(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        item
        for item in rows
        if _success(item)
        and str(item.get("method")) in STRONG_BASELINES
        and str(item.get("dataset")) == str(row.get("dataset"))
        and str(item.get("seed")) == str(row.get("seed"))
        and str(item.get("requested_support_ratio")) == str(row.get("requested_support_ratio"))
        and _bool(item.get("node_budget_exact_match", item.get("support_budget_exact_match", True)))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))))


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
            }
        )
    return sorted(out, key=lambda item: (str(item.get("dataset")), str(item.get("method")), str(item.get("requested_support_ratio"))))


def _dblp_gap(gaps: Sequence[Mapping[str, Any]], best_main_method: str | None = None) -> float | None:
    rows = [
        row
        for row in gaps
        if str(row.get("dataset", "")).upper() == "DBLP"
        and (best_main_method is None or str(row.get("method")) == str(best_main_method))
    ]
    if not rows:
        return None
    return _mean(_float(row.get("delta_macro_f1")) for row in rows)


def _acm_support_saturated(rows: Sequence[Mapping[str, Any]]) -> bool:
    acm = [
        row
        for row in rows
        if str(row.get("dataset", "")).upper() == "ACM"
        and _gate(row)
        and _float(row.get("requested_support_ratio")) <= 0.05
    ]
    return bool(acm) and all(_bool(row.get("candidate_allclose_to_full", row.get("allclose_to_full"))) for row in acm)


def _write_reports(output_dir: Path, result: Mapping[str, Any], selected_by_method: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> None:
    decision_lines = [
        "# Gate17.3 Decision",
        "",
        "Single-seed diagnostic only; do not interpret as paper-level performance.",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Failure Reasons",
        "",
        *([f"- `{reason}`" for reason in result.get("failure_reasons", [])] or ["- none"]),
    ]
    (output_dir / "gate17_3_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    report_lines = [
        "# Gate17.3 Final Report",
        "",
        "Single-seed diagnostic only.",
        "",
        f"- decision: `{result['decision']}`",
        f"- best_main_method: `{result.get('best_main_method')}`",
        f"- dblp_gap_vs_best_strong_baseline: `{result.get('dblp_gap_vs_best_strong_baseline')}`",
        "",
        "## Validation-Selected Methods",
        "",
        markdown_table(selected_by_method, ["method", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean", "eligible_for_main_decision_mean"]),
        "",
        "## Exact-Budget Main Paired Gaps",
        "",
        markdown_table(gaps[:30], ["dataset", "seed", "method", "best_baseline_method", "delta_macro_f1", "delta_accuracy"]),
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    diag_dir = input_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows = _find_raw_rows(input_dir)
    assert_dataset_integrity(rows)
    selected = validation_selected([dict(row) for row in rows])
    selected_by_method = _aggregate_selected_by_method(selected)
    gaps = exact_budget_paired_gaps([dict(row) for row in rows])
    write_csv(output_dir / "gate17_3_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_3_by_dataset_selected.csv", selected)
    write_csv(output_dir / "gate17_3_exact_budget_paired_gaps.csv", gaps)

    no_test_leakage = _no_test_leakage(rows)
    eligible_rows = [row for row in rows if _eligible(row)]
    best_row = max(eligible_rows, key=lambda row: (_float(row.get("macro_f1")), _float(row.get("accuracy"))), default=None)
    best_main_method = None if best_row is None else str(best_row.get("method"))
    dblp_gap = _dblp_gap(gaps, best_main_method)
    failure_set: set[str] = set()
    if not no_test_leakage:
        failure_set.add("FAIL_NO_TEST_LEAKAGE_CHECK")
    if any(_gate(row) and not _diagnostic(row) and not _bool(row.get("node_budget_exact_match", True)) for row in rows):
        failure_set.add("FAIL_NODE_BUDGET")
    if any(_gate(row) and not _diagnostic(row) and not _bool(row.get("represented_context_exact_or_bounded", True)) for row in rows):
        failure_set.add("FAIL_REPRESENTED_CONTEXT_BUDGET")
    if any(_gate(row) and _diagnostic(row) and _float(row.get("represented_context_ratio")) >= 0.99 for row in rows):
        failure_set.add("FAIL_FULL_RESIDUAL_SHORTCUT_ONLY")
    candidate_rows = [row for row in rows if _gate(row) and not _diagnostic(row) and _float(row.get("requested_support_ratio"), 1.0) < 1.0]
    if candidate_rows and _mean(1.0 if _bool(row.get("candidate_allclose_to_full", row.get("allclose_to_full"))) else 0.0 for row in candidate_rows) >= 0.20:
        failure_set.add("FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT")
    dblp_gate = [row for row in rows if str(row.get("dataset", "")).upper() == "DBLP" and _gate(row) and not _diagnostic(row)]
    if dblp_gate and not any(_bool(row.get("occlusion_task_signal_pass")) or _bool(row.get("validation_signal_pass")) for row in dblp_gate):
        failure_set.add("FAIL_REAL_FEEDBACK_TASK_DEGENERATE")
    for row in dblp_gate:
        cap = _float(row.get("max_members_per_prototype"), 512.0)
        if (
            _float(row.get("prototype_saturation_rate")) > 0.50
            or _float(row.get("prototype_member_count_p90")) >= cap
            or _float(row.get("prototype_member_count_p99")) >= cap
        ):
            failure_set.add("FAIL_DBLP_PROTOTYPE_SATURATION")
            break
    if dblp_gap is not None and dblp_gap < -0.05:
        failure_set.add("FAIL_DBLP_GAP_VS_STRONG_BASELINE")
    all_methods_tied = _all_methods_tied(rows)
    if all_methods_tied:
        failure_set.add("FAIL_ALL_METHODS_TIED")
    failure_reasons = [reason for reason in FAILURE_PRIORITY if reason in failure_set]
    meaningful = bool(best_main_method and not failure_reasons)
    if failure_reasons:
        best_main_method = None
    if failure_reasons:
        if set(failure_reasons) == {"FAIL_FULL_RESIDUAL_SHORTCUT_ONLY"}:
            decision = "FAIL_FULL_RESIDUAL_SHORTCUT_ONLY"
        else:
            decision = "FAIL_SELECTOR_AND_LOSSY_PROTOTYPE"
    else:
        decision = "PASS_GATE17_3_READY_FOR_GATE18" if (dblp_gap is not None and dblp_gap >= 0.0) else "CONTINUE_LOSSY_FEEDBACK_DBLP_CLOSE"
    result: dict[str, Any] = {
        "decision": decision,
        "failure_reasons": failure_reasons,
        "gate18_allowed": bool(decision == "PASS_GATE17_3_READY_FOR_GATE18"),
        "primary_eval_mode": "compressed_projected",
        "single_seed_diagnostic_only": True,
        "dataset_seed_map": dict(GATE17_3_SINGLE_SEED_BY_DATASET),
        "node_budget_pass": "FAIL_NODE_BUDGET" not in failure_reasons,
        "represented_context_budget_pass": "FAIL_REPRESENTED_CONTEXT_BUDGET" not in failure_reasons,
        "candidate_full_equivalence_pass": "FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT" not in failure_reasons,
        "occlusion_task_signal_pass": any(_bool(row.get("occlusion_task_signal_pass")) for row in dblp_gate),
        "validation_signal_pass": any(_bool(row.get("validation_signal_pass")) for row in dblp_gate),
        "prototype_saturation_dblp_pass": "FAIL_DBLP_PROTOTYPE_SATURATION" not in failure_reasons,
        "acm_support_saturated": _acm_support_saturated(rows),
        "dblp_gap_vs_best_strong_baseline": None if dblp_gap is None else float(round(dblp_gap, 12)),
        "best_main_method": best_main_method,
        "best_method_is_meaningful": bool(meaningful and best_main_method is not None),
        "full_residual_upperbound_excluded_from_decision": True,
        "no_test_leakage": bool(no_test_leakage),
        "all_methods_tied": bool(all_methods_tied),
        "rows": int(len(rows)),
        "success": int(sum(1 for row in rows if _success(row))),
        "failed": int(sum(1 for row in rows if not _success(row))),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_reports(output_dir, result, selected_by_method, gaps)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17.3 lossy feedback outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17_3_single_seed"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
