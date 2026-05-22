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
from experiments.scripts.summarize_gate17 import (
    GATE17_PREFIX,
    STRONG_BASELINES,
    _bool,
    _float,
    _mean,
    _no_test_leakage,
    _success,
    read_csv,
    validation_selected,
)


PASS_DECISION = "PASS_EFFECTIVE_BUDGET_AND_FEEDBACK_SMOKE"
FAILURE_PRIORITY = [
    "FAIL_EFFECTIVE_BUDGET_LEAK",
    "FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT",
    "FAIL_REAL_VALIDATION_FEEDBACK_DEGENERATE",
    "FAIL_REAL_OCCLUSION_FEEDBACK_DEGENERATE",
    "FAIL_PROTOTYPE_SATURATION_DBLP",
    "FAIL_ALL_METHODS_TIED",
    "FAIL_NO_TEST_LEAKAGE_CHECK",
]


def _find_raw_rows(input_dir: Path) -> list[dict[str, Any]]:
    for path in [
        input_dir / "gate17_2_raw_rows.csv",
        input_dir / "main" / "gate17_2_raw_rows.csv",
        input_dir / "gate17_1_raw_rows.csv",
    ]:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def _gate_candidate(row: Mapping[str, Any]) -> bool:
    return str(row.get("method", "")).startswith(GATE17_PREFIX)


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
    if not groups:
        return False
    tied = []
    for group in groups.values():
        methods = {str(row.get("method")) for row in group}
        if len(methods) < 2:
            tied.append(False)
            continue
        tied.append(
            _nunique([row.get("macro_f1") for row in group]) <= 1
            and _nunique([row.get("accuracy") for row in group]) <= 1
            and _nunique([row.get("validation_macro_f1") for row in group]) <= 1
        )
    return bool(tied) and all(tied)


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
                "macro_f1_std": float(np.std([_float(row.get("macro_f1")) for row in group], ddof=1)) if len(group) > 1 else 0.0,
                "accuracy_mean": _mean(_float(row.get("accuracy")) for row in group),
                "accuracy_std": float(np.std([_float(row.get("accuracy")) for row in group], ddof=1)) if len(group) > 1 else 0.0,
                "validation_macro_f1_mean": _mean(_float(row.get("validation_macro_f1")) for row in group),
                "effective_budget_exact_match_mean": _mean(1.0 if _bool(row.get("effective_budget_exact_match")) else 0.0 for row in group),
                "candidate_allclose_to_full_rate": _mean(1.0 if _bool(row.get("candidate_allclose_to_full", row.get("allclose_to_full"))) else 0.0 for row in group),
            }
        )
    return sorted(out, key=lambda row: (-float(row["macro_f1_mean"]), -float(row["accuracy_mean"]), str(row["method"])))


def _best_baseline_for(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        item
        for item in rows
        if _success(item)
        and str(item.get("method")) in STRONG_BASELINES
        and str(item.get("dataset")) == str(row.get("dataset"))
        and str(item.get("seed")) == str(row.get("seed"))
        and str(item.get("requested_support_ratio")) == str(row.get("requested_support_ratio"))
        and _bool(item.get("effective_budget_exact_match", item.get("support_budget_exact_match")))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))))


def effective_exact_only_paired_gaps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _success(row) or not _gate_candidate(row):
            continue
        if not _bool(row.get("effective_budget_exact_match", row.get("support_budget_exact_match"))):
            continue
        baseline = _best_baseline_for(row, rows)
        if baseline is None:
            continue
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "requested_support_ratio": row.get("requested_support_ratio"),
                "method": row.get("method"),
                "best_baseline_method": baseline.get("method"),
                "method_effective_budget_exact": _bool(row.get("effective_budget_exact_match", row.get("support_budget_exact_match"))),
                "baseline_budget_exact": _bool(baseline.get("effective_budget_exact_match", baseline.get("support_budget_exact_match"))),
                "method_macro_f1": _float(row.get("macro_f1")),
                "baseline_macro_f1": _float(baseline.get("macro_f1")),
                "delta_macro_f1": float(round(_float(row.get("macro_f1")) - _float(baseline.get("macro_f1")), 12)),
                "method_accuracy": _float(row.get("accuracy")),
                "baseline_accuracy": _float(baseline.get("accuracy")),
                "delta_accuracy": float(round(_float(row.get("accuracy")) - _float(baseline.get("accuracy")), 12)),
            }
        )
    return sorted(out, key=lambda item: (str(item.get("dataset")), str(item.get("method")), str(item.get("seed")), str(item.get("requested_support_ratio"))))


def _budget_leak(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(_budget_leak_rows(rows))


def _budget_leak_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    leaked: list[Mapping[str, Any]] = []
    for row in rows:
        if not _gate_candidate(row) or _float(row.get("requested_support_ratio"), 1.0) >= 1.0:
            continue
        requested = _float(row.get("requested_support_ratio"))
        effective = _float(row.get("effective_support_node_ratio"), requested)
        represented = _float(row.get("represented_support_context_ratio"), effective)
        leak = _float(row.get("budget_leak_ratio"), effective - requested)
        represented_leak = _float(row.get("represented_context_leak_ratio"), represented - requested)
        if (
            leak > 0.02
            or represented_leak > 0.02
            or effective > requested + 0.02
            or represented > requested + 0.02
            or not _bool(row.get("effective_budget_exact_match", True))
        ):
            leaked.append(row)
    return leaked


def _candidate_full_graph_equivalent(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _gate_candidate(row)
        and _float(row.get("requested_support_ratio"), 1.0) < 1.0
        and _bool(row.get("candidate_allclose_to_full", row.get("allclose_to_full")))
        for row in rows
    )


def _validation_degenerate(rows: Sequence[Mapping[str, Any]]) -> bool:
    relevant = [row for row in rows if str(row.get("method")) == "HeSF-SS-real-validation-no-fallback"]
    if not relevant:
        return False
    return any(
        _bool(row.get("real_validation_degenerate"))
        or _float(row.get("validation_greedy_best_gain_max")) <= 0.0
        or _float(row.get("proxy_fallback_fill_count")) > 0.0
        or _float(row.get("validation_scores_unique_count"), 2.0) <= 1.0
        for row in relevant
        if _success(row)
    )


def _occlusion_degenerate(rows: Sequence[Mapping[str, Any]]) -> bool:
    relevant = [
        row
        for row in rows
        if str(row.get("method")) in {"HeSF-SS-real-occlusion-no-fallback", "HeSF-SS-real-occlusion-plus-dblp-prototype-budgeted"}
    ]
    if not relevant:
        return False
    return any(
        _bool(row.get("occlusion_degenerate"))
        or not _bool(row.get("occlusion_metric_complete"))
        or _bool(row.get("occlusion_proxy_fallback_used"))
        or (
            _float(row.get("occlusion_nonzero_delta_rate")) <= 0.0
            and _float(row.get("occlusion_tree_delta_nonzero_rate")) <= 0.0
        )
        for row in relevant
        if _success(row)
    )


def _prototype_saturation_dblp(rows: Sequence[Mapping[str, Any]]) -> bool:
    relevant = [
        row
        for row in rows
        if str(row.get("dataset", "")).upper() == "DBLP"
        and _gate_candidate(row)
        and "prototype" in str(row.get("method", "")).lower()
    ]
    if not relevant:
        return False
    for row in relevant:
        cap = _float(row.get("max_members_per_prototype"), 512.0)
        if (
            _float(row.get("prototype_saturation_rate")) > 0.50
            or _float(row.get("prototype_member_count_p90")) >= cap
            or _float(row.get("prototype_member_count_p99")) >= cap
            or _float(row.get("rare_class_never_fallback_violation_count", row.get("rare_class_fallback_count"))) > 0.0
        ):
            return True
    return False


def _write_reports(output_dir: Path, result: Mapping[str, Any], selected_by_method: Sequence[Mapping[str, Any]], exact_gaps: Sequence[Mapping[str, Any]]) -> None:
    decision_lines = [
        "# Gate17.2 Decision",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    reasons = list(result.get("failure_reasons", []))
    decision_lines.extend([f"- `{reason}`" for reason in reasons] or ["- none"])
    decision_lines += [
        "",
        "## Checks",
        "",
        f"- effective_budget_pass: `{result.get('effective_budget_pass')}`",
        f"- candidate_support_sensitivity_pass: `{result.get('candidate_support_sensitivity_pass')}`",
        f"- validation_feedback_pass: `{result.get('validation_feedback_pass')}`",
        f"- occlusion_feedback_pass: `{result.get('occlusion_feedback_pass')}`",
        f"- prototype_saturation_dblp_pass: `{result.get('prototype_saturation_dblp_pass')}`",
        f"- no_test_leakage: `{result.get('no_test_leakage')}`",
        f"- gate18_allowed: `{result.get('gate18_allowed')}`",
    ]
    (output_dir / "gate17_2_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Gate17.2 Final Report",
        "",
        "## Decision",
        "",
        f"- `{result['decision']}`",
        "",
        "## Validation-Selected Method Aggregates",
        "",
        markdown_table(
            selected_by_method,
            ["method", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean", "effective_budget_exact_match_mean"],
        ),
        "",
        "## Effective Exact-Budget Paired Gaps",
        "",
        markdown_table(exact_gaps[:20], ["dataset", "seed", "method", "best_baseline_method", "delta_macro_f1", "delta_accuracy"]),
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None, diag_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    diag_dir = Path(diag_dir) if diag_dir is not None else input_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows = _find_raw_rows(input_dir)
    effective_rows = read_csv(diag_dir / "effective_budget.csv") or [row for row in rows if _gate_candidate(row)]
    candidate_rows = read_csv(diag_dir / "candidate_semantic_delta.csv") or [row for row in rows if _gate_candidate(row)]
    prototype_rows = read_csv(diag_dir / "prototype_budget_saturation.csv") or [row for row in rows if _gate_candidate(row)]
    validation_rows = read_csv(diag_dir / "validation_feedback_trials.csv")
    occlusion_rows = read_csv(diag_dir / "occlusion_feedback_scores.csv")

    selected = validation_selected([dict(row) for row in rows])
    selected_by_method = _aggregate_selected_by_method(selected)
    exact_gaps = effective_exact_only_paired_gaps([dict(row) for row in rows])
    write_csv(output_dir / "gate17_2_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_2_by_dataset_selected.csv", selected)
    write_csv(output_dir / "gate17_2_exact_only_paired_gaps.csv", exact_gaps)

    failure_reasons: list[str] = []
    if _budget_leak(effective_rows):
        failure_reasons.append("FAIL_EFFECTIVE_BUDGET_LEAK")
    if _candidate_full_graph_equivalent(candidate_rows):
        failure_reasons.append("FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT")
    if _validation_degenerate(rows) or any(_bool(row.get("real_validation_degenerate")) for row in validation_rows if str(row.get("method")) == "HeSF-SS-real-validation-no-fallback"):
        failure_reasons.append("FAIL_REAL_VALIDATION_FEEDBACK_DEGENERATE")
    if _occlusion_degenerate(rows) or any(not _bool(row.get("occlusion_metric_complete", True)) for row in occlusion_rows if str(row.get("method", "")).startswith("HeSF-SS-real-occlusion")):
        failure_reasons.append("FAIL_REAL_OCCLUSION_FEEDBACK_DEGENERATE")
    if _prototype_saturation_dblp(prototype_rows):
        failure_reasons.append("FAIL_PROTOTYPE_SATURATION_DBLP")
    all_methods_tied = _all_methods_tied(rows)
    if all_methods_tied:
        failure_reasons.append("FAIL_ALL_METHODS_TIED")
    no_test_leakage = _no_test_leakage(rows)
    if not no_test_leakage:
        failure_reasons.append("FAIL_NO_TEST_LEAKAGE_CHECK")

    failure_reasons = [reason for reason in FAILURE_PRIORITY if reason in set(failure_reasons)]
    decision = next((reason for reason in FAILURE_PRIORITY if reason in failure_reasons), PASS_DECISION)
    best_agg = None
    if not failure_reasons:
        gate_aggs = [row for row in selected_by_method if str(row.get("method", "")).startswith(GATE17_PREFIX)]
        best_agg = max(gate_aggs, key=lambda row: (_float(row.get("macro_f1_mean")), _float(row.get("accuracy_mean"))), default=None)
    best_method = None if best_agg is None else str(best_agg.get("method"))
    result: dict[str, Any] = {
        "decision": decision,
        "failure_reasons": failure_reasons,
        "gate18_allowed": bool(decision == PASS_DECISION),
        "primary_eval_mode": "compressed_projected",
        "rows": int(len(rows)),
        "success": int(sum(1 for row in rows if _success(row))),
        "failed": int(sum(1 for row in rows if not _success(row))),
        "all_methods_tied": bool(all_methods_tied),
        "effective_budget_pass": "FAIL_EFFECTIVE_BUDGET_LEAK" not in failure_reasons,
        "candidate_support_sensitivity_pass": "FAIL_CANDIDATE_FULL_GRAPH_EQUIVALENT" not in failure_reasons,
        "validation_feedback_pass": "FAIL_REAL_VALIDATION_FEEDBACK_DEGENERATE" not in failure_reasons,
        "occlusion_feedback_pass": "FAIL_REAL_OCCLUSION_FEEDBACK_DEGENERATE" not in failure_reasons,
        "prototype_saturation_dblp_pass": "FAIL_PROTOTYPE_SATURATION_DBLP" not in failure_reasons,
        "no_test_leakage": bool(no_test_leakage),
        "best_method_is_meaningful": bool(best_method is not None and not failure_reasons),
        "best_validation_selected_method": best_method,
        "best_validation_selected_macro_f1_mean": None if best_agg is None else _float(best_agg.get("macro_f1_mean")),
        "best_validation_selected_accuracy_mean": None if best_agg is None else _float(best_agg.get("accuracy_mean")),
        "effective_budget_leak_row_count": int(len(_budget_leak_rows(effective_rows))),
        "candidate_allclose_to_full_count": int(sum(1 for row in candidate_rows if _gate_candidate(row) and _bool(row.get("candidate_allclose_to_full", row.get("allclose_to_full"))))),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_reports(output_dir, result, selected_by_method, exact_gaps)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17.2 effective-budget feedback gate outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17_2_single_seed"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--diag-dir", type=Path)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir, args.diag_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
