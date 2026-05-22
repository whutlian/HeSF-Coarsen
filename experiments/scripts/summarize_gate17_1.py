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
    exact_only_paired_gaps,
    read_csv,
    validation_selected,
)


ALLOWED_DECISIONS = {
    "FAIL_MAIN_BRANCH_CODE_INCONSISTENT",
    "FAIL_SUPPORT_BLIND_EVALUATOR",
    "CODEPATH_SMOKE_PASS_METRIC_DEGENERATE",
    "REAL_FEEDBACK_DEGENERATE",
    "PROTOTYPE_SATURATION_BLOCKER",
    "TEACHER_UNRELIABLE_DIAGNOSTIC_ONLY",
    "CONTINUE_TO_GATE18_SUPPORT_SENSITIVE_RUN",
    "DROP_AFTER_NONDEGENERATE_GATE17_1",
}


def _find_raw_rows(input_dir: Path) -> list[dict[str, Any]]:
    for path in [
        input_dir / "gate17_1_raw_rows.csv",
        input_dir / "main" / "gate17_1_raw_rows.csv",
        input_dir / "gate17_raw_rows.csv",
        input_dir / "tables" / "gate17_raw_rows.csv",
    ]:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def _nunique(values: Sequence[Any]) -> int:
    normalized: set[str] = set()
    for value in values:
        try:
            normalized.add(f"{float(value):.12g}")
        except (TypeError, ValueError):
            normalized.add(str(value))
    return int(len(normalized))


def evaluator_metric_nunique(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success(row):
            groups[(str(row.get("dataset")), str(row.get("seed")))].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, seed), group in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "seed": seed,
            "macro_f1_nunique": _nunique([row.get("macro_f1") for row in group]),
            "accuracy_nunique": _nunique([row.get("accuracy") for row in group]),
            "validation_macro_f1_nunique": _nunique([row.get("validation_macro_f1") for row in group]),
            "projected_macro_f1_nunique": _nunique([row.get("projected_macro_f1") for row in group]),
            "transfer_macro_f1_nunique": _nunique([row.get("transfer_macro_f1") for row in group]),
            "method_count": int(len({str(row.get("method")) for row in group})),
            "ratio_count": int(len({str(row.get("requested_support_ratio")) for row in group})),
        }
        item["all_methods_tied"] = bool(
            item["macro_f1_nunique"] <= 1
            and item["accuracy_nunique"] <= 1
            and item["validation_macro_f1_nunique"] <= 1
        )
        out.append(item)
    return out


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
                "support_budget_exact_match_mean": _mean(1.0 if _bool(row.get("support_budget_exact_match")) else 0.0 for row in group),
            }
        )
    return sorted(out, key=lambda row: (-float(row["macro_f1_mean"]), -float(row["accuracy_mean"]), str(row["method"])))


def _dataset_gap(gaps: Sequence[Mapping[str, Any]], method: str, dataset: str) -> float:
    return _mean(
        _float(row.get("delta_macro_f1"))
        for row in gaps
        if str(row.get("method")) == method and str(row.get("dataset")).upper() == dataset.upper()
    )


def _signal_pass(rows: Sequence[Mapping[str, Any]], kind: str) -> bool:
    if kind == "validation":
        return any(
            _float(row.get("validation_trial_count")) > _float(row.get("validation_candidate_pool_size"), 0.0)
            and abs(_float(row.get("validation_greedy_best_gain_max"))) > 1.0e-12
            for row in rows
            if str(row.get("method", "")).startswith(GATE17_PREFIX)
        )
    return any(
        _float(row.get("occlusion_trial_count")) >= max(1.0, _float(row.get("occlusion_candidate_pool_size"), 1.0))
        and (
            abs(_float(row.get("occlusion_delta_macro_f1_mean"))) > 1.0e-12
            or abs(_float(row.get("occlusion_delta_ce_mean"))) > 1.0e-12
            or abs(_float(row.get("occlusion_tree_tensor_l2_delta_mean"))) > 1.0e-12
        )
        and not _bool(row.get("occlusion_degenerate"))
        for row in rows
        if str(row.get("method", "")).startswith(GATE17_PREFIX)
    )


def _semantic_sensitivity(diag_dir: Path) -> tuple[bool, bool, float, float]:
    rows = read_csv(diag_dir / "semantic_tree_delta.csv")
    l2_values = [_float(row.get("tree_tensor_l2_delta_vs_full")) for row in rows]
    changed_values = [_float(row.get("target_path_feature_changed_fraction")) for row in rows]
    p50_l2 = float(np.percentile(l2_values, 50)) if l2_values else 0.0
    changed = max(changed_values) if changed_values else 0.0
    support_blind = bool(p50_l2 <= 1.0e-8 and changed <= 0.0)
    return (not support_blind), support_blind, p50_l2, changed


def _teacher_consistency(diag_dir: Path) -> bool:
    rows = read_csv(diag_dir / "teacher_metric_consistency.csv")
    if not rows:
        return True
    return all(_bool(row.get("teacher_metric_consistent")) for row in rows)


def _prototype_pass(rows: Sequence[Mapping[str, Any]], diag_dir: Path) -> bool:
    proto_rows = read_csv(diag_dir / "prototype_diagnostics.csv") or list(rows)
    if not proto_rows:
        return True
    for row in proto_rows:
        saturation = _float(row.get("prototype_saturation_rate"))
        p90 = _float(row.get("prototype_member_count_p90"))
        cap = _float(row.get("max_members_per_prototype"), 512.0)
        if saturation < 0.50 or p90 < cap:
            return True
    return False


def _decision(result: Mapping[str, Any]) -> str:
    if int(result.get("failed", 0)) > 0 and int(result.get("success", 0)) == 0:
        return "FAIL_MAIN_BRANCH_CODE_INCONSISTENT"
    if bool(result.get("all_methods_tied")) and bool(result.get("support_blind_evaluator")):
        return "FAIL_SUPPORT_BLIND_EVALUATOR"
    if bool(result.get("all_methods_tied")):
        return "CODEPATH_SMOKE_PASS_METRIC_DEGENERATE"
    if not bool(result.get("occlusion_signal_pass")) and not bool(result.get("validation_signal_pass")):
        return "REAL_FEEDBACK_DEGENERATE"
    if not bool(result.get("prototype_saturation_pass")):
        return "PROTOTYPE_SATURATION_BLOCKER"
    if bool(result.get("support_sensitivity_pass")) and (
        bool(result.get("occlusion_signal_pass")) or bool(result.get("validation_signal_pass"))
    ):
        return "CONTINUE_TO_GATE18_SUPPORT_SENSITIVE_RUN"
    return "CODEPATH_SMOKE_PASS_METRIC_DEGENERATE"


def _write_reports(output_dir: Path, result: Mapping[str, Any], selected_by_method: Sequence[Mapping[str, Any]], exact_gaps: Sequence[Mapping[str, Any]]) -> None:
    decision_lines = ["# Gate17.1 Decision", "", f"Decision: `{result['decision']}`", "", "## Checks", ""]
    for key in [
        "primary_eval_mode",
        "all_methods_tied",
        "support_sensitivity_pass",
        "validation_signal_pass",
        "occlusion_signal_pass",
        "prototype_saturation_pass",
        "teacher_metric_consistency_pass",
        "no_test_leakage",
    ]:
        decision_lines.append(f"- {key}: `{result.get(key)}`")
    decision_lines += ["", "## Result JSON", ""]
    decision_lines += [f"- {key}: `{value}`" for key, value in result.items()]
    (output_dir / "gate17_1_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Gate17.1 Final Report",
        "",
        "## Decision",
        "",
        f"- `{result['decision']}`",
        "",
        "## Validation-Selected Method Aggregates",
        "",
        markdown_table(
            selected_by_method,
            ["method", "runs", "macro_f1_mean", "accuracy_mean", "validation_macro_f1_mean", "support_budget_exact_match_mean"],
        ),
        "",
        "## Exact-Budget Paired Gaps",
        "",
        markdown_table(exact_gaps[:20], ["dataset", "seed", "method", "best_baseline_method", "delta_macro_f1", "delta_accuracy"]),
        "",
        "## Degeneracy Checks",
        "",
        f"- all_methods_tied: `{result.get('all_methods_tied')}`",
        f"- support_blind_evaluator: `{result.get('support_blind_evaluator')}`",
        f"- best_validation_selected_method: `{result.get('best_validation_selected_method')}`",
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None, diag_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir / "main"
    diag_dir = Path(diag_dir) if diag_dir is not None else input_dir / "diag"
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows = _find_raw_rows(input_dir)
    selected = validation_selected([dict(row) for row in rows])
    selected_by_method = _aggregate_selected_by_method(selected)
    exact_gaps = exact_only_paired_gaps([dict(row) for row in rows])
    nunique_rows = evaluator_metric_nunique(rows)
    write_csv(output_dir / "gate17_1_validation_selected_by_method.csv", selected_by_method)
    write_csv(output_dir / "gate17_1_by_dataset_selected.csv", selected)
    write_csv(output_dir / "gate17_1_exact_only_paired_gaps.csv", exact_gaps)
    write_csv(diag_dir / "evaluator_metric_nunique.csv", nunique_rows)

    all_methods_tied = bool(nunique_rows) and all(_bool(row.get("all_methods_tied")) for row in nunique_rows)
    method_metric_nunique_max = max(
        [
            int(row.get("macro_f1_nunique", 0))
            for row in nunique_rows
        ]
        + [0]
    )
    support_sensitivity_pass, support_blind, p50_l2, changed = _semantic_sensitivity(diag_dir)
    validation_signal_pass = _signal_pass(rows, "validation")
    occlusion_signal_pass = _signal_pass(rows, "occlusion")
    prototype_saturation_pass = _prototype_pass(rows, diag_dir)
    teacher_metric_consistency_pass = _teacher_consistency(diag_dir)
    best_agg = None
    if not all_methods_tied:
        gate_rows = [row for row in selected_by_method if str(row.get("method", "")).startswith(GATE17_PREFIX)]
        best_agg = max(gate_rows, key=lambda row: (_float(row.get("macro_f1_mean")), _float(row.get("accuracy_mean"))), default=None)
    best_method = None if best_agg is None else str(best_agg.get("method"))
    method_exact_gaps = [row for row in exact_gaps if str(row.get("method")) == str(best_method)]
    selected_best_rows = [row for row in selected if str(row.get("method")) == str(best_method)]
    best_macro = None if best_agg is None else _float(best_agg.get("macro_f1_mean"))
    best_acc = None if best_agg is None else _float(best_agg.get("accuracy_mean"))
    result: dict[str, Any] = {
        "decision": "CODEPATH_SMOKE_PASS_METRIC_DEGENERATE",
        "primary_eval_mode": "compressed_projected",
        "all_methods_tied": bool(all_methods_tied),
        "best_method_tie_count": int(len(selected_by_method)) if all_methods_tied else 1,
        "best_validation_selected_method": best_method,
        "best_validation_selected_macro_f1_mean": best_macro,
        "best_validation_selected_accuracy_mean": best_acc,
        "method_metric_nunique_max": int(method_metric_nunique_max),
        "support_sensitivity_pass": bool(support_sensitivity_pass),
        "support_blind_evaluator": bool(support_blind),
        "semantic_tree_l2_delta_p50": float(p50_l2),
        "semantic_tree_changed_fraction_max": float(changed),
        "occlusion_signal_pass": bool(occlusion_signal_pass),
        "validation_signal_pass": bool(validation_signal_pass),
        "teacher_metric_consistency_pass": bool(teacher_metric_consistency_pass),
        "teacher_diagnostic_only": True,
        "prototype_saturation_pass": bool(prototype_saturation_pass),
        "no_test_leakage": _no_test_leakage(rows),
        "support_budget_exact_match_rate_for_best_method": _mean(1.0 if _bool(row.get("support_budget_exact_match")) else 0.0 for row in selected_best_rows),
        "mean_exact_budget_macro_gap_vs_best_strong_baseline": _mean(_float(row.get("delta_macro_f1")) for row in method_exact_gaps),
        "mean_exact_budget_accuracy_gap_vs_best_strong_baseline": _mean(_float(row.get("delta_accuracy")) for row in method_exact_gaps),
        "dblp_exact_budget_macro_gap": _dataset_gap(exact_gaps, str(best_method), "DBLP") if best_method else 0.0,
        "strong_baselines": sorted(STRONG_BASELINES),
        "failed": sum(1 for row in rows if not _success(row)),
        "success": sum(1 for row in rows if _success(row)),
    }
    result["decision"] = _decision(result)
    if result["decision"] not in ALLOWED_DECISIONS:
        raise ValueError(f"unsupported Gate17.1 decision: {result['decision']}")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_reports(output_dir, result, selected_by_method, exact_gaps)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate17.1 support sensitivity outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate17_1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--diag-dir", type=Path)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir, args.diag_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
