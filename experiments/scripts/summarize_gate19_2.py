from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_gate19 import _bool, _float, read_csv


SUPPORT_FAMILIES = {"calibrated_support_baseline", "hesf_cal_support", "hesf_cal_ensemble"}
STC_FAMILIES = {"stc_compressed", "stc_support_teacher_distill"}
PARETO_FAMILIES = SUPPORT_FAMILIES | STC_FAMILIES | {"support_baseline", "full_stc_reference"}
REQUIRED_CALIBRATED_SUPPORT = {
    "H6-no-spec-support-only-logit-calibrated",
    "flatten-sum-support-only-logit-calibrated",
    "TypedHash-ChebHeat-support-only-logit-calibrated",
    "best-support-baseline-logit-calibrated",
}
REQUIRED_HESF_CAL = {"HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash", "HeSF-CAL-best-support"}
ALLOWED_DECISIONS = {
    "CONTINUE_TO_GATE20_HESF_CAL_MULTI_SEED",
    "CONTINUE_GATE19_X_CALIBRATION_STABILITY",
    "STC_DEMOTED_TO_DEPLOYMENT_AUXILIARY",
    "STC_SUPPORT_TEACHER_DISTILLATION_PROMISING",
    "GATE20_BLOCKED_BY_CALIBRATION_INSTABILITY",
    "GATE20_BLOCKED_BY_STC_DOMINATED",
}


def _status_ok(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "success")) == "success" and not _bool(row.get("method_invalid", False))


def _eligible(row: Mapping[str, Any]) -> bool:
    if not _status_ok(row):
        return False
    if _bool(row.get("diagnostic_only", False)):
        return False
    if not _bool(row.get("eligible_for_main_decision", True)):
        return False
    return str(row.get("method_family", "")) in PARETO_FAMILIES


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        _float(row.get("validation_accuracy")),
        _float(row.get("validation_macro_f1", row.get("val_macro"))),
        -_float(row.get("total_storage_ratio_vs_full_stc"), 1.0e12),
        str(row.get("method", "")),
    )


def validation_selected_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not _status_ok(row):
            continue
        if str(row.get("dataset", "")) == "" or str(row.get("method", "")) == "":
            continue
        groups.setdefault((str(row.get("dataset")), str(row.get("method"))), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(groups.items()):
        best = dict(max(group, key=_selection_key))
        best["selection_rule"] = "validation_accuracy_then_validation_macro_then_lower_cost"
        best["selected_requested_budget"] = best.get("requested_budget", best.get("requested_support_ratio", ""))
        best["test_oracle_used_for_selection"] = False
        out.append(best)
    return out


def best_by_validation_selection(
    rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Mapping[str, Any]]:
    selected = [row for row in validation_selected_rows(rows) if predicate(row) and _eligible(row)]
    out: dict[str, Mapping[str, Any]] = {}
    for dataset in sorted({str(row.get("dataset")) for row in selected if row.get("dataset") not in {"", None}}):
        candidates = [row for row in selected if str(row.get("dataset")) == dataset]
        if candidates:
            out[dataset] = max(candidates, key=_selection_key)
    return out


def best_by_test_oracle(
    rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    candidates = [row for row in rows if predicate(row) and _eligible(row)]
    for dataset in sorted({str(row.get("dataset")) for row in candidates if row.get("dataset") not in {"", None}}):
        group = [row for row in candidates if str(row.get("dataset")) == dataset]
        if group:
            out[dataset] = max(
                group,
                key=lambda item: (
                    _float(item.get("accuracy")),
                    _float(item.get("macro_f1")),
                    -_float(item.get("total_storage_ratio_vs_full_stc"), 1.0e12),
                ),
            )
    return out


def build_gate19_2_pareto(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if _eligible(row)]
    out: list[dict[str, Any]] = []
    for row in eligible:
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed", ""))
        cost = _float(row.get("total_storage_ratio_vs_full_stc"), 0.0)
        macro = _float(row.get("macro_f1"))
        accuracy = _float(row.get("accuracy"))
        dominated_by = ""
        best_same_or_lower_cost: Mapping[str, Any] | None = None
        best_above_compression: Mapping[str, Any] | None = None
        for other in eligible:
            if other is row or str(other.get("dataset")) != dataset or str(other.get("seed", "")) != seed:
                continue
            other_cost = _float(other.get("total_storage_ratio_vs_full_stc"), 0.0)
            other_macro = _float(other.get("macro_f1"))
            other_accuracy = _float(other.get("accuracy"))
            if other_cost <= cost + 1.0e-12:
                if best_same_or_lower_cost is None or (_float(other.get("accuracy")), _float(other.get("macro_f1"))) > (
                    _float(best_same_or_lower_cost.get("accuracy")),
                    _float(best_same_or_lower_cost.get("macro_f1")),
                ):
                    best_same_or_lower_cost = other
            if other_cost >= cost - 1.0e-12:
                if best_above_compression is None or (_float(other.get("accuracy")), _float(other.get("macro_f1"))) > (
                    _float(best_above_compression.get("accuracy")),
                    _float(best_above_compression.get("macro_f1")),
                ):
                    best_above_compression = other
            if (
                other_cost <= cost + 1.0e-12
                and other_macro >= macro - 1.0e-12
                and other_accuracy >= accuracy - 1.0e-12
                and (other_cost < cost - 1.0e-12 or other_macro > macro + 1.0e-12 or other_accuracy > accuracy + 1.0e-12)
            ):
                dominated_by = str(other.get("method", ""))
                break
        frontier_ref = best_same_or_lower_cost or row
        out.append(
            {
                "dataset": dataset,
                "seed": int(_float(seed, 0.0)),
                "method": row.get("method", ""),
                "method_family": row.get("method_family", ""),
                "support_ratio": row.get("requested_support_ratio", row.get("requested_budget", "")),
                "requested_budget": row.get("requested_budget", ""),
                "total_storage_ratio_vs_full_stc": cost,
                "total_storage_ratio_vs_full_graph": _float(row.get("total_storage_ratio_vs_full_graph")),
                "macro_f1": macro,
                "accuracy": accuracy,
                "validation_macro_f1": _float(row.get("validation_macro_f1")),
                "validation_accuracy": _float(row.get("validation_accuracy")),
                "support_node_ratio": _float(row.get("support_node_ratio")),
                "support_edge_ratio": _float(row.get("support_edge_ratio")),
                "unit_count_ratio": _float(row.get("unit_count_ratio")),
                "feature_cache_size_ratio": _float(row.get("feature_cache_size_ratio")),
                "path_channel_count_ratio": _float(row.get("path_channel_count_ratio")),
                "pareto_dominated_by": dominated_by,
                "best_baseline_at_or_below_cost": "" if best_same_or_lower_cost is None else best_same_or_lower_cost.get("method", ""),
                "best_baseline_at_or_above_compression": "" if best_above_compression is None else best_above_compression.get("method", ""),
                "delta_macro_vs_frontier": float(macro - _float(frontier_ref.get("macro_f1"))),
                "delta_accuracy_vs_frontier": float(accuracy - _float(frontier_ref.get("accuracy"))),
            }
        )
    return sorted(out, key=lambda item: (str(item["dataset"]), float(item["total_storage_ratio_vs_full_stc"]), -float(item["accuracy"]), str(item["method"])))


def _best_by_pareto(pareto_rows: Sequence[Mapping[str, Any]], method_set: set[str] | None = None) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    frontier = [
        row
        for row in pareto_rows
        if str(row.get("pareto_dominated_by", "")) == "" and (method_set is None or str(row.get("method")) in method_set)
    ]
    for dataset in sorted({str(row.get("dataset")) for row in frontier if row.get("dataset") not in {"", None}}):
        group = [row for row in frontier if str(row.get("dataset")) == dataset]
        if group:
            out[dataset] = max(group, key=lambda item: (_float(item.get("accuracy")), _float(item.get("macro_f1")), -_float(item.get("total_storage_ratio_vs_full_stc"))))
    return out


def _nested_stats(nested_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    db = [
        row
        for row in nested_rows
        if str(row.get("dataset")) == "DBLP"
        and str(row.get("method", "")).startswith("HeSF-CAL")
        and str(row.get("method")) != "HeSF-CAL-best-support"
    ]
    if not db:
        return {"pass": False, "accuracy_std": 0.0, "macro_std": 0.0, "constraint_rate": 0.0}
    by_method: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in db:
        by_method.setdefault((str(row.get("method")), str(row.get("support_ratio", row.get("requested_budget", "")))), []).append(row)
    best_pass = False
    best_acc_std = 0.0
    best_macro_std = 0.0
    best_rate = 0.0
    for group in by_method.values():
        acc = np.asarray([_float(row.get("test_accuracy")) for row in group], dtype=np.float64)
        macro = np.asarray([_float(row.get("test_macro", row.get("test_macro_f1"))) for row in group], dtype=np.float64)
        rate = float(np.mean([_bool(row.get("constraint_satisfied", False)) for row in group]))
        acc_std = float(np.std(acc)) if len(acc) else 0.0
        macro_std = float(np.std(macro)) if len(macro) else 0.0
        passed = acc_std <= 0.01 and macro_std <= 0.01 and rate >= 0.8
        key = (passed, -acc_std, -macro_std, rate)
        if not best_pass or key > (best_pass, -best_acc_std, -best_macro_std, best_rate):
            best_pass = bool(passed)
            best_acc_std = acc_std
            best_macro_std = macro_std
            best_rate = rate
    return {"pass": bool(best_pass), "accuracy_std": best_acc_std, "macro_std": best_macro_std, "constraint_rate": best_rate}


def _row_method(rows_by_dataset: Mapping[str, Mapping[str, Any]], dataset: str) -> str:
    row = rows_by_dataset.get(dataset, {})
    return str(row.get("method", "none"))


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    nested_rows: Sequence[Mapping[str, Any]] | None = None,
    per_class_present: bool,
    confusion_present: bool,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    nested_rows = list(nested_rows or [])
    pareto = build_gate19_2_pareto(rows)
    support_val = best_by_validation_selection(rows, lambda row: str(row.get("method_family")) in SUPPORT_FAMILIES)
    stc_val = best_by_validation_selection(rows, lambda row: str(row.get("method_family")) in STC_FAMILIES)
    support_test = best_by_test_oracle(rows, lambda row: str(row.get("method_family")) in SUPPORT_FAMILIES)
    stc_test = best_by_test_oracle(rows, lambda row: str(row.get("method_family")) in STC_FAMILIES)
    stc_methods = {str(row.get("method")) for row in rows if str(row.get("method_family")) in STC_FAMILIES}
    support_methods = {str(row.get("method")) for row in rows if str(row.get("method_family")) in SUPPORT_FAMILIES}
    stc_pareto = _best_by_pareto(pareto, stc_methods)
    support_pareto = _best_by_pareto(pareto, support_methods)

    success_rows = [row for row in rows if _status_ok(row)]
    primary_modes = {str(row.get("primary_eval_mode", "")) for row in success_rows}
    primary_eval_mode = "compressed_projected" if primary_modes in ({"compressed_projected"}, set()) else "mixed"
    no_test_leakage = all(_bool(row.get("no_test_leakage", True)) and not _bool(row.get("calibration_uses_test_labels", False)) for row in success_rows)
    method_names = {str(row.get("method")) for row in success_rows}
    typedhash_included = any("TypedHash" in method for method in method_names)
    calibrated_support_included = REQUIRED_CALIBRATED_SUPPORT <= method_names or REQUIRED_HESF_CAL <= method_names
    nested_stats = _nested_stats(nested_rows)
    if not nested_rows:
        nested_stats["pass"] = any(_bool(row.get("nested_calibration_pass", False)) for row in success_rows if str(row.get("dataset")) == "DBLP")
    per_class_confusion_present = bool(per_class_present and confusion_present)

    support_dblp = support_val.get("DBLP", {})
    stc_dblp = stc_val.get("DBLP", {})
    best_all = best_by_validation_selection(rows, lambda row: str(row.get("method_family")) in (SUPPORT_FAMILIES | STC_FAMILIES))
    best_validation = best_all.get("DBLP", support_dblp or stc_dblp or {})
    dblp_support_acc = _float(support_dblp.get("accuracy"))
    dblp_support_macro = _float(support_dblp.get("macro_f1"))
    dblp_stc_acc = _float(stc_dblp.get("accuracy"))
    dblp_stc_macro = _float(stc_dblp.get("macro_f1"))
    acc_gap = float(dblp_stc_acc - dblp_support_acc)
    macro_gap = float(dblp_stc_macro - dblp_support_macro)
    stc_pareto_non_dominated = bool(
        stc_dblp
        and any(
            str(row.get("dataset")) == "DBLP"
            and str(row.get("method")) == str(stc_dblp.get("method"))
            and str(row.get("pareto_dominated_by", "")) == ""
            for row in pareto
        )
    )
    hesf_gate20_conditions = [
        dblp_support_acc >= 0.90 or dblp_support_acc >= max(dblp_support_acc, dblp_stc_acc) - 0.005,
        dblp_support_macro >= _float(support_val.get("DBLP", {}).get("macro_f1")) - 0.005,
        bool(nested_stats["pass"]),
        typedhash_included,
        no_test_leakage,
        primary_eval_mode == "compressed_projected",
        per_class_confusion_present,
    ]
    stc_promising = bool(stc_pareto_non_dominated and acc_gap >= -0.005 and macro_gap >= -0.005)
    if not bool(nested_stats["pass"]):
        decision = "GATE20_BLOCKED_BY_CALIBRATION_INSTABILITY"
    elif stc_promising:
        decision = "STC_SUPPORT_TEACHER_DISTILLATION_PROMISING"
    elif all(hesf_gate20_conditions):
        decision = "CONTINUE_TO_GATE20_HESF_CAL_MULTI_SEED"
    elif acc_gap < -0.005 or macro_gap < -0.005:
        decision = "STC_DEMOTED_TO_DEPLOYMENT_AUXILIARY"
    else:
        decision = "GATE20_BLOCKED_BY_STC_DOMINATED"
    gate20_allowed = decision in {"CONTINUE_TO_GATE20_HESF_CAL_MULTI_SEED", "STC_SUPPORT_TEACHER_DISTILLATION_PROMISING"}
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"invalid Gate19.2 decision: {decision}")
    return {
        "stage": "Gate19.2",
        "decision": decision,
        "gate20_allowed": bool(gate20_allowed),
        "primary_eval_mode": primary_eval_mode,
        "no_test_leakage": bool(no_test_leakage),
        "typedhash_included": bool(typedhash_included),
        "calibrated_support_baselines_included": bool(calibrated_support_included),
        "nested_calibration_audit_pass": bool(nested_stats["pass"]),
        "per_class_confusion_present": bool(per_class_confusion_present),
        "test_oracle_used_for_decision": False,
        "best_validation_selected_method": str(best_validation.get("method", "")),
        "best_validation_selected_family": str(best_validation.get("method_family", "")),
        "best_he_sf_cal_method": _row_method(support_val, "DBLP"),
        "best_calibrated_support_by_validation_selection": _row_method(support_val, "DBLP"),
        "best_calibrated_support_by_test_oracle": _row_method(support_test, "DBLP"),
        "best_calibrated_support_by_pareto": _row_method(support_pareto, "DBLP"),
        "best_stc_by_validation_selection": _row_method(stc_val, "DBLP"),
        "best_stc_by_test_oracle": _row_method(stc_test, "DBLP"),
        "best_stc_by_pareto": _row_method(stc_pareto, "DBLP"),
        "dblp_best_calibrated_support_accuracy": float(dblp_support_acc),
        "dblp_best_calibrated_support_macro": float(dblp_support_macro),
        "dblp_best_stc_validation_selected_accuracy": float(dblp_stc_acc),
        "dblp_best_stc_validation_selected_macro": float(dblp_stc_macro),
        "dblp_stc_vs_calibrated_support_accuracy_gap": float(round(acc_gap, 12)),
        "dblp_stc_vs_calibrated_support_macro_gap": float(round(macro_gap, 12)),
        "dblp_stc_vs_calibrated_support_accuracy_gap_validation_selected": float(round(acc_gap, 12)),
        "dblp_stc_vs_calibrated_support_macro_gap_validation_selected": float(round(macro_gap, 12)),
        "dblp_he_sf_cal_nested_accuracy_std": float(nested_stats["accuracy_std"]),
        "dblp_he_sf_cal_nested_macro_std": float(nested_stats["macro_std"]),
        "dblp_he_sf_cal_nested_constraint_rate": float(nested_stats["constraint_rate"]),
        "stc_pareto_non_dominated_vs_calibrated_support": bool(stc_pareto_non_dominated),
        "best_by_validation_selection": {dataset: row.get("method", "") for dataset, row in best_all.items()},
        "best_by_test_oracle": {dataset: row.get("method", "") for dataset, row in best_by_test_oracle(rows, lambda row: str(row.get("method_family")) in (SUPPORT_FAMILIES | STC_FAMILIES)).items()},
        "best_by_pareto": {dataset: row.get("method", "") for dataset, row in _best_by_pareto(pareto).items()},
    }


def _write_decision(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Gate19.2 Decision",
        "",
        f"- decision: {result.get('decision')}",
        f"- gate20_allowed: {result.get('gate20_allowed')}",
        f"- primary_eval_mode: {result.get('primary_eval_mode')}",
        f"- test_oracle_used_for_decision: {result.get('test_oracle_used_for_decision')}",
        "",
        "## DBLP Validation-Selected",
        f"- best HeSF-CAL/support: {result.get('best_calibrated_support_by_validation_selection')}",
        f"- best STC: {result.get('best_stc_by_validation_selection')}",
        f"- accuracy gap STC-support: {result.get('dblp_stc_vs_calibrated_support_accuracy_gap')}",
        f"- macro gap STC-support: {result.get('dblp_stc_vs_calibrated_support_macro_gap')}",
        "",
        "## Diagnostics",
        f"- best STC by test oracle: {result.get('best_stc_by_test_oracle')} (diagnostic only)",
        f"- best STC by Pareto: {result.get('best_stc_by_pareto')} (diagnostic only)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_dir / "gate19_2_raw_rows.csv")
    nested_rows = read_csv(input_dir / "gate19_2_nested_calibration.csv")
    per_class = read_csv(input_dir / "gate19_2_per_class_metrics.csv")
    confusion = read_csv(input_dir / "gate19_2_confusion_matrix_by_method.csv")
    selected = validation_selected_rows(rows)
    pareto = build_gate19_2_pareto(rows)
    by_dataset = []
    best = best_by_validation_selection(rows, lambda row: str(row.get("method_family")) in (SUPPORT_FAMILIES | STC_FAMILIES))
    for dataset, row in sorted(best.items()):
        by_dataset.append(
            {
                "dataset": dataset,
                "method": row.get("method", ""),
                "method_family": row.get("method_family", ""),
                "selected_requested_budget": row.get("selected_requested_budget", row.get("requested_budget", "")),
                "validation_macro_f1": row.get("validation_macro_f1", ""),
                "validation_accuracy": row.get("validation_accuracy", ""),
                "macro_f1": row.get("macro_f1", ""),
                "accuracy": row.get("accuracy", ""),
                "test_oracle_used_for_selection": False,
            }
        )
    write_csv(output_dir / "gate19_2_validation_selected_by_method.csv", selected)
    write_csv(output_dir / "gate19_2_pareto_frontier.csv", pareto)
    write_csv(output_dir / "gate19_2_by_dataset_selected.csv", by_dataset)
    result = summarize_rows(rows, nested_rows=nested_rows, per_class_present=bool(per_class), confusion_present=bool(confusion))
    (output_dir / "gate19_2_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_decision(output_dir / "gate19_2_decision.md", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate19.2 HeSF-CAL support teacher outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate19_2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir or args.input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
