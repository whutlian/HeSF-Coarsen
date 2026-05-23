from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_gate19 import _bool, _float, read_csv


MAIN_FAMILIES = {"hesf_cal", "calibrated_support_baseline"}
ALLOWED_DECISIONS = {
    "CONTINUE_HESF_CAL_TO_GATE21_OFFICIAL_EVAL",
    "CONTINUE_HESF_CAL_MORE_MULTI_SEED",
    "BLOCKED_BY_CALIBRATION_INSTABILITY",
    "BLOCKED_BY_DBLP_ACCURACY_FAILURE",
    "BLOCKED_BY_LEAKAGE_OR_PROTOCOL_ERROR",
    "BLOCKED_BY_LITE_EVALUATOR_CEILING",
}


def _ok(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "success")) == "success" and not _bool(row.get("failed", False)) and not _bool(row.get("method_invalid", False))


def _main_eligible(row: Mapping[str, Any]) -> bool:
    if not _ok(row):
        return False
    if str(row.get("method_family")) not in MAIN_FAMILIES:
        return False
    if _bool(row.get("diagnostic_only", False)) or not _bool(row.get("eligible_for_main_decision", True)):
        return False
    if str(row.get("primary_eval_mode")) != "compressed_projected":
        return False
    if not _bool(row.get("no_test_leakage", True)):
        return False
    if _bool(row.get("calibration_uses_test_labels", False)) or _bool(row.get("selector_uses_test_labels", False)):
        return False
    method = str(row.get("method"))
    if "random" in method.lower() or "Ensemble" in method or method.startswith("STC-"):
        return False
    return True


def _ratio(row: Mapping[str, Any]) -> float:
    return _float(row.get("ratio", row.get("support_ratio", row.get("requested_budget"))))


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _float(row.get("validation_accuracy")),
        _float(row.get("validation_macro_f1")),
        -_float(row.get("total_storage_ratio_vs_full_stc"), 1.0e9),
        -_ratio(row),
    )


def validation_selected_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not _ok(row):
            continue
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        method = str(row.get("method", ""))
        if not dataset or not seed or not method:
            continue
        groups.setdefault((dataset, seed, method), []).append(row)
    out: list[dict[str, Any]] = []
    for (_dataset, _seed, _method), group in sorted(groups.items()):
        best = dict(max(group, key=_selection_key))
        best["selected_ratio"] = _ratio(best)
        best["selection_rule"] = "validation_accuracy_then_validation_macro_then_lower_cost"
        best["test_oracle_used_for_selection"] = False
        out.append(best)
    return out


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    acc = np.asarray([_float(row.get("accuracy")) for row in rows], dtype=np.float64)
    macro = np.asarray([_float(row.get("macro_f1")) for row in rows], dtype=np.float64)
    val_acc = np.asarray([_float(row.get("validation_accuracy")) for row in rows], dtype=np.float64)
    val_macro = np.asarray([_float(row.get("validation_macro_f1")) for row in rows], dtype=np.float64)
    cost = np.asarray([_float(row.get("total_storage_ratio_vs_full_stc")) for row in rows], dtype=np.float64)
    gain_acc = np.asarray([_float(row.get("accuracy")) - _float(row.get("uncalibrated_accuracy")) for row in rows], dtype=np.float64)
    gain_macro = np.asarray([_float(row.get("macro_f1")) - _float(row.get("uncalibrated_macro_f1")) for row in rows], dtype=np.float64)
    return {
        "seed_count": int(len(rows)),
        "accuracy_mean": float(np.mean(acc)) if len(acc) else 0.0,
        "accuracy_std": float(np.std(acc)) if len(acc) else 0.0,
        "macro_mean": float(np.mean(macro)) if len(macro) else 0.0,
        "macro_std": float(np.std(macro)) if len(macro) else 0.0,
        "validation_accuracy_mean": float(np.mean(val_acc)) if len(val_acc) else 0.0,
        "validation_macro_mean": float(np.mean(val_macro)) if len(val_macro) else 0.0,
        "cost_mean": float(np.mean(cost)) if len(cost) else 0.0,
        "calibration_gain_accuracy_mean": float(np.mean(gain_acc)) if len(gain_acc) else 0.0,
        "calibration_gain_macro_mean": float(np.mean(gain_macro)) if len(gain_macro) else 0.0,
    }


def build_gate20_pareto(rows: Sequence[Mapping[str, Any]], *, main_only: bool = False) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if _ok(row)
        and (not main_only or _main_eligible(row))
        and not _bool(row.get("method_invalid", False))
        and str(row.get("dataset", "")) != ""
    ]
    out: list[dict[str, Any]] = []
    for row in candidates:
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed", ""))
        cost = _float(row.get("total_storage_ratio_vs_full_stc"))
        acc = _float(row.get("accuracy"))
        macro = _float(row.get("macro_f1"))
        dominated_by = ""
        for other in candidates:
            if other is row or str(other.get("dataset")) != dataset or str(other.get("seed", "")) != seed:
                continue
            other_cost = _float(other.get("total_storage_ratio_vs_full_stc"))
            other_acc = _float(other.get("accuracy"))
            other_macro = _float(other.get("macro_f1"))
            if (
                other_cost <= cost + 1.0e-12
                and other_acc >= acc - 1.0e-12
                and other_macro >= macro - 1.0e-12
                and (other_cost < cost - 1.0e-12 or other_acc > acc + 1.0e-12 or other_macro > macro + 1.0e-12)
            ):
                dominated_by = str(other.get("method"))
                break
        out.append(
            {
                "dataset": dataset,
                "seed": int(_float(seed, 0.0)),
                "method": row.get("method", ""),
                "method_family": row.get("method_family", ""),
                "ratio": _ratio(row),
                "total_storage_ratio_vs_full_stc": cost,
                "total_storage_ratio_vs_full_graph": _float(row.get("total_storage_ratio_vs_full_graph")),
                "accuracy": acc,
                "macro_f1": macro,
                "pareto_dominated_by": dominated_by,
                "main_only": bool(main_only),
            }
        )
    return sorted(out, key=lambda item: (str(item["dataset"]), int(item["seed"]), float(item["total_storage_ratio_vs_full_stc"]), str(item["method"])))


def exact_ratio_comparison(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}}):
        for method in sorted({str(row.get("method")) for row in rows if str(row.get("dataset")) == dataset}):
            for ratio in sorted({_ratio(row) for row in rows if str(row.get("dataset")) == dataset and str(row.get("method")) == method}):
                group = [row for row in rows if str(row.get("dataset")) == dataset and str(row.get("method")) == method and abs(_ratio(row) - ratio) < 1.0e-12 and _ok(row)]
                if not group:
                    continue
                stats = _aggregate(group)
                out.append({"dataset": dataset, "method": method, "ratio": ratio, **stats})
    return out


def _nested_lookup(nested_rows: Sequence[Mapping[str, Any]], dataset: str, method: str, ratio: float) -> Mapping[str, Any]:
    candidates = [
        row
        for row in nested_rows
        if str(row.get("dataset")) == dataset and str(row.get("method")) == method and abs(_ratio(row) - float(ratio)) < 1.0e-12
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: (_float(row.get("nested_accuracy_mean")), -_float(row.get("nested_accuracy_std"), 1.0e9)))


def _primary_best_rows(rows: Sequence[Mapping[str, Any]], *, dataset: str = "DBLP", method: str = "HeSF-CAL-best-support", ratio: float = 0.30) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if _main_eligible(row)
        and str(row.get("dataset")) == dataset
        and str(row.get("method")) == method
        and abs(_ratio(row) - float(ratio)) < 1.0e-12
    ]


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    nested_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    per_class_present: bool,
    confusion_present: bool,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    selected = validation_selected_rows(rows)
    main_selected = [row for row in selected if _main_eligible(row)]
    primary_rows = _primary_best_rows(rows)
    if not primary_rows:
        primary_rows = [row for row in main_selected if str(row.get("dataset")) == "DBLP" and str(row.get("method")) == "HeSF-CAL-best-support"]
    primary_stats = _aggregate(primary_rows)
    best_method = "HeSF-CAL-best-support" if primary_rows else (str(main_selected[0].get("method")) if main_selected else "none")
    best_ratio = 0.30 if primary_rows else (_ratio(main_selected[0]) if main_selected else 0.0)
    nested = _nested_lookup(nested_rows, "DBLP", best_method, best_ratio)
    method_names = {str(row.get("method")) for row in rows if _ok(row)}
    no_test_leakage = all(
        _bool(row.get("no_test_leakage", True))
        and not _bool(row.get("calibration_uses_test_labels", False))
        and not _bool(row.get("selector_uses_test_labels", False))
        for row in rows
        if _ok(row)
    )
    primary_eval_ok = all(str(row.get("primary_eval_mode")) == "compressed_projected" for row in rows if _main_eligible(row))
    typedhash_included = "TypedHash-ChebHeat-support-only" in method_names and "HeSF-CAL-TypedHash" in method_names
    quality_bad = any(
        str(row.get("dataset")) == "DBLP"
        and str(row.get("method")) == best_method
        and abs(_ratio(row) - best_ratio) < 1.0e-12
        and (_float(row.get("delta_nll")) > 1.0 or _float(row.get("delta_ece")) > 0.2)
        for row in quality_rows
    )
    nested_acc_std = _float(nested.get("nested_accuracy_std", primary_stats.get("accuracy_std", 0.0)))
    nested_macro_std = _float(nested.get("nested_macro_std", primary_stats.get("macro_std", 0.0)))
    constraint_rate = _float(nested.get("calibration_constraint_satisfied_rate", 0.0 if nested_rows else 1.0))
    if not no_test_leakage or not primary_eval_ok or not typedhash_included:
        decision = "BLOCKED_BY_LEAKAGE_OR_PROTOCOL_ERROR"
    elif primary_stats["accuracy_mean"] < 0.88:
        decision = "BLOCKED_BY_DBLP_ACCURACY_FAILURE"
    elif nested_acc_std > 0.02 or nested_macro_std > 0.02 or constraint_rate < 0.8:
        decision = "BLOCKED_BY_CALIBRATION_INSTABILITY"
    elif primary_stats["accuracy_mean"] >= 0.89 and primary_stats["macro_mean"] >= 0.885 and primary_stats["calibration_gain_accuracy_mean"] > 0.015 and primary_stats["calibration_gain_macro_mean"] > 0.015 and nested_acc_std <= 0.015 and nested_macro_std <= 0.015 and not quality_bad:
        decision = "CONTINUE_HESF_CAL_TO_GATE21_OFFICIAL_EVAL"
    else:
        decision = "CONTINUE_HESF_CAL_MORE_MULTI_SEED"
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"invalid Gate20 decision: {decision}")
    return {
        "stage": "Gate20-CAL",
        "decision": decision,
        "gate21_allowed": decision == "CONTINUE_HESF_CAL_TO_GATE21_OFFICIAL_EVAL",
        "primary_eval_mode": "compressed_projected" if primary_eval_ok else "mixed_or_invalid",
        "no_test_leakage": bool(no_test_leakage),
        "typedhash_included": bool(typedhash_included),
        "test_oracle_used_for_decision": False,
        "acm_used_for_success_evidence": False,
        "imdb_used_for_success_evidence": False,
        "per_class_confusion_present": bool(per_class_present and confusion_present),
        "best_method": best_method,
        "best_method_ratio": float(best_ratio),
        "best_validation_selected_method": best_method,
        "dblp_best_support_accuracy_mean": primary_stats["accuracy_mean"],
        "dblp_best_support_accuracy_std": primary_stats["accuracy_std"],
        "dblp_best_support_macro_mean": primary_stats["macro_mean"],
        "dblp_best_support_macro_std": primary_stats["macro_std"],
        "dblp_calibration_gain_accuracy_mean": primary_stats["calibration_gain_accuracy_mean"],
        "dblp_calibration_gain_macro_mean": primary_stats["calibration_gain_macro_mean"],
        "dblp_total_storage_ratio_vs_full_stc_mean": primary_stats["cost_mean"],
        "best_method_nested_accuracy_mean": _float(nested.get("nested_accuracy_mean", primary_stats["accuracy_mean"])),
        "best_method_nested_accuracy_std": nested_acc_std,
        "best_method_nested_macro_mean": _float(nested.get("nested_macro_mean", primary_stats["macro_mean"])),
        "best_method_nested_macro_std": nested_macro_std,
        "best_method_constraint_satisfied_rate": constraint_rate,
        "best_method_temperature_mean": _float(nested.get("temperature_mean")),
        "best_method_temperature_std": _float(nested.get("temperature_std")),
        "best_method_class_bias_l2_mean": _float(nested.get("class_bias_l2_mean")),
        "best_method_class_bias_l2_std": _float(nested.get("class_bias_l2_std")),
        "calibration_quality_pathological": bool(quality_bad),
        "main_seed_count": int(primary_stats["seed_count"]),
    }


def _write_decision(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Gate20-CAL Decision",
        "",
        f"- decision: {result.get('decision')}",
        f"- gate21_allowed: {result.get('gate21_allowed')}",
        f"- best_method: {result.get('best_method')} @ {result.get('best_method_ratio')}",
        f"- DBLP accuracy mean/std: {result.get('dblp_best_support_accuracy_mean')} / {result.get('dblp_best_support_accuracy_std')}",
        f"- DBLP macro mean/std: {result.get('dblp_best_support_macro_mean')} / {result.get('dblp_best_support_macro_std')}",
        f"- calibration gain accuracy/macro: {result.get('dblp_calibration_gain_accuracy_mean')} / {result.get('dblp_calibration_gain_macro_mean')}",
        f"- test_oracle_used_for_decision: {result.get('test_oracle_used_for_decision')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_dir / "gate20_cal_raw_rows.csv")
    nested_rows = read_csv(input_dir / "gate20_cal_nested_calibration.csv")
    quality_rows = read_csv(input_dir / "gate20_cal_calibration_quality.csv")
    per_class_rows = read_csv(input_dir / "gate20_cal_per_class_metrics.csv")
    confusion_rows = read_csv(input_dir / "gate20_cal_confusion_matrix_by_method.csv")
    selected = validation_selected_rows(rows)
    exact = exact_ratio_comparison(rows)
    pareto_all = build_gate20_pareto(rows)
    pareto_main = build_gate20_pareto(rows, main_only=True)
    write_csv(output_dir / "gate20_cal_validation_selected_by_method.csv", selected)
    write_csv(output_dir / "gate20_cal_exact_ratio_comparison.csv", exact)
    write_csv(output_dir / "gate20_cal_pareto_frontier.csv", pareto_all)
    write_csv(output_dir / "pareto_frontier_all_methods.csv", pareto_all)
    write_csv(output_dir / "pareto_frontier_main_methods_only.csv", pareto_main)
    by_dataset = []
    for dataset in sorted({str(row.get("dataset")) for row in selected if row.get("dataset") not in {"", None}}):
        candidates = [row for row in selected if str(row.get("dataset")) == dataset and _main_eligible(row)]
        if candidates:
            best = max(candidates, key=_selection_key)
            by_dataset.append(
                {
                    "dataset": dataset,
                    "method": best.get("method", ""),
                    "ratio": _ratio(best),
                    "validation_accuracy": best.get("validation_accuracy", ""),
                    "validation_macro_f1": best.get("validation_macro_f1", ""),
                    "accuracy": best.get("accuracy", ""),
                    "macro_f1": best.get("macro_f1", ""),
                    "test_oracle_used_for_selection": False,
                }
            )
    write_csv(output_dir / "gate20_cal_by_dataset_selected.csv", by_dataset)
    stability_rows = exact
    write_csv(diag_dir / "gate20_cal_nested_stability_by_method.csv", stability_rows)
    seedwise = [
        row
        for row in selected
        if str(row.get("dataset")) == "DBLP" and str(row.get("method")) in {"HeSF-CAL-best-support", "HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash"}
    ]
    write_csv(diag_dir / "gate20_cal_seedwise_dblp_summary.csv", seedwise)
    result = summarize_rows(rows, nested_rows=nested_rows, quality_rows=quality_rows, per_class_present=bool(per_class_rows), confusion_present=bool(confusion_rows))
    (output_dir / "gate20_cal_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_decision(output_dir / "gate20_cal_decision.md", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate20-CAL multiseed results.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate20_cal"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir or args.input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
