from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_gate19 import _bool, _float, read_csv, validation_selected


ELIGIBLE_FAMILIES = {
    "support_baseline",
    "calibrated_support_baseline",
    "full_stc_reference",
    "stc_compressed",
}
MAIN_STC_METHODS = {
    "STC-feature-cache-quantized-int8",
    "STC-feature-cache-quantized-fp16",
    "STC-feature-cache-MLP-compressed-logit-calibrated",
    "STC-path-channel-hard-gate-logit-calibrated",
}


def _eligible_for_pareto(row: Mapping[str, Any]) -> bool:
    if str(row.get("status", "success")) != "success":
        return False
    if _bool(row.get("diagnostic_only", False)) or _bool(row.get("method_invalid", False)):
        return False
    family = str(row.get("method_family", ""))
    if family not in ELIGIBLE_FAMILIES:
        return False
    method = str(row.get("method", ""))
    if method.startswith("ClusterGate-") or method.startswith("HeSF-SS-"):
        return False
    if method == "STC-feature-cache-true-distill":
        return False
    return _bool(row.get("eligible_for_main_decision", True))


def build_gate19_1_pareto(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if _eligible_for_pareto(row)]
    out: list[dict[str, Any]] = []
    for row in eligible:
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed"))
        cost = _float(row.get("total_storage_ratio_vs_full_stc"), 0.0)
        macro = _float(row.get("macro_f1"))
        accuracy = _float(row.get("accuracy"))
        dominated = False
        for other in eligible:
            if other is row or str(other.get("dataset")) != dataset or str(other.get("seed")) != seed:
                continue
            other_cost = _float(other.get("total_storage_ratio_vs_full_stc"), 0.0)
            other_macro = _float(other.get("macro_f1"))
            other_accuracy = _float(other.get("accuracy"))
            if (
                other_cost <= cost + 1.0e-12
                and other_macro >= macro - 1.0e-12
                and other_accuracy >= accuracy - 1.0e-12
                and (other_cost < cost - 1.0e-12 or other_macro > macro + 1.0e-12 or other_accuracy > accuracy + 1.0e-12)
            ):
                dominated = True
                break
        if not dominated:
            out.append(
                {
                    "dataset": dataset,
                    "seed": int(_float(seed, 0.0)),
                    "method": row.get("method"),
                    "method_family": row.get("method_family", ""),
                    "requested_budget": row.get("requested_budget", ""),
                    "total_storage_ratio_vs_full_stc": cost,
                    "total_storage_ratio_vs_full_graph": _float(row.get("total_storage_ratio_vs_full_graph")),
                    "support_node_ratio": _float(row.get("support_node_ratio")),
                    "support_edge_ratio": _float(row.get("support_edge_ratio")),
                    "feature_cache_size_ratio": _float(row.get("feature_cache_size_ratio")),
                    "path_channel_count_ratio": _float(row.get("path_channel_count_ratio")),
                    "macro_f1": macro,
                    "accuracy": accuracy,
                    "validation_macro_f1": _float(row.get("validation_macro_f1")),
                    "validation_accuracy": _float(row.get("validation_accuracy")),
                    "cost_axis_used": "total_storage_ratio_vs_full_stc",
                    "pareto_dominated": False,
                    "primary_eval_mode": row.get("primary_eval_mode", "compressed_projected"),
                    "no_test_leakage": _bool(row.get("no_test_leakage", True)),
                }
            )
    return sorted(out, key=lambda item: (str(item["dataset"]), float(item["total_storage_ratio_vs_full_stc"]), -float(item["accuracy"]), str(item["method"])))


def _best_by_dataset(rows: Sequence[Mapping[str, Any]], predicate, metric: str = "accuracy") -> dict[str, Mapping[str, Any]]:
    datasets = sorted({str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}})
    out: dict[str, Mapping[str, Any]] = {}
    for dataset in datasets:
        candidates = [
            row
            for row in rows
            if str(row.get("dataset")) == dataset
            and predicate(row)
            and str(row.get("status", "success")) == "success"
            and not _bool(row.get("diagnostic_only", False))
            and not _bool(row.get("method_invalid", False))
        ]
        if candidates:
            out[dataset] = max(candidates, key=lambda item: (_float(item.get(metric)), _float(item.get("macro_f1")), -_float(item.get("total_storage_ratio_vs_full_stc"), 1.0e9)))
    return out


def _gap_dict(stc: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]], metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for dataset, row in stc.items():
        if dataset in support:
            out[dataset] = float(_float(row.get(metric)) - _float(support[dataset].get(metric)))
    return out


def _write_decision_md(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Gate19.1 Decision",
        "",
        f"- decision: {result.get('decision')}",
        f"- gate20_allowed: {result.get('gate20_allowed')}",
        f"- gate20_blocker: {result.get('gate20_blocker')}",
        "",
        "## DBLP Primary",
        f"- best calibrated support: {result.get('best_calibrated_support_baseline_by_dataset', {}).get('DBLP', '')}",
        f"- best compressed STC: {result.get('best_compressed_stc_by_dataset', {}).get('DBLP', '')}",
        f"- accuracy gap STC-support: {result.get('dblp_best_stc_accuracy_gap_vs_calibrated_support')}",
        f"- macro gap STC-support: {result.get('dblp_best_stc_macro_gap_vs_calibrated_support')}",
        "",
        "## Notes",
        "- ACM is sanity-only and not used as success evidence.",
        "- Full-STC-MLP is reported as a reference, not as a universal ceiling.",
        "- Pareto uses total_storage_ratio_vs_full_stc and includes formal calibrated support baselines.",
        "",
        "## Failure Reasons",
        json.dumps(result.get("failure_reasons", []), indent=2, sort_keys=True),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, per_class_present: bool, confusion_present: bool) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    pareto = build_gate19_1_pareto(rows)
    calibrated_support = _best_by_dataset(rows, lambda item: str(item.get("method_family")) == "calibrated_support_baseline")
    stc = _best_by_dataset(rows, lambda item: str(item.get("method")) in MAIN_STC_METHODS)
    full_refs = _best_by_dataset(rows, lambda item: str(item.get("method_family")) == "full_stc_reference")
    typedhash_included = any(str(row.get("method")) == "TypedHash-ChebHeat-support-only" for row in rows)
    primary_modes = {str(row.get("primary_eval_mode", "")) for row in rows if str(row.get("status", "success")) == "success"}
    primary_eval_mode = "compressed_projected" if primary_modes in ({"compressed_projected"}, set()) else "mixed"
    no_test_leakage = all(_bool(row.get("no_test_leakage", True)) for row in rows if str(row.get("status", "success")) == "success")
    calibrated_support_included = bool(calibrated_support)
    nested_rows = [row for row in rows if str(row.get("method_family")) == "calibrated_support_baseline"]
    required_nested = [
        row
        for row in nested_rows
        if str(row.get("dataset")) == "DBLP" and str(row.get("method")) == "H6-no-spec-support-only-logit-calibrated"
    ]
    nested_present = bool(required_nested)
    nested_pass = bool(nested_present and any(_bool(row.get("nested_calibration_pass", False)) for row in required_nested))
    pareto_methods_by_dataset: dict[str, set[str]] = {}
    for row in pareto:
        pareto_methods_by_dataset.setdefault(str(row["dataset"]), set()).add(str(row["method"]))
    stc_pareto = {dataset: str(row.get("method")) in pareto_methods_by_dataset.get(dataset, set()) for dataset, row in stc.items()}
    acc_gaps = _gap_dict(stc, calibrated_support, "accuracy")
    macro_gaps = _gap_dict(stc, calibrated_support, "macro_f1")
    dblp_acc_gap = acc_gaps.get("DBLP", 0.0)
    dblp_macro_gap = macro_gaps.get("DBLP", 0.0)
    failure_reasons: list[str] = []
    decision = "CONTINUE_TO_GATE20_MULTI_SEED_STC"
    gate20_allowed = True
    gate20_blocker = ""
    if not calibrated_support_included:
        decision = "FAIL_CALIBRATED_SUPPORT_BASELINES_MISSING"
        failure_reasons.append("calibrated_support_baselines_missing")
    elif not nested_pass:
        decision = "FAIL_NESTED_CALIBRATION_AUDIT"
        failure_reasons.append("nested_calibration_missing_or_failed")
    elif not (per_class_present and confusion_present):
        decision = "FAIL_PER_CLASS_CONFUSION_MISSING"
        failure_reasons.append("per_class_or_confusion_missing")
    elif not no_test_leakage:
        decision = "FAIL_TEST_LEAKAGE_DETECTED"
        failure_reasons.append("test_leakage_detected")
    elif primary_eval_mode != "compressed_projected":
        decision = "FAIL_PRIMARY_EVAL_MODE_INVALID"
        failure_reasons.append("primary_eval_mode_not_compressed_projected")
    elif not typedhash_included:
        decision = "FAIL_TYPEDHASH_MISSING"
        failure_reasons.append("typedhash_missing")
    elif dblp_acc_gap < -0.005 or dblp_macro_gap < -0.005:
        decision = "GATE20_BLOCKED_BY_CALIBRATED_SUPPORT_BASELINE"
        gate20_blocker = "calibrated_support_baseline_beats_best_stc_on_dblp"
        failure_reasons.append(gate20_blocker)
    elif not stc_pareto.get("DBLP", False):
        decision = "GATE20_BLOCKED_BY_CALIBRATED_SUPPORT_BASELINE"
        gate20_blocker = "best_dblp_stc_not_pareto_non_dominated"
        failure_reasons.append(gate20_blocker)
    gate20_allowed = decision == "CONTINUE_TO_GATE20_MULTI_SEED_STC"
    if not gate20_blocker and not gate20_allowed:
        gate20_blocker = decision.lower()
    return {
        "stage": "Gate19.1",
        "decision": decision,
        "gate20_allowed": bool(gate20_allowed),
        "primary_eval_mode": primary_eval_mode,
        "no_test_leakage": bool(no_test_leakage),
        "typedhash_included": bool(typedhash_included),
        "calibrated_support_baselines_included": bool(calibrated_support_included),
        "nested_calibration_audit_pass": bool(nested_pass),
        "per_class_confusion_present": bool(per_class_present and confusion_present),
        "best_full_stc_reference_by_dataset": {dataset: row.get("method") for dataset, row in full_refs.items()},
        "best_calibrated_support_baseline_by_dataset": {dataset: row.get("method") for dataset, row in calibrated_support.items()},
        "best_calibrated_support_accuracy_by_dataset": {dataset: _float(row.get("accuracy")) for dataset, row in calibrated_support.items()},
        "best_calibrated_support_macro_by_dataset": {dataset: _float(row.get("macro_f1")) for dataset, row in calibrated_support.items()},
        "best_compressed_stc_by_dataset": {dataset: row.get("method") for dataset, row in stc.items()},
        "best_compressed_stc_accuracy_by_dataset": {dataset: _float(row.get("accuracy")) for dataset, row in stc.items()},
        "best_compressed_stc_macro_by_dataset": {dataset: _float(row.get("macro_f1")) for dataset, row in stc.items()},
        "best_stc_gap_vs_best_calibrated_support_accuracy_by_dataset": acc_gaps,
        "best_stc_gap_vs_best_calibrated_support_macro_by_dataset": macro_gaps,
        "best_stc_pareto_non_dominated_by_dataset": stc_pareto,
        "dblp_best_stc_accuracy_gap_vs_calibrated_support": float(round(dblp_acc_gap, 12)),
        "dblp_best_stc_macro_gap_vs_calibrated_support": float(round(dblp_macro_gap, 12)),
        "gate20_blocker": gate20_blocker,
        "failure_reasons": failure_reasons,
        "pareto_non_dominated_count": int(len(pareto)),
        "acm_used_for_success_evidence": False,
    }


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_dir / "gate19_1_raw_rows.csv")
    per_class_present = (input_dir / "gate19_1_per_class_metrics.csv").exists() and bool(read_csv(input_dir / "gate19_1_per_class_metrics.csv"))
    confusion_present = (input_dir / "gate19_1_confusion_matrix_by_method.csv").exists() and bool(read_csv(input_dir / "gate19_1_confusion_matrix_by_method.csv"))
    pareto = build_gate19_1_pareto(rows)
    selected = validation_selected(rows)
    by_dataset_rows = []
    for dataset, row in sorted(_best_by_dataset(rows, lambda item: str(item.get("method_family")) in {"calibrated_support_baseline", "stc_compressed"}).items()):
        by_dataset_rows.append(
            {
                "dataset": dataset,
                "method": row.get("method"),
                "method_family": row.get("method_family"),
                "total_storage_ratio_vs_full_stc": row.get("total_storage_ratio_vs_full_stc", ""),
                "macro_f1": row.get("macro_f1", ""),
                "accuracy": row.get("accuracy", ""),
                "validation_macro_f1": row.get("validation_macro_f1", ""),
                "validation_accuracy": row.get("validation_accuracy", ""),
            }
        )
    write_csv(output_dir / "gate19_1_validation_selected_by_method.csv", selected)
    write_csv(output_dir / "gate19_1_pareto_frontier.csv", pareto)
    write_csv(output_dir / "gate19_1_by_dataset_selected.csv", by_dataset_rows)
    result = summarize_rows(rows, per_class_present=per_class_present, confusion_present=confusion_present)
    (output_dir / "gate19_1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_decision_md(output_dir / "gate19_1_decision.md", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Gate19.1 calibrated support audit outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate19_1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir or args.input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
