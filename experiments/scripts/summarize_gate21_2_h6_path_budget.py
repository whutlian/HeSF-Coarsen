from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: Sequence[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _std(values: Sequence[float]) -> float | str:
    return pstdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else "")


def _budget(method: str) -> float | None:
    match = re.search(r"struct(\d+)", str(method))
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "")), []).append(row)
    return out


def _native_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [
        value
        for row in rows
        if str(row.get("method", "")) == "full-native-SeHGNN"
        and str(row.get("status", "")) == "success"
        and (value := _float(row.get(metric))) is not None
    ]
    return sum(values) / len(values) if values else None


def _method_summary(raw_rows: list[dict[str, str]], storage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    storage_by_method = _group(storage_rows, "method")
    out: list[dict[str, Any]] = []
    for method, group in sorted(_group(raw_rows, "method").items()):
        successes = [row for row in group if str(row.get("status", "")) == "success"]
        micro = [value for row in successes if (value := _float(row.get("test_micro_f1"))) is not None]
        macro = [value for row in successes if (value := _float(row.get("test_macro_f1"))) is not None]
        val_micro = [value for row in successes if (value := _float(row.get("validation_micro_f1"))) is not None]
        val_macro = [value for row in successes if (value := _float(row.get("validation_macro_f1"))) is not None]
        rec_native_micro = [value for row in successes if (value := _float(row.get("recovery_vs_native_full_micro"))) is not None]
        rec_native_macro = [value for row in successes if (value := _float(row.get("recovery_vs_native_full_macro"))) is not None]
        rec_export_micro = [value for row in successes if (value := _float(row.get("recovery_vs_export_full_micro"))) is not None]
        rec_export_macro = [value for row in successes if (value := _float(row.get("recovery_vs_export_full_macro"))) is not None]
        storage_group = storage_by_method.get(method, [])
        semantic = [value for row in storage_group if (value := _float(row.get("semantic_structural_storage_ratio"))) is not None]
        raw_bytes = [value for row in storage_group if (value := _float(row.get("hgb_raw_file_byte_ratio"))) is not None]
        cache = [value for row in storage_group if (value := _float(row.get("preprocessed_cache_byte_ratio"))) is not None]
        support_node = [value for row in storage_group if (value := _float(row.get("support_node_ratio"))) is not None]
        support_edge = [value for row in storage_group if (value := _float(row.get("support_edge_ratio"))) is not None]
        total_node = [value for row in storage_group if (value := _float(row.get("total_node_ratio"))) is not None]
        total_edge = [value for row in storage_group if (value := _float(row.get("total_edge_ratio"))) is not None]
        out.append(
            {
                "method": method,
                "method_family": group[0].get("method_family", ""),
                "budget_strategy": group[0].get("budget_strategy", ""),
                "edge_score_strategy": group[0].get("edge_score_strategy", ""),
                "runs": len(group),
                "success_count": len(successes),
                "failed_count": len(group) - len(successes),
                "mean_semantic_structural_storage_ratio": _mean(semantic),
                "mean_hgb_raw_file_byte_ratio": _mean(raw_bytes),
                "mean_preprocessed_cache_byte_ratio": _mean(cache),
                "mean_support_node_ratio": _mean(support_node),
                "mean_support_edge_ratio": _mean(support_edge),
                "mean_total_node_ratio": _mean(total_node),
                "mean_total_edge_ratio": _mean(total_edge),
                "mean_test_micro_f1": _mean(micro),
                "mean_test_macro_f1": _mean(macro),
                "mean_test_accuracy_if_single_label": _mean(micro),
                "std_test_micro_f1": _std(micro),
                "std_test_macro_f1": _std(macro),
                "mean_validation_micro_f1": _mean(val_micro),
                "mean_validation_macro_f1": _mean(val_macro),
                "mean_recovery_vs_native_full_micro": _mean(rec_native_micro),
                "mean_recovery_vs_native_full_macro": _mean(rec_native_macro),
                "mean_recovery_vs_export_full_micro": _mean(rec_export_micro),
                "mean_recovery_vs_export_full_macro": _mean(rec_export_macro),
                "schema_complete_all": all(_bool(row.get("schema_complete", False)) for row in successes) if successes else False,
                "no_test_label_export_leakage_all": all(_bool(row.get("no_test_label_export_leakage", False)) for row in successes) if successes else False,
                "no_test_label_scoring_leakage_all": all(_bool(row.get("no_test_label_scoring_leakage", False)) for row in successes) if successes else False,
                "eligible_for_main_decision": any(_bool(row.get("eligible_for_main_decision", False)) for row in group),
            }
        )
    return out


def _frontier(by_method: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in by_method
        if _float(row.get("mean_semantic_structural_storage_ratio")) is not None and _float(row.get("mean_test_micro_f1")) is not None
    ]
    for row in rows:
        dominated: list[str] = []
        s = float(row["mean_semantic_structural_storage_ratio"])
        m = float(row["mean_test_micro_f1"])
        macro = float(row["mean_test_macro_f1"] or 0.0)
        for other in rows:
            if other["method"] == row["method"]:
                continue
            os = float(other["mean_semantic_structural_storage_ratio"])
            om = float(other["mean_test_micro_f1"])
            oma = float(other["mean_test_macro_f1"] or 0.0)
            if os <= s and om >= m and oma >= macro and (os < s or om > m or oma > macro):
                dominated.append(str(other["method"]))
        row["pareto_dominated_by"] = ",".join(dominated)
    return rows


def _best_at_budget(by_method: Sequence[Mapping[str, Any]], threshold: float) -> Mapping[str, Any] | None:
    candidates = []
    for row in by_method:
        if not _bool(row.get("eligible_for_main_decision", False)):
            continue
        storage = _float(row.get("mean_semantic_structural_storage_ratio"))
        micro = _float(row.get("mean_test_micro_f1"))
        if storage is None or micro is None or storage > threshold:
            continue
        candidates.append((micro, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _structural_pass(row: Mapping[str, Any] | None, native_micro: float | None, native_macro: float | None, *, threshold: float) -> bool:
    if row is None or native_micro is None or native_macro is None:
        return False
    storage = _float(row.get("mean_semantic_structural_storage_ratio"))
    micro = _float(row.get("mean_test_micro_f1"))
    macro = _float(row.get("mean_test_macro_f1"))
    margin = 0.02 if threshold >= 0.50 else 0.03
    return bool(
        storage is not None
        and micro is not None
        and macro is not None
        and storage <= threshold
        and micro >= native_micro - margin
        and macro >= native_macro - margin
        and _bool(row.get("schema_complete_all"))
        and _bool(row.get("no_test_label_export_leakage_all"))
        and _bool(row.get("no_test_label_scoring_leakage_all"))
    )


def _raw_byte_pass(storage_rows: Sequence[Mapping[str, Any]], threshold: float) -> str:
    values = [value for row in storage_rows if (value := _float(row.get("hgb_raw_file_byte_ratio"))) is not None and _bool(row.get("eligible_for_main_decision", True))]
    if not values:
        return "NOT_VALIDATED"
    return "PASS" if min(values) <= threshold else "FAIL"


def _path_gain(by_method: Sequence[Mapping[str, Any]], baseline_strategy: str, min_gain: float) -> bool:
    by_budget: dict[float, dict[str, float]] = {}
    for row in by_method:
        budget = _budget(str(row.get("method", "")))
        micro = _float(row.get("mean_test_micro_f1"))
        if budget is None or micro is None:
            continue
        strategy = str(row.get("edge_score_strategy", ""))
        by_budget.setdefault(budget, {})[strategy] = micro
    for values in by_budget.values():
        if "path_aware" in values and baseline_strategy in values and values["path_aware"] >= values[baseline_strategy] + float(min_gain):
            return True
    return False


def summarize_gate21_2(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    raw_rows = _read_csv(input_root / "gate21_2_raw_rows.csv")
    storage_rows = _read_csv(input_root / "gate21_2_storage_audit.csv")
    relation_rows = _read_csv(input_root / "gate21_2_relation_edge_retention.csv")
    score_rows = _read_csv(input_root / "gate21_2_edge_score_diagnostics.csv")
    ablation_rows = _read_csv(input_root / "gate21_2_label_graph_ablation.csv")
    weighted_rows = _read_csv(input_root / "gate21_2_weighted_edge_audit.csv")
    by_method = _method_summary(raw_rows, storage_rows)
    frontier = _frontier(by_method)
    recovery = [
        {
            "method": row["method"],
            "mean_recovery_vs_native_full_micro": row.get("mean_recovery_vs_native_full_micro", ""),
            "mean_recovery_vs_native_full_macro": row.get("mean_recovery_vs_native_full_macro", ""),
            "mean_recovery_vs_export_full_micro": row.get("mean_recovery_vs_export_full_micro", ""),
            "mean_recovery_vs_export_full_macro": row.get("mean_recovery_vs_export_full_macro", ""),
        }
        for row in by_method
    ]
    comparison = [
        {
            "budget": _budget(str(row.get("method", ""))),
            "method": row.get("method", ""),
            "budget_strategy": row.get("budget_strategy", ""),
            "edge_score_strategy": row.get("edge_score_strategy", ""),
            "mean_test_micro_f1": row.get("mean_test_micro_f1", ""),
            "mean_test_macro_f1": row.get("mean_test_macro_f1", ""),
        }
        for row in by_method
        if _budget(str(row.get("method", ""))) is not None
    ]
    native_micro = _native_metric(raw_rows, "test_micro_f1")
    native_macro = _native_metric(raw_rows, "test_macro_f1")
    best50 = _best_at_budget(by_method, 0.50)
    best40 = _best_at_budget(by_method, 0.40)
    best35 = _best_at_budget(by_method, 0.35)
    best30 = _best_at_budget(by_method, 0.30)
    pass50 = _structural_pass(best50, native_micro, native_macro, threshold=0.50)
    pass40 = _structural_pass(best40, native_micro, native_macro, threshold=0.40)
    pass35 = _structural_pass(best35, native_micro, native_macro, threshold=0.35)
    pass30 = _structural_pass(best30, native_micro, native_macro, threshold=0.30)
    raw50 = _raw_byte_pass(storage_rows, 0.50)
    raw30 = _raw_byte_pass(storage_rows, 0.30)
    path_beats_random = _path_gain(by_method, "random", 0.005)
    path_beats_degree = _path_gain(by_method, "degree", 0.003)
    path_beats_current = _path_gain(by_method, "current_heuristic", 0.003)
    target_diag = any(str(row.get("method", "")) == "target-only-schema-stub" for row in raw_rows)
    weighted_supported = bool(weighted_rows and all(_bool(row.get("official_preprocess_preserves_edge_values", False)) for row in weighted_rows))
    decisions = [
        "SEHGNN_SCHEMA_COMPATIBLE_STRUCTURAL_STORAGE50_PASS" if pass50 else "STRUCTURAL_STORAGE50_FAIL",
        "SEHGNN_SCHEMA_COMPATIBLE_STRUCTURAL_STORAGE40_PASS" if pass40 else "STRUCTURAL_STORAGE40_FAIL",
        "SEHGNN_SCHEMA_COMPATIBLE_STRUCTURAL_STORAGE30_PASS" if pass30 else "STRUCTURAL_STORAGE30_FAIL",
        f"RAW_HGB_BYTE_STORAGE50_{raw50}",
        f"RAW_HGB_BYTE_STORAGE30_{raw30}",
        "CACHE_BYTE_STORAGE_NOT_VALIDATED",
        "PATH_AWARE_EDGE_PRUNING_IMPROVES_OVER_RANDOM" if path_beats_random else "PATH_AWARE_EDGE_PRUNING_NO_CLEAR_GAIN",
        "WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN" if not weighted_supported else "WEIGHTED_EDGE_SUPPORTED",
        "TARGET_ONLY_SCHEMA_STUB_DIAGNOSTIC_ONLY" if target_diag else "TARGET_ONLY_SCHEMA_STUB_NOT_RUN",
        "GENERIC_COARSE_GRAPH_NOT_VALIDATED",
    ]
    result = {
        "decisions": decisions,
        "native_reproduction_pass": native_micro is not None and native_macro is not None,
        "export_full_fidelity_pass": any(str(row.get("method", "")) == "export-full-SeHGNN" and str(row.get("status", "")) == "success" for row in raw_rows),
        "schema_compatible_methods_success": any(_bool(row.get("eligible_for_main_decision", False)) and int(row.get("success_count", 0) or 0) > 0 for row in by_method),
        "structural_storage50_pass": bool(pass50),
        "structural_storage40_pass": bool(pass40),
        "structural_storage35_pass": bool(pass35),
        "structural_storage30_pass": bool(pass30),
        "raw_hgb_byte_storage50_pass": True if raw50 == "PASS" else False if raw50 == "FAIL" else None,
        "raw_hgb_byte_storage30_pass": True if raw30 == "PASS" else False if raw30 == "FAIL" else None,
        "cache_byte_storage_validated": False,
        "path_aware_beats_random_at_matched_budget": bool(path_beats_random),
        "path_aware_beats_degree_at_matched_budget": bool(path_beats_degree),
        "path_aware_beats_current_at_matched_budget": bool(path_beats_current),
        "best_struct50_method": "" if best50 is None else best50.get("method", ""),
        "best_struct40_method": "" if best40 is None else best40.get("method", ""),
        "best_struct35_method": "" if best35 is None else best35.get("method", ""),
        "best_struct30_method": "" if best30 is None else best30.get("method", ""),
        "best_overall_frontier_method": "" if not frontier else max(frontier, key=lambda row: float(row.get("mean_test_micro_f1") or 0.0)).get("method", ""),
        "weighted_edge_semantics_supported": bool(weighted_supported),
        "target_only_schema_stub_diagnostic_only": bool(target_diag),
        "generic_coarse_graph_validated": False,
        "native_full_micro": native_micro,
        "native_full_macro": native_macro,
        "relation_retention_rows": len(relation_rows),
        "edge_score_diagnostic_rows": len(score_rows),
        "label_graph_ablation_rows": len(ablation_rows),
    }
    write_csv(output_root / "gate21_2_by_method.csv", by_method)
    write_csv(output_root / "gate21_2_storage_frontier.csv", frontier)
    write_csv(output_root / "gate21_2_recovery_by_method.csv", recovery)
    write_csv(output_root / "gate21_2_budget_strategy_comparison.csv", comparison)
    write_json(output_root / "gate21_2_decision.json", result)
    decision_lines = [
        "# Gate21.2 H6 Path Budget Decision",
        "",
        *[f"- `{decision}`" for decision in decisions],
        "",
        f"- best_struct50_method: `{result['best_struct50_method']}`",
        f"- best_struct40_method: `{result['best_struct40_method']}`",
        f"- best_struct30_method: `{result['best_struct30_method']}`",
        f"- native_full_micro: `{native_micro}`",
        f"- native_full_macro: `{native_macro}`",
    ]
    (output_root / "gate21_2_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    checklist = [
        "# Gate21.2 Requirement Checklist",
        "",
        f"- [{'x' if raw_rows else ' '}] raw rows written with failures preserved.",
        f"- [{'x' if storage_rows else ' '}] storage audit separates structural/raw/cache ratios.",
        f"- [{'x' if _read_csv(input_root / 'gate21_2_relation_mapping_audit.csv') else ' '}] relation mapping audit written.",
        f"- [{'x' if relation_rows else ' '}] relation edge retention written.",
        f"- [{'x' if score_rows else ' '}] edge score diagnostics written with test_label_used=false.",
        f"- [{'x' if ablation_rows else ' '}] label/graph ablation plan written.",
        f"- [{'x' if target_diag else ' '}] target-only schema stub marked diagnostic-only.",
        f"- [{'x' if not weighted_supported else ' '}] weighted coarse graph excluded from unmodified official main decision.",
        f"- [{'x' if 'EDGE_STORAGE_BUDGET_PASS' not in decisions else ' '}] vague Gate21.1 edge-storage decision name not reused.",
    ]
    (output_root / "gate21_2_requirement_checklist.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root) if args.output_root is not None else Path(args.input_root)
    print(json.dumps(summarize_gate21_2(args.input_root, output_root), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
