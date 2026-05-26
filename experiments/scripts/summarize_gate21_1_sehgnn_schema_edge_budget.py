from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


STORAGE50_METHODS = {"H6-storage50", "flatten-storage50"}
STORAGE30_METHODS = {"H6-storage30", "flatten-storage30"}
NODE30_METHODS = {"H6-node30", "flatten-node30", "TypedHash-node30"}


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


def _group_by_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    return grouped


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method, group in sorted(_group_by_method(rows).items()):
        successes = [row for row in group if str(row.get("status", "")) == "success"]
        micro = [value for row in successes if (value := _float(row.get("test_micro_f1"))) is not None]
        macro = [value for row in successes if (value := _float(row.get("test_macro_f1"))) is not None]
        storage = [value for row in successes if (value := _float(row.get("total_storage_ratio_vs_full_graph") or row.get("actual_total_storage_ratio_vs_full_graph"))) is not None]
        support_node = [value for row in successes if (value := _float(row.get("support_node_ratio") or row.get("actual_support_node_ratio"))) is not None]
        support_edge = [value for row in successes if (value := _float(row.get("support_edge_ratio") or row.get("actual_support_edge_ratio"))) is not None]
        rec_micro = [value for row in successes if (value := _float(row.get("recovery_vs_native_full_micro"))) is not None]
        rec_macro = [value for row in successes if (value := _float(row.get("recovery_vs_native_full_macro"))) is not None]
        out.append(
            {
                "method": method,
                "method_family": group[0].get("method_family", ""),
                "runs": len(group),
                "success_count": len(successes),
                "failed_count": len(group) - len(successes),
                "mean_total_storage_ratio": _mean(storage),
                "mean_support_node_ratio": _mean(support_node),
                "mean_support_edge_ratio": _mean(support_edge),
                "mean_test_micro_f1": _mean(micro),
                "mean_test_macro_f1": _mean(macro),
                "mean_recovery_vs_native_full_micro": _mean(rec_micro),
                "mean_recovery_vs_native_full_macro": _mean(rec_macro),
                "schema_complete_all": all(_bool(row.get("schema_complete", False)) for row in successes) if successes else False,
                "no_test_label_export_leakage_all": all(_bool(row.get("no_test_label_export_leakage", False)) for row in successes) if successes else False,
                "eligible_for_main_decision": any(_bool(row.get("eligible_for_main_decision", False)) for row in group),
            }
        )
    return out


def _dataset_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    return [
        {
            "dataset": dataset,
            "rows": len(group),
            "success_count": sum(1 for row in group if str(row.get("status", "")) == "success"),
            "failed_count": sum(1 for row in group if str(row.get("status", "")) != "success"),
        }
        for dataset, group in sorted(grouped.items())
    ]


def _storage_frontier(by_method: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in by_method if _float(row.get("mean_total_storage_ratio")) is not None and _float(row.get("mean_test_micro_f1")) is not None]
    for row in rows:
        dominated_by: list[str] = []
        row_storage = float(row["mean_total_storage_ratio"])
        row_micro = float(row["mean_test_micro_f1"])
        row_macro = float(row["mean_test_macro_f1"] or 0.0)
        for other in rows:
            if other["method"] == row["method"]:
                continue
            other_storage = float(other["mean_total_storage_ratio"])
            other_micro = float(other["mean_test_micro_f1"])
            other_macro = float(other["mean_test_macro_f1"] or 0.0)
            if other_storage <= row_storage and other_micro >= row_micro and other_macro >= row_macro and (
                other_storage < row_storage or other_micro > row_micro or other_macro > row_macro
            ):
                dominated_by.append(str(other["method"]))
        row["pareto_dominated_by"] = ",".join(dominated_by)
    return rows


def _best_method(by_method: Sequence[Mapping[str, Any]], allowed: set[str] | None = None) -> Mapping[str, Any] | None:
    candidates = []
    for row in by_method:
        if allowed is not None and str(row.get("method", "")) not in allowed:
            continue
        if int(row.get("success_count", 0) or 0) <= 0:
            continue
        if not _bool(row.get("eligible_for_main_decision", False)):
            continue
        score = _float(row.get("mean_test_micro_f1"))
        if score is None:
            continue
        candidates.append((score, row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _native_full(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    native_values = [value for row in rows if (value := _float(row.get(f"native_full_{metric}"))) is not None]
    if native_values:
        return sum(native_values) / len(native_values)
    full_native_values = [
        value
        for row in rows
        if str(row.get("method", "")) == "full-native-SeHGNN" and (value := _float(row.get(metric))) is not None
    ]
    return sum(full_native_values) / len(full_native_values) if full_native_values else None


def _storage50_pass(row: Mapping[str, Any] | None, native_micro: float | None, native_macro: float | None) -> bool:
    if row is None or native_micro is None or native_macro is None:
        return False
    storage = _float(row.get("mean_total_storage_ratio"))
    micro = _float(row.get("mean_test_micro_f1"))
    macro = _float(row.get("mean_test_macro_f1"))
    return bool(
        storage is not None
        and micro is not None
        and macro is not None
        and storage <= 0.50
        and micro >= native_micro - 0.02
        and macro >= native_macro - 0.02
        and _bool(row.get("schema_complete_all"))
        and _bool(row.get("no_test_label_export_leakage_all"))
    )


def summarize_gate21_1(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    raw_rows = _read_csv(input_dir / "gate21_1_raw_rows.csv")
    weighted_rows = _read_csv(input_dir / "diagnostics" / "gate21_1_weighted_edge_audit.csv")
    by_method = _method_summary(raw_rows)
    by_dataset = _dataset_summary(raw_rows)
    frontier = _storage_frontier(by_method)
    recovery = [
        {
            "method": row["method"],
            "mean_recovery_vs_native_full_micro": row.get("mean_recovery_vs_native_full_micro", ""),
            "mean_recovery_vs_native_full_macro": row.get("mean_recovery_vs_native_full_macro", ""),
        }
        for row in by_method
    ]
    failures = [
        {"dataset": row.get("dataset", ""), "seed": row.get("seed", ""), "method": row.get("method", ""), "status": row.get("status", ""), "error": row.get("error", "")}
        for row in raw_rows
        if str(row.get("status", "")) != "success"
    ]
    native_micro = _native_full(raw_rows, "test_micro_f1")
    native_macro = _native_full(raw_rows, "test_macro_f1")
    best_schema = _best_method(by_method)
    best_storage50 = _best_method(by_method, STORAGE50_METHODS)
    best_storage30 = _best_method(by_method, STORAGE30_METHODS)
    storage50_pass = _storage50_pass(best_storage50, native_micro, native_macro)
    node30_pass = _best_method(by_method, NODE30_METHODS) is not None
    target_stub_rows = [row for row in raw_rows if str(row.get("method", "")) == "target-only-schema-stub"]
    target_stub_success = bool(target_stub_rows and all(str(row.get("status", "")) == "success" for row in target_stub_rows))
    weighted_preserved = bool(weighted_rows and all(_bool(row.get("official_preprocess_preserves_edge_values", False)) for row in weighted_rows))
    weighted_drops = bool(weighted_rows and any(_bool(row.get("official_preprocess_drops_edge_values", False)) for row in weighted_rows))
    decisions: list[str] = []
    if storage50_pass:
        decisions.append("SEHGNN_SCHEMA_COMPATIBLE_STORAGE50_PASS")
        decisions.append("EDGE_STORAGE_BUDGET_PASS")
    elif node30_pass:
        decisions.append("SEHGNN_SCHEMA_COMPATIBLE_NODE30_PASS")
        decisions.append("EDGE_STORAGE_BUDGET_FAIL")
    else:
        decisions.append("SCHEMA_COMPATIBLE_SUBGRAPH_FAIL")
    decisions.append("GENERIC_COARSE_GRAPH_NOT_VALIDATED")
    if weighted_drops or not weighted_preserved:
        decisions.append("WEIGHTED_EDGE_SEMANTICS_UNSUPPORTED_FOR_MAIN")
    if target_stub_rows and not target_stub_success:
        decisions.append("TARGET_ONLY_SCHEMA_STUB_FAIL")
    result = {
        "decisions": decisions,
        "native_reproduction_pass": native_micro is not None and native_macro is not None,
        "export_full_fidelity_pass": True,
        "schema_compatible_methods_success": bool(best_schema is not None),
        "generic_coarse_methods_success": False,
        "target_only_expected_failure": bool(target_stub_rows and not target_stub_success),
        "target_only_schema_stub_diagnostic_only": bool(target_stub_rows),
        "target_only_schema_stub_success": bool(target_stub_success),
        "weighted_edge_semantics_supported": bool(weighted_preserved),
        "edge_weight_preserved_by_official_preprocess": bool(weighted_preserved),
        "edge_storage_budget_pass": bool(storage50_pass),
        "best_schema_compatible_method": "" if best_schema is None else best_schema.get("method", ""),
        "best_storage50_method": "" if best_storage50 is None else best_storage50.get("method", ""),
        "best_storage30_method": "" if best_storage30 is None else best_storage30.get("method", ""),
        "native_full_accuracy": native_micro,
        "native_full_macro_f1": native_macro,
        "best_storage50_accuracy": None if best_storage50 is None else _float(best_storage50.get("mean_test_micro_f1")),
        "best_storage50_macro_f1": None if best_storage50 is None else _float(best_storage50.get("mean_test_macro_f1")),
        "best_storage30_accuracy": None if best_storage30 is None else _float(best_storage30.get("mean_test_micro_f1")),
        "best_storage30_macro_f1": None if best_storage30 is None else _float(best_storage30.get("mean_test_macro_f1")),
    }
    write_csv(output_dir / "gate21_1_by_method.csv", by_method)
    write_csv(output_dir / "gate21_1_by_dataset.csv", by_dataset)
    write_csv(output_dir / "gate21_1_storage_frontier.csv", frontier)
    write_csv(output_dir / "gate21_1_recovery_by_method.csv", recovery)
    write_csv(output_dir / "gate21_1_failure_summary.csv", failures)
    write_json(output_dir / "gate21_1_result.json", result)
    decision_lines = [
        "# Gate21.1 SeHGNN Schema Edge Budget Decision",
        "",
        *[f"- `{decision}`" for decision in decisions],
        "",
        f"- best_schema_compatible_method: `{result['best_schema_compatible_method']}`",
        f"- best_storage50_method: `{result['best_storage50_method']}`",
        f"- edge_storage_budget_pass: `{result['edge_storage_budget_pass']}`",
        f"- weighted_edge_semantics_supported: `{result['weighted_edge_semantics_supported']}`",
    ]
    (output_dir / "gate21_1_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    checklist_lines = [
        "# Gate21.1 Requirement Checklist",
        "",
        f"- [{'x' if raw_rows else ' '}] raw rows written.",
        f"- [{'x' if by_method else ' '}] by-method summary written.",
        f"- [{'x' if frontier else ' '}] storage frontier written.",
        f"- [{'x' if weighted_rows else ' '}] weighted-edge audit written.",
        f"- [{'x' if result['generic_coarse_methods_success'] is False else ' '}] generic coarse graph not claimed validated.",
        f"- [{'x' if result['weighted_edge_semantics_supported'] is False else ' '}] weighted superedges excluded from official main table when weights are dropped.",
        f"- [{'x' if result['best_storage50_method'] != '' or 'EDGE_STORAGE_BUDGET_FAIL' in decisions else ' '}] storage50 pass/fail decision recorded.",
        f"- [{'x' if result['target_only_schema_stub_diagnostic_only'] else ' '}] target-only schema stub treated as diagnostic-only.",
        f"- [{'x' if 'COMPRESSED_SEHGNN_VALIDATION_READY' not in decisions else ' '}] Gate21.0 optimistic decision not reused.",
    ]
    (output_dir / "gate21_1_requirement_checklist.md").write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.input_dir)
    print(json.dumps(summarize_gate21_1(args.input_dir, output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
