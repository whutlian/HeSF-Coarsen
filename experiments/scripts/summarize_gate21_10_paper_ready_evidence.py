from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_10_common import (
    DEFAULT_GATE21_9_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SUMMARY_FILES,
    bool_value,
    ensure_layout,
    parse_bool_arg,
    read_csv,
    write_summary_csv,
    write_summary_json,
)
from hesf_coarsen.eval.official.gate21_10_decision import REQUIRED_DECISION_FLAGS, decision_status, gate21_10_decision


def summarize(
    input_root: Path,
    output_root: Path | None = None,
    *,
    gate21_9_root: Path = DEFAULT_GATE21_9_ROOT,
    fail_on_missing_required: bool = False,
) -> dict[str, Any]:
    root = Path(input_root)
    paths = ensure_layout(root)
    summary = Path(output_root or paths["summary"])
    summary.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(paths)
    decision = gate21_10_decision(
        official_rows=tables["official_main"],
        auto_selector_rows=tables["auto_selector"],
        external_tp_rows=tables["external_tp_task"],
        freehgc_standard_rows=tables["freehgc_standard_task"],
        freehgc_env_rows=tables["freehgc_standard_env"],
        freehgc_tp_rows=tables["freehgc_tp_audit"],
        metapath_rows=tables["metapath"],
        cache_rows=tables["cache"],
        feature_ablation_rows=tables["feature_ablation_task"],
        adapter_rows=tables["adapter_task"],
        storage_denominator_rows=tables["storage_denominator"],
        system_cost_rows=tables["system_workload"],
        coverage_rows=tables["coverage"],
        cross_dataset_rows=tables["cross_dataset_task"],
    )
    status = decision_status(decision)
    blockers = [name for name in REQUIRED_DECISION_FLAGS if not bool_value(decision.get(name))]
    payload: dict[str, Any] = {
        **decision,
        "paper_ready_status": status,
        "blocking_issues": blockers,
        "counts": {name: len(rows) for name, rows in tables.items()},
        "input_root": str(root),
        "gate21_9_root": str(gate21_9_root),
    }
    payload["paper_safe_claims"] = _safe_claims(decision)
    payload["paper_unsafe_claims"] = _unsafe_claims(decision)

    _write_summary_tables(summary, tables, payload)
    _write_decision_md(summary / "gate21_10_decision.md", payload)
    _write_requirement_checklist(summary / "gate21_10_requirement_checklist.md", payload, root)
    _write_prompt_completion_checklist(summary / "gate21_10_prompt_completion_checklist.md", payload, summary)

    missing = [name for name in SUMMARY_FILES if not (summary / name).exists()]
    if missing and fail_on_missing_required:
        raise RuntimeError(f"missing Gate21.10 summary files: {missing}")
    return payload


def _load_tables(paths: Mapping[str, Path]) -> dict[str, list[dict[str, str]]]:
    return {
        "official_main": read_csv(paths["official_main"] / "gate21_10_official_main_by_method.csv"),
        "auto_selector": read_csv(paths["auto_selector"] / "gate21_10_auto_selector_by_method.csv"),
        "channel_utility": read_csv(paths["auto_selector"] / "gate21_10_channel_utility.csv"),
        "external_tp_task": read_csv(paths["external_tp"] / "gate21_10_external_tp_task_rows.csv"),
        "external_tp_by_method": read_csv(paths["external_tp"] / "gate21_10_external_tp_by_method.csv"),
        "external_tp_budget": read_csv(paths["external_tp"] / "gate21_10_external_tp_budget_audit.csv"),
        "freehgc_standard_task": read_csv(paths["freehgc_standard"] / "gate21_10_freehgc_standard_task_rows.csv"),
        "freehgc_standard_by_method": read_csv(paths["freehgc_standard"] / "gate21_10_freehgc_standard_by_method.csv"),
        "freehgc_standard_env": read_csv(paths["freehgc_standard"] / "gate21_10_freehgc_standard_env_audit.csv"),
        "freehgc_tp_by_method": read_csv(paths["freehgc_tp"] / "gate21_10_freehgc_tp_by_method.csv"),
        "freehgc_tp_audit": read_csv(paths["freehgc_tp"] / "gate21_10_freehgc_tp_adapter_audit.csv"),
        "metapath": read_csv(paths["metapath_cache"] / "gate21_10_metapath_tensor_audit.csv"),
        "cache": read_csv(paths["metapath_cache"] / "gate21_10_cache_hash_audit.csv"),
        "feature_ablation_task": read_csv(paths["feature_ablation"] / "gate21_10_feature_ablation_task_rows.csv"),
        "feature_ablation_by_method": read_csv(paths["feature_ablation"] / "gate21_10_feature_ablation_by_method.csv"),
        "adapter_task": read_csv(paths["adapter"] / "gate21_10_adapter_task_rows.csv"),
        "adapter_by_method": read_csv(paths["adapter"] / "gate21_10_adapter_by_method.csv"),
        "adapter_package_audit": read_csv(paths["adapter"] / "gate21_10_adapter_package_audit.csv"),
        "storage_artifact": read_csv(paths["storage_system"] / "gate21_10_storage_system_by_artifact.csv"),
        "storage_denominator": read_csv(paths["storage_system"] / "gate21_10_storage_denominator_audit.csv"),
        "system_workload": read_csv(paths["storage_system"] / "gate21_10_system_workload_cost.csv"),
        "cross_dataset_task": read_csv(paths["cross_dataset"] / "gate21_10_cross_dataset_task_rows.csv"),
        "cross_dataset_by_method": read_csv(paths["cross_dataset"] / "gate21_10_cross_dataset_by_method.csv"),
        "coverage": read_csv(paths["audits"] / "gate21_10_coverage_semantic.csv"),
    }


def _write_summary_tables(summary: Path, tables: Mapping[str, list[dict[str, str]]], payload: Mapping[str, Any]) -> None:
    write_summary_json(summary / "gate21_10_decision.json", payload)
    _copy_table(summary / "gate21_10_official_main_by_method.csv", tables["official_main"])
    _copy_table(summary / "gate21_10_auto_selector_by_method.csv", tables["auto_selector"])
    _copy_table(summary / "gate21_10_channel_utility.csv", tables["channel_utility"])
    _copy_table(summary / "gate21_10_external_tp_by_method.csv", tables["external_tp_by_method"])
    _copy_table(summary / "gate21_10_external_tp_task_rows.csv", tables["external_tp_task"])
    _copy_table(summary / "gate21_10_external_tp_budget_audit.csv", tables["external_tp_budget"])
    _copy_table(summary / "gate21_10_freehgc_standard_by_method.csv", tables["freehgc_standard_by_method"])
    _copy_table(summary / "gate21_10_freehgc_standard_env_audit.csv", tables["freehgc_standard_env"])
    _copy_table(summary / "gate21_10_freehgc_tp_by_method.csv", tables["freehgc_tp_by_method"])
    _copy_table(summary / "gate21_10_freehgc_tp_adapter_audit.csv", tables["freehgc_tp_audit"])
    _copy_table(summary / "gate21_10_metapath_tensor_audit.csv", tables["metapath"])
    _copy_table(summary / "gate21_10_cache_hash_audit.csv", tables["cache"])
    _copy_table(summary / "gate21_10_feature_ablation_by_method.csv", tables["feature_ablation_by_method"])
    _copy_table(summary / "gate21_10_feature_ablation_task_rows.csv", tables["feature_ablation_task"])
    _copy_table(summary / "gate21_10_adapter_by_method.csv", tables["adapter_by_method"])
    _copy_table(summary / "gate21_10_adapter_package_audit.csv", tables["adapter_package_audit"])
    _copy_table(summary / "gate21_10_storage_system_by_artifact.csv", tables["storage_artifact"])
    _copy_table(summary / "gate21_10_storage_denominator_audit.csv", tables["storage_denominator"])
    _copy_table(summary / "gate21_10_system_workload_cost.csv", tables["system_workload"])
    _copy_table(summary / "gate21_10_cross_dataset_by_method.csv", tables["cross_dataset_by_method"])
    write_summary_csv(summary / "gate21_10_by_method.csv", _combined_by_method(tables))
    write_summary_csv(summary / "gate21_10_failures.csv", _failure_rows(tables))


def _copy_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_summary_csv(path, rows, fieldnames=["status"] if not rows else None)


def _combined_by_method(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    sources = {
        "official_main": tables["official_main"],
        "auto_selector": tables["auto_selector"],
        "external_tp": tables["external_tp_by_method"],
        "freehgc_standard": tables["freehgc_standard_by_method"],
        "freehgc_tp": tables["freehgc_tp_by_method"],
        "feature_ablation": tables["feature_ablation_by_method"],
        "adapter": tables["adapter_by_method"],
        "storage_system": tables["storage_artifact"],
        "cross_dataset": tables["cross_dataset_by_method"],
    }
    rows: list[dict[str, Any]] = []
    for table_name, table_rows in sources.items():
        for row in table_rows:
            item = dict(row)
            item["summary_table"] = table_name
            rows.append(item)
    return rows


def _failure_rows(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_name, table_rows in tables.items():
        for row in table_rows:
            failure = str(row.get("failure_type", "")).strip()
            success = row.get("success")
            training = row.get("training_executed")
            if failure or (success not in {"", None} and not bool_value(success) and training not in {"", None} and not bool_value(training)):
                item = {
                    "source_table": table_name,
                    "dataset": row.get("dataset", ""),
                    "method": row.get("method", row.get("freehgc_variant", row.get("artifact_name", ""))),
                    "failure_type": failure or "not_ready",
                    "failure_message": row.get("failure_message", row.get("failed_reason", "")),
                    "eligible_for_decision": row.get("eligible_for_decision", ""),
                }
                rows.append(item)
    return rows


def _write_decision_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.10 Decision",
        "",
        f"- paper_ready_status: `{payload.get('paper_ready_status', '')}`",
        f"- ICDE_EVIDENCE_READY: `{payload.get('ICDE_EVIDENCE_READY', False)}`",
        "",
        "## Flags",
    ]
    for name in REQUIRED_DECISION_FLAGS:
        lines.append(f"- [{'x' if bool_value(payload.get(name)) else ' '}] `{name}`")
    lines.append("")
    lines.append("## Blocking Issues")
    blockers = list(payload.get("blocking_issues", []))
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_requirement_checklist(path: Path, payload: Mapping[str, Any], root: Path) -> None:
    checks = [
        ("P0 Gate21.9 DBLP official anchors preserved", payload.get("OFFICIAL_MAIN_DBLP_APV12_PASS") and payload.get("OFFICIAL_MAIN_DBLP_APV16_PASS")),
        ("P0 protocol eligibility columns emitted", _csv_has_any(root / "summary" / "gate21_10_official_main_by_method.csv", "eligible_for_official_main_table")),
        ("P0 compression denominator fields separated", _csv_has_any(root / "summary" / "gate21_10_official_main_by_method.csv", "official_text_hgb_byte_ratio")),
        ("P0 no test leakage fields emitted for selector", _csv_has_any(root / "summary" / "gate21_10_auto_selector_by_method.csv", "uses_test_metrics_for_selection")),
        ("P1 budgeted planner emitted APV12/APV16 aligned rows", payload.get("AUTO_SELECTOR_DBLP_BUDGET12_ALIGNED") and payload.get("AUTO_SELECTOR_DBLP_BUDGET16_ALIGNED")),
        ("P2 external TP task rows, budget audit, and by-method summary emitted", _exists(root, "summary/gate21_10_external_tp_task_rows.csv") and _exists(root, "summary/gate21_10_external_tp_budget_audit.csv")),
        ("P3 FreeHGC standard env audit and TP hard proof emitted", _exists(root, "summary/gate21_10_freehgc_standard_env_audit.csv") and payload.get("FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY")),
        ("P4 metapath tensor and cache hash audits emitted", _exists(root, "summary/gate21_10_metapath_tensor_audit.csv") and _exists(root, "summary/gate21_10_cache_hash_audit.csv")),
        ("P5 feature ablation task grid emitted", _exists(root, "summary/gate21_10_feature_ablation_task_rows.csv")),
        ("P6 adapter ratio merge and package audit emitted", _exists(root, "summary/gate21_10_adapter_by_method.csv") and _exists(root, "summary/gate21_10_adapter_package_audit.csv")),
        ("P7 storage denominator and system workload cost emitted", payload.get("STORAGE_DENOMINATOR_AUDIT_PASS") and _exists(root, "summary/gate21_10_system_workload_cost.csv")),
        ("P8 semantic coverage rows emitted without overclaim", _exists(root, "audits/gate21_10_coverage_semantic.csv")),
        ("P9 cross-dataset rows emitted", _exists(root, "summary/gate21_10_cross_dataset_by_method.csv")),
        ("P10 required Gate21.10 tests exist", all(Path(name).exists() for name in _required_tests())),
        ("All required summary files emitted", all(_exists(root, f"summary/{name}") for name in SUMMARY_FILES)),
    ]
    _write_checklist(path, "# Gate21.10 Requirement Checklist", checks, payload)


def _write_prompt_completion_checklist(path: Path, payload: Mapping[str, Any], summary: Path) -> None:
    checks = [(f"Decision flag `{name}` evaluated", name in payload) for name in REQUIRED_DECISION_FLAGS]
    checks.extend((f"Required summary file `{name}` exists", (summary / name).exists()) for name in SUMMARY_FILES)
    _write_checklist(path, "# Gate21.10 Prompt Completion Checklist", checks, payload)


def _write_checklist(path: Path, title: str, checks: Sequence[tuple[str, Any]], payload: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{payload.get('paper_ready_status', '')}`", ""]
    for label, passed in checks:
        lines.append(f"- [{'x' if bool_value(passed) else ' '}] {label}")
    lines.append("")
    lines.append("## Blocking Issues")
    blockers = list(payload.get("blocking_issues", []))
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_claims(flags: Mapping[str, Any]) -> list[str]:
    claims = []
    if bool_value(flags.get("AUTO_SELECTOR_DBLP_BUDGET16_ALIGNED")):
        claims.append("DBLP budgeted planner recovers the APV16-style channel plan without test leakage fields.")
    if bool_value(flags.get("FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY")):
        claims.append("FreeHGC-TP is recorded as hard-incompatible with unmodified official HGB/SeHGNN export unless provenance is supplied.")
    if bool_value(flags.get("STORAGE_DENOMINATOR_AUDIT_PASS")):
        claims.append("Storage ratios use explicit denominators and avoid mixed compression metrics.")
    return claims


def _unsafe_claims(flags: Mapping[str, Any]) -> list[str]:
    claims = []
    if not bool_value(flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY")):
        claims.append("Do not claim external TP 5x5 SOTA task superiority.")
    if not bool_value(flags.get("METAPATH_INTROSPECTION_PASS")):
        claims.append("Do not claim official SeHGNN metapath tensors were dumped when only fallback cache audits exist.")
    if not bool_value(flags.get("FEATURE_ABLATION_TASK_RESULTS_READY")):
        claims.append("Do not claim feature ablation task conclusions from shape-only rows.")
    if not bool_value(flags.get("CROSS_DATASET_ACM_TASK_RESULTS_READY")) or not bool_value(flags.get("CROSS_DATASET_IMDB_TASK_RESULTS_READY")):
        claims.append("Do not claim ACM/IMDB task generalization is complete.")
    return claims


def _exists(root: Path, relative: str) -> bool:
    return (root / relative).exists()


def _csv_has_any(path: Path, field: str) -> bool:
    rows = read_csv(path)
    return bool(rows and field in rows[0])


def _required_tests() -> list[str]:
    return [
        "tests/test_gate21_10_decision_flags.py",
        "tests/test_budgeted_channel_planner.py",
        "tests/test_external_tp_budget_matching.py",
        "tests/test_freehgc_tp_adapter_audit.py",
        "tests/test_sehgnn_metapath_tensor_dump.py",
        "tests/test_adapter_package_manifest.py",
        "tests/test_storage_denominator_audit.py",
        "tests/test_system_workload_cost_schema.py",
        "tests/test_feature_ablation_task_schema.py",
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.10 paper-ready evidence.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gate21-9-root", type=Path, default=DEFAULT_GATE21_9_ROOT)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = summarize(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root) if args.output_root else None,
        gate21_9_root=Path(args.gate21_9_root),
        fail_on_missing_required=bool(args.fail_on_missing_required),
    )
    print(json.dumps({"paper_ready_status": payload.get("paper_ready_status"), "blocking_issues": payload.get("blocking_issues", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
