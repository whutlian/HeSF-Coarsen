from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_14_common import DEFAULT_OUTPUT_ROOT, SUMMARY_FILES, bool_value, read_csv, read_json, write_payload, write_rows
from hesf_coarsen.eval.official.gate21_14_decision import REQUIRED_DECISION_FLAGS, decision_status, gate21_14_decision


def summarize(*, input_dir: Path, output_dir: Path | None = None, fail_on_missing_required: bool = False) -> dict[str, Any]:
    source_root = Path(input_dir)
    root = Path(output_dir) if output_dir is not None else source_root
    root.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(source_root)
    manifest = read_json(source_root / "gate21_14_manifest.json") or read_json(source_root / "audits" / "gate21_14_manifest.json")
    flags = gate21_14_decision(
        official_rows=tables["official_main"],
        budgeted_selector_rows=tables["budgeted_selector"],
        selector_hash_audit=tables["selector_hash_audit"],
        external_tp_runs=tables["external_tp_runs"],
        external_tp_by_method=tables["external_tp_by_method"],
        freehgc_standard_by_method=tables["freehgc_standard_by_method"],
        freehgc_tp_by_method=tables["freehgc_tp_by_method"],
        freehgc_protocol_audit=tables["freehgc_protocol_audit"],
        freehgc_score_selector_by_method=tables["freehgc_score_selector_by_method"],
        feature_ablation_runs=tables["feature_ablation_runs"],
        feature_ablation_by_method=tables["feature_ablation_by_method"],
        metapath_rows=tables["metapath"],
        cache_assertions=tables["cache"],
        coverage_rows=tables["coverage"],
        adapter_rows=tables["adapter_runs"],
        adapter_audit=tables["adapter_package"],
        system_cost_rows=tables["system_cost_runs"],
        cross_dataset_rows=tables["cross_dataset_runs"],
        pareto_rows=tables["pareto"],
    )
    payload: dict[str, Any] = {
        "gate": "21.14",
        "paper_ready_status": decision_status(flags),
        "manifest": manifest,
        **flags,
    }
    payload["blocking_issues"] = [name for name in REQUIRED_DECISION_FLAGS if not bool_value(payload.get(name))]
    _copy_tables(root, source_root, tables)
    write_payload(root / "gate21_14_manifest.json", manifest)
    write_payload(root / "gate21_14_decision.json", payload)
    _write_decision_md(root / "gate21_14_decision.md", payload)
    write_rows(root / "gate21_14_by_method.csv", _combined_by_method(tables))
    write_rows(root / "gate21_14_failure_audit.csv", _failure_rows(tables))
    _write_checklists(root, payload)
    missing = [name for name in SUMMARY_FILES if not (root / name).exists()]
    payload["missing_summary_files"] = missing
    if missing:
        write_payload(root / "gate21_14_decision.json", payload)
        _write_decision_md(root / "gate21_14_decision.md", payload)
    if missing and fail_on_missing_required:
        raise RuntimeError(f"missing Gate21.14 summary artifacts: {missing}")
    return payload


def _load_tables(root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "official_main": _read(root, "official_main", "gate21_14_official_main_by_method.csv"),
        "budgeted_selector": _read(root, "budgeted_selector", "gate21_14_budgeted_selector_by_method.csv"),
        "selector_hash_audit": _read(root, "budgeted_selector", "gate21_14_selector_hash_audit.csv"),
        "external_tp_runs": _read(root, "external_tp", "gate21_14_external_tp_runs.csv"),
        "external_tp_by_method": _read(root, "external_tp", "gate21_14_external_tp_by_method.csv"),
        "external_tp_budget_audit": _read(root, "external_tp", "gate21_14_external_tp_budget_audit.csv"),
        "freehgc_standard_by_method": _read(root, "freehgc", "gate21_14_freehgc_standard_by_method.csv"),
        "freehgc_tp_by_method": _read(root, "freehgc", "gate21_14_freehgc_tp_by_method.csv"),
        "freehgc_protocol_audit": _read(root, "freehgc", "gate21_14_freehgc_protocol_audit.csv"),
        "freehgc_score_selector_by_method": _read(root, "freehgc", "gate21_14_freehgc_score_selector_by_method.csv"),
        "feature_ablation_runs": _read(root, "feature_ablation", "gate21_14_feature_ablation_runs.csv"),
        "feature_ablation_by_method": _read(root, "feature_ablation", "gate21_14_feature_ablation_by_method.csv"),
        "metapath": _read(root, "metapath_cache", "gate21_14_metapath_tensor_dump.csv"),
        "cache": _read(root, "metapath_cache", "gate21_14_cache_hash_assertions.csv"),
        "coverage": _read(root, "coverage", "gate21_14_coverage_semantic_diagnostics.csv"),
        "adapter_runs": _read(root, "adapter", "gate21_14_adapter_runs.csv"),
        "adapter_by_method": _read(root, "adapter", "gate21_14_adapter_by_method.csv"),
        "adapter_package": _read(root, "adapter", "gate21_14_adapter_package_audit.csv"),
        "system_cost_runs": _read(root, "system_cost", "gate21_14_system_workload_cost_runs.csv"),
        "system_cost_by_method": _read(root, "system_cost", "gate21_14_system_workload_cost_by_method.csv"),
        "cross_dataset_runs": _read(root, "cross_dataset", "gate21_14_cross_dataset_runs.csv"),
        "cross_dataset_by_method": _read(root, "cross_dataset", "gate21_14_cross_dataset_by_method.csv"),
        "pareto": _read(root, "pareto", "gate21_14_pareto_frontier.csv"),
    }


def _read(root: Path, component: str, filename: str) -> list[dict[str, str]]:
    rows = read_csv(root / component / filename)
    return rows if rows else read_csv(root / filename)


def _copy_tables(root: Path, source_root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    mapping = {
        "official_main": "gate21_14_official_main_by_method.csv",
        "budgeted_selector": "gate21_14_budgeted_selector_by_method.csv",
        "selector_hash_audit": "gate21_14_selector_hash_audit.csv",
        "external_tp_runs": "gate21_14_external_tp_runs.csv",
        "external_tp_by_method": "gate21_14_external_tp_by_method.csv",
        "external_tp_budget_audit": "gate21_14_external_tp_budget_audit.csv",
        "freehgc_standard_by_method": "gate21_14_freehgc_standard_by_method.csv",
        "freehgc_tp_by_method": "gate21_14_freehgc_tp_by_method.csv",
        "freehgc_protocol_audit": "gate21_14_freehgc_protocol_audit.csv",
        "freehgc_score_selector_by_method": "gate21_14_freehgc_score_selector_by_method.csv",
        "feature_ablation_runs": "gate21_14_feature_ablation_runs.csv",
        "feature_ablation_by_method": "gate21_14_feature_ablation_by_method.csv",
        "metapath": "gate21_14_metapath_tensor_dump.csv",
        "cache": "gate21_14_cache_hash_assertions.csv",
        "coverage": "gate21_14_coverage_semantic_diagnostics.csv",
        "adapter_by_method": "gate21_14_adapter_by_method.csv",
        "adapter_package": "gate21_14_adapter_package_audit.csv",
        "system_cost_runs": "gate21_14_system_workload_cost_runs.csv",
        "system_cost_by_method": "gate21_14_system_workload_cost_by_method.csv",
        "cross_dataset_runs": "gate21_14_cross_dataset_runs.csv",
        "cross_dataset_by_method": "gate21_14_cross_dataset_by_method.csv",
        "pareto": "gate21_14_pareto_frontier.csv",
    }
    for key, filename in mapping.items():
        write_rows(root / filename, tables.get(key, []))
    adapter_runs = source_root / "adapter" / "gate21_14_adapter_runs.csv"
    if adapter_runs.exists():
        dst = root / "gate21_14_adapter_runs.csv"
        if adapter_runs.resolve() != dst.resolve():
            shutil.copy2(adapter_runs, dst)


def _combined_by_method(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, key in (
        ("official_main", "official_main"),
        ("budgeted_selector", "budgeted_selector"),
        ("external_tp", "external_tp_by_method"),
        ("freehgc_standard", "freehgc_standard_by_method"),
        ("freehgc_tp", "freehgc_tp_by_method"),
        ("freehgc_score_selector", "freehgc_score_selector_by_method"),
        ("feature_ablation", "feature_ablation_by_method"),
        ("metapath_cache", "metapath"),
        ("coverage", "coverage"),
        ("adapter", "adapter_by_method"),
        ("system_cost", "system_cost_by_method"),
        ("cross_dataset", "cross_dataset_by_method"),
        ("pareto", "pareto"),
    ):
        for row in tables.get(key, []):
            out = dict(row)
            out["evidence_family"] = family
            rows.append(out)
    return rows


def _failure_rows(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, table in tables.items():
        for row in table:
            failed = str(row.get("failure_type", row.get("failure_reason", ""))).strip() or str(row.get("hard_failure_reason", "")).strip()
            if failed or (("success" in row) and not bool_value(row.get("success"))):
                out = dict(row)
                out["source_table"] = key
                rows.append(out)
    return rows


def _write_decision_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.14 Decision",
        "",
        f"- paper_ready_status: `{payload.get('paper_ready_status')}`",
        f"- ICDE_EVIDENCE_READY: `{payload.get('ICDE_EVIDENCE_READY')}`",
        "",
        "## Decision Flags",
    ]
    for name in REQUIRED_DECISION_FLAGS:
        lines.append(f"- {name}: `{payload.get(name)}`")
    blockers = payload.get("blocking_issues", [])
    if blockers:
        lines.extend(["", "## Blocking Issues"])
        lines.extend(f"- `{item}`" for item in blockers)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checklists(root: Path, payload: Mapping[str, Any]) -> None:
    checks = [
        ("P0 budgeted selector hash/linkage/no-test-leakage audit", payload.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS") and payload.get("BUDGETED_SELECTOR_LINKAGE_PASS") and payload.get("BUDGETED_SELECTOR_NO_TEST_LEAKAGE_PASS")),
        ("P1 external TP 5x5 real task metrics", payload.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY")),
        ("P2 FreeHGC standard/TP/score protocols separated", "FREEHGC_STANDARD_5SEED_READY" in payload and "FREEHGC_TP_SELECTION_READY" in payload and "FREEHGC_SCORE_SELECTOR_READY" in payload),
        ("P3 feature ablation task metrics and redundancy tests", payload.get("FEATURE_ABLATION_TASK_RESULTS_READY") and payload.get("PAPER_FEATURE_REDUNDANCY_TESTED") and payload.get("SUPPORT_FEATURE_REDUNDANCY_TESTED")),
        ("P4 real SeHGNN tensor/cache dump", payload.get("METAPATH_TENSOR_DUMP_READY") and payload.get("CACHE_HASH_REAL_PASS")),
        ("P5 semantic coverage distributional diagnostics", payload.get("COVERAGE_DISTRIBUTIONAL_MECHANISM_READY")),
        ("P6 APV12 adapter restored and APV16 attempted", payload.get("APV12_RP64_ADAPTER_RESTORED") and "APV16_RP64_ADAPTER_READY" in payload),
        ("P7 end-to-end system workload cost", payload.get("SYSTEM_WORKLOAD_COST_READY")),
        ("P8 ACM/IMDB cross-dataset real task results", payload.get("CROSS_DATASET_ACM_TASK_RESULTS_READY") and payload.get("CROSS_DATASET_IMDB_TASK_RESULTS_READY")),
        ("P9 Pareto frontier emitted", (root / "gate21_14_pareto_frontier.csv").exists()),
        ("All required summary artifacts emitted", all((root / name).exists() for name in SUMMARY_FILES if name != "gate21_14_prompt_completion_checklist.md")),
    ]
    _write_checklist(root / "gate21_14_requirement_checklist.md", "# Gate21.14 Requirement Checklist", checks, payload)
    prompt_checks = [(f"Required summary artifact `{name}` exists", True if name == "gate21_14_prompt_completion_checklist.md" else (root / name).exists()) for name in SUMMARY_FILES]
    prompt_checks.extend((f"Decision flag `{name}` evaluated", name in payload) for name in REQUIRED_DECISION_FLAGS)
    _write_checklist(root / "gate21_14_prompt_completion_checklist.md", "# Gate21.14 Prompt Completion Checklist", prompt_checks, payload)


def _write_checklist(path: Path, title: str, checks: Sequence[tuple[str, Any]], payload: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{payload.get('paper_ready_status')}`", ""]
    lines.extend(f"- [{'x' if bool_value(ok) else ' '}] {label}" for label, ok in checks)
    blockers = payload.get("blocking_issues", [])
    if blockers:
        lines.extend(["", "## Blocking Issues"])
        lines.extend(f"- `{item}`" for item in blockers)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.14 full execution push outputs.")
    parser.add_argument("--input-dir", "--result-dir", dest="input_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=_parse_bool_arg)
    return parser


def _parse_bool_arg(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = summarize(input_dir=args.input_dir, output_dir=args.output_dir, fail_on_missing_required=bool(args.fail_on_missing_required))
    print(json.dumps({"paper_ready_status": payload.get("paper_ready_status"), "blocking_issues": payload.get("blocking_issues", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
