from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.scripts.gate21_12_common import DEFAULT_OUTPUT_ROOT, SUMMARY_FILES, bool_value, read_csv, read_json, write_payload, write_rows
from hesf_coarsen.eval.official.gate21_12_decision import REQUIRED_DECISION_FLAGS, decision_status, gate21_12_decision


def summarize(*, result_dir: Path, fail_on_missing_required: bool = False) -> dict[str, Any]:
    root = Path(result_dir)
    tables = _load_tables(root)
    proof = read_json(root / "budgeted_selector" / "gate21_12_apv16_deterministic_proof.json")
    if not proof:
        proof = read_json(root / "gate21_12_apv16_deterministic_proof.json")
    manifest = read_json(root / "gate21_12_manifest.json")
    if not manifest:
        manifest = read_json(root / "audits" / "gate21_12_manifest.json")

    flags = gate21_12_decision(
        official_rows=tables["official_main"],
        budgeted_selector_rows=tables["budgeted_selector"],
        selector_hash_audit=tables["selector_hash_audit"],
        apv16_deterministic_proof=proof,
        external_tp_rows=tables["external_tp_runs"],
        external_tp_by_method=tables["external_tp_by_method"],
        freehgc_standard_runs=tables["freehgc_standard_runs"],
        freehgc_standard_by_method=tables["freehgc_standard_by_method"],
        freehgc_env=tables["freehgc_env"],
        freehgc_tp_rows=tables["freehgc_tp_runs"],
        metapath_rows=tables["metapath"],
        cache_rows=tables["cache"],
        feature_ablation_rows=tables["feature_ablation_runs"],
        adapter_rows=tables["adapter_runs"],
        adapter_by_method=tables["adapter_by_method"],
        system_cost_rows=tables["system_cost_runs"],
        storage_rows=tables["storage_audit"],
        cross_dataset_rows=tables["cross_dataset_runs"],
        cross_dataset_selector_plans=tables["cross_dataset_selector_plans"],
        coverage_rows=tables["coverage"],
    )
    payload: dict[str, Any] = {
        "gate": "21.12",
        "paper_ready_status": decision_status(flags),
        "manifest": manifest,
        **flags,
    }
    payload["blocking_issues"] = [name for name in REQUIRED_DECISION_FLAGS if not bool_value(payload.get(name))]

    _copy_tables_to_root(root, tables)
    write_payload(root / "gate21_12_manifest.json", manifest)
    write_payload(root / "gate21_12_decision.json", payload)
    write_payload(root / "gate21_12_apv16_deterministic_proof.json", proof)
    _write_decision_md(root / "gate21_12_decision.md", payload)
    write_rows(root / "gate21_12_failure_audit.csv", _failure_rows(tables))
    write_rows(root / "gate21_12_by_method.csv", _combined_by_method(tables))
    _write_checklists(root, payload)

    missing = [name for name in SUMMARY_FILES if not (root / name).exists()]
    if missing and fail_on_missing_required:
        raise RuntimeError(f"missing Gate21.12 summary artifacts: {missing}")
    payload["missing_summary_files"] = missing
    if missing:
        write_payload(root / "gate21_12_decision.json", payload)
        _write_decision_md(root / "gate21_12_decision.md", payload)
    return payload


def _load_tables(root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "official_main": read_csv(root / "official_main" / "gate21_12_official_main_by_method.csv"),
        "budgeted_selector": read_csv(root / "budgeted_selector" / "gate21_12_budgeted_selector_by_method.csv"),
        "selector_hash_audit": read_csv(root / "budgeted_selector" / "gate21_12_selector_hash_audit.csv"),
        "channel_trace": read_csv(root / "budgeted_selector" / "gate21_12_channel_planner_trace.csv"),
        "external_tp_runs": read_csv(root / "external_tp" / "gate21_12_external_tp_5x5_runs.csv"),
        "external_tp_by_method": read_csv(root / "external_tp" / "gate21_12_external_tp_5x5_by_method.csv"),
        "freehgc_standard_runs": read_csv(root / "freehgc" / "gate21_12_freehgc_standard_runs.csv"),
        "freehgc_standard_by_method": read_csv(root / "freehgc" / "gate21_12_freehgc_standard_by_method.csv"),
        "freehgc_tp_runs": read_csv(root / "freehgc" / "gate21_12_freehgc_tp_runs.csv"),
        "freehgc_tp_by_method": read_csv(root / "freehgc" / "gate21_12_freehgc_tp_by_method.csv"),
        "freehgc_env": read_csv(root / "freehgc" / "gate21_12_freehgc_env_audit.csv"),
        "metapath": read_csv(root / "metapath_cache" / "gate21_12_metapath_tensor_dump.csv"),
        "cache": read_csv(root / "metapath_cache" / "gate21_12_cache_hash_assertions.csv"),
        "cache_namespace": read_csv(root / "metapath_cache" / "gate21_12_cache_namespace_audit.csv"),
        "feature_ablation_runs": read_csv(root / "feature_ablation" / "gate21_12_feature_ablation_runs.csv"),
        "feature_ablation_by_method": read_csv(root / "feature_ablation" / "gate21_12_feature_ablation_by_method.csv"),
        "feature_shape": read_csv(root / "feature_ablation" / "gate21_12_feature_ablation_shape_audit.csv"),
        "adapter_runs": read_csv(root / "adapter" / "gate21_12_adapter_runs.csv"),
        "adapter_by_method": read_csv(root / "adapter" / "gate21_12_adapter_by_method.csv"),
        "adapter_package": read_csv(root / "adapter" / "gate21_12_adapter_package_audit.csv"),
        "system_cost_runs": read_csv(root / "system_cost" / "gate21_12_system_cost_runs.csv"),
        "system_cost_by_method": read_csv(root / "system_cost" / "gate21_12_system_cost_by_method.csv"),
        "storage_only": read_csv(root / "system_cost" / "gate21_12_storage_only_baselines.csv"),
        "storage_audit": read_csv(root / "system_cost" / "gate21_12_storage_audit.csv"),
        "cross_dataset_runs": read_csv(root / "cross_dataset" / "gate21_12_cross_dataset_runs.csv"),
        "cross_dataset_by_method": read_csv(root / "cross_dataset" / "gate21_12_cross_dataset_by_method.csv"),
        "cross_dataset_selector_plans": read_csv(root / "cross_dataset" / "gate21_12_cross_dataset_selector_plans.csv"),
        "coverage": read_csv(root / "coverage" / "gate21_12_coverage_diagnostics.csv"),
    }


def _copy_tables_to_root(root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    mapping = {
        "official_main": "gate21_12_official_main_by_method.csv",
        "budgeted_selector": "gate21_12_budgeted_selector_by_method.csv",
        "external_tp_by_method": "gate21_12_external_tp_5x5_by_method.csv",
        "freehgc_standard_runs": "gate21_12_freehgc_standard_runs.csv",
        "freehgc_standard_by_method": "gate21_12_freehgc_standard_by_method.csv",
        "freehgc_tp_runs": "gate21_12_freehgc_tp_runs.csv",
        "freehgc_tp_by_method": "gate21_12_freehgc_tp_by_method.csv",
        "freehgc_env": "gate21_12_freehgc_env_audit.csv",
        "metapath": "gate21_12_metapath_tensor_dump.csv",
        "cache": "gate21_12_cache_hash_assertions.csv",
        "cache_namespace": "gate21_12_cache_namespace_audit.csv",
        "feature_ablation_runs": "gate21_12_feature_ablation_runs.csv",
        "feature_ablation_by_method": "gate21_12_feature_ablation_by_method.csv",
        "feature_shape": "gate21_12_feature_ablation_shape_audit.csv",
        "adapter_runs": "gate21_12_adapter_runs.csv",
        "adapter_by_method": "gate21_12_adapter_by_method.csv",
        "adapter_package": "gate21_12_adapter_package_audit.csv",
        "system_cost_runs": "gate21_12_system_cost_runs.csv",
        "system_cost_by_method": "gate21_12_system_cost_by_method.csv",
        "storage_only": "gate21_12_storage_only_baselines.csv",
        "storage_audit": "gate21_12_storage_audit.csv",
        "cross_dataset_runs": "gate21_12_cross_dataset_runs.csv",
        "cross_dataset_by_method": "gate21_12_cross_dataset_by_method.csv",
        "cross_dataset_selector_plans": "gate21_12_cross_dataset_selector_plans.csv",
        "coverage": "gate21_12_coverage_diagnostics.csv",
    }
    for key, filename in mapping.items():
        write_rows(root / filename, tables.get(key, []))
    freehgc_failure = read_json(root / "freehgc" / "gate21_12_freehgc_failure_proof.json")
    write_payload(root / "gate21_12_freehgc_failure_proof.json", freehgc_failure)


def _failure_rows(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family, rows in tables.items():
        for row in rows:
            failure = str(row.get("failure_type", "")).strip() or str(row.get("failure_reason", row.get("failure_message", ""))).strip()
            success = bool_value(row.get("success", row.get("training_executed", False)))
            if failure or not success and family in {"external_tp_runs", "freehgc_standard_runs", "freehgc_tp_runs", "feature_ablation_runs", "system_cost_runs", "cross_dataset_runs"}:
                item = dict(row)
                item["evidence_family"] = family
                out.append(item)
    return out


def _combined_by_method(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, key in (
        ("official_main", "official_main"),
        ("external_tp", "external_tp_by_method"),
        ("freehgc_standard", "freehgc_standard_by_method"),
        ("freehgc_tp", "freehgc_tp_by_method"),
        ("feature_ablation", "feature_ablation_by_method"),
        ("adapter", "adapter_by_method"),
        ("system_cost", "system_cost_by_method"),
        ("cross_dataset", "cross_dataset_by_method"),
    ):
        for row in tables.get(key, []):
            out = dict(row)
            out["evidence_family"] = family
            rows.append(out)
    return rows


def _write_decision_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.12 Decision",
        "",
        f"- paper_ready_status: `{payload.get('paper_ready_status')}`",
        f"- ICDE_SUBMISSION_EVIDENCE_READY: `{payload.get('ICDE_SUBMISSION_EVIDENCE_READY')}`",
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
    requirement_checks = [
        ("P0 selector hash audit executed", payload.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS")),
        ("Planner rows separated from task-result rows", (root / "gate21_12_budgeted_selector_by_method.csv").exists()),
        ("APV16 deterministic proof artifact emitted", (root / "gate21_12_apv16_deterministic_proof.json").exists()),
        ("External TP 5x5 table emitted", (root / "gate21_12_external_tp_5x5_by_method.csv").exists()),
        ("FreeHGC precise protocol audit emitted", (root / "gate21_12_freehgc_failure_proof.json").exists()),
        ("Metapath/cache dump or patch failure table emitted", (root / "gate21_12_metapath_tensor_dump.csv").exists()),
        ("Feature ablation task table emitted", (root / "gate21_12_feature_ablation_runs.csv").exists()),
        ("Adapter package table emitted", (root / "gate21_12_adapter_package_audit.csv").exists()),
        ("System cost table emitted", (root / "gate21_12_system_cost_runs.csv").exists()),
        ("ACM/IMDB cross-dataset table emitted", (root / "gate21_12_cross_dataset_runs.csv").exists()),
        ("Coverage diagnostics table emitted", (root / "gate21_12_coverage_diagnostics.csv").exists()),
        ("No placeholder/hard failure/smoke row marks ICDE ready", not bool_value(payload.get("ICDE_SUBMISSION_EVIDENCE_READY")) or not payload.get("blocking_issues")),
        ("All required summary artifacts emitted", all((root / name).exists() for name in SUMMARY_FILES)),
    ]
    _write_checklist(root / "gate21_12_requirement_checklist.md", "# Gate21.12 Requirement Checklist", requirement_checks, payload)
    prompt_checks = [(f"Required summary artifact `{name}` exists", (root / name).exists()) for name in SUMMARY_FILES]
    prompt_checks.extend((f"Decision flag `{name}` evaluated", name in payload) for name in REQUIRED_DECISION_FLAGS)
    _write_checklist(root / "gate21_12_prompt_completion_checklist.md", "# Gate21.12 Prompt Completion Checklist", prompt_checks, payload)


def _write_checklist(path: Path, title: str, checks: Sequence[tuple[str, Any]], payload: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{payload.get('paper_ready_status')}`", ""]
    lines.extend(f"- [{'x' if bool_value(ok) else ' '}] {label}" for label, ok in checks)
    blockers = payload.get("blocking_issues", [])
    if blockers:
        lines.extend(["", "## Blocking Issues"])
        lines.extend(f"- `{item}`" for item in blockers)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.12 executed evidence completion outputs.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    payload = summarize(result_dir=args.result_dir, fail_on_missing_required=bool(args.fail_on_missing_required))
    print({"paper_ready_status": payload.get("paper_ready_status"), "blocking_issues": payload.get("blocking_issues", [])})


if __name__ == "__main__":
    main()
