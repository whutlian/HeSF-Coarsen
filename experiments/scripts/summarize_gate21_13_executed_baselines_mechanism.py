from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_13_common import DEFAULT_OUTPUT_ROOT, SUMMARY_FILES, bool_value, read_csv, read_json, write_payload, write_rows
from hesf_coarsen.eval.official.gate21_13_decision import REQUIRED_DECISION_FLAGS, decision_status, gate21_13_decision


def summarize(*, result_dir: Path, out_dir: Path | None = None, fail_on_missing_required: bool = False) -> dict[str, Any]:
    source_root = Path(result_dir)
    root = Path(out_dir) if out_dir is not None else source_root
    root.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(source_root)
    manifest = read_json(source_root / "gate21_13_manifest.json")
    if not manifest:
        manifest = read_json(source_root / "audits" / "gate21_13_manifest.json")
    flags = gate21_13_decision(
        official_rows=tables["official_main"],
        selector_hash_audit=tables["selector_hash_audit"],
        deterministic_proof_rows=tables["deterministic_proof"],
        external_tp_rows=tables["external_tp_runs"],
        external_tp_by_method_budget=tables["external_tp_by_method_budget"],
        external_tp_budget_fairness=tables["external_tp_budget_fairness"],
        freehgc_env_rows=tables["freehgc_env"],
        freehgc_standard_runs=tables["freehgc_standard_runs"],
        freehgc_standard_by_ratio=tables["freehgc_standard_by_ratio"],
        freehgc_tp_runs=tables["freehgc_tp_runs"],
        freehgc_tp_adapter_audit=tables["freehgc_tp_adapter_audit"],
        metapath_rows=tables["metapath"],
        cache_rows=tables["cache"],
        feature_ablation_rows=tables["feature_ablation_runs"],
        adapter_rows=tables["adapter_runs"],
        system_cost_rows=tables["system_cost_runs"],
        cross_dataset_rows=tables["cross_dataset_runs"],
    )
    payload: dict[str, Any] = {
        "gate": "21.13",
        "paper_ready_status": decision_status(flags),
        "manifest": manifest,
        **flags,
    }
    payload["blocking_issues"] = [name for name in REQUIRED_DECISION_FLAGS if not bool_value(payload.get(name))]
    _copy_tables_to_summary(root, source_root, tables)
    write_payload(root / "gate21_13_manifest.json", manifest)
    write_payload(root / "gate21_13_decision.json", payload)
    _write_decision_md(root / "gate21_13_decision.md", payload)
    write_payload(root / "gate21_13_icde_evidence_manifest.json", _evidence_manifest(root, tables, payload))
    write_rows(root / "gate21_13_by_method.csv", _combined_by_method(tables))
    write_rows(root / "gate21_13_failure_audit.csv", _failure_rows(tables))
    _write_checklists(root, payload)

    missing = [name for name in SUMMARY_FILES if not (root / name).exists()]
    payload["missing_summary_files"] = missing
    if missing:
        write_payload(root / "gate21_13_decision.json", payload)
        _write_decision_md(root / "gate21_13_decision.md", payload)
    if missing and fail_on_missing_required:
        raise RuntimeError(f"missing Gate21.13 summary artifacts: {missing}")
    return payload


def _load_tables(root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "official_main": _read(root, "official_main", "gate21_13_official_main_by_method.csv"),
        "budgeted_selector": _read(root, "budgeted_selector", "gate21_13_budgeted_selector_by_method.csv"),
        "selector_hash_audit": _read(root, "budgeted_selector", "gate21_13_selector_hash_audit.csv"),
        "deterministic_proof": _read(root, "budgeted_selector", "gate21_13_deterministic_selector_proof.csv"),
        "selector_modes": _read(root, "budgeted_selector", "gate21_13_selector_modes.csv"),
        "selector_pareto": _read(root, "budgeted_selector", "gate21_13_selector_pareto_frontier.csv"),
        "external_tp_runs": _read(root, "external_baselines", "gate21_13_external_tp_runs.csv"),
        "external_tp_by_method_budget": _read(root, "external_baselines", "gate21_13_external_tp_by_method_budget.csv"),
        "external_tp_budget_fairness": _read(root, "external_baselines", "gate21_13_external_tp_budget_fairness.csv"),
        "external_tp_failure_report": _read(root, "external_baselines", "gate21_13_external_tp_failure_report.csv"),
        "freehgc_env": _read(root, "freehgc", "gate21_13_freehgc_env_audit.csv"),
        "freehgc_standard_runs": _read(root, "freehgc", "gate21_13_freehgc_standard_runs.csv"),
        "freehgc_standard_by_ratio": _read(root, "freehgc", "gate21_13_freehgc_standard_by_ratio.csv"),
        "freehgc_tp_adapter_audit": _read(root, "freehgc", "gate21_13_freehgc_tp_adapter_audit.csv"),
        "freehgc_tp_runs": _read(root, "freehgc", "gate21_13_freehgc_tp_runs.csv"),
        "metapath": _read(root, "metapath_cache", "gate21_13_metapath_tensor_dump.csv"),
        "cache": _read(root, "metapath_cache", "gate21_13_cache_hash_assertions.csv"),
        "metapath_key_diff": _read(root, "metapath_cache", "gate21_13_metapath_key_diff.csv"),
        "feature_ablation_runs": _read(root, "feature_ablation", "gate21_13_feature_ablation_runs.csv"),
        "feature_ablation_by_method": _read(root, "feature_ablation", "gate21_13_feature_ablation_by_method.csv"),
        "feature_shape": _read(root, "feature_ablation", "gate21_13_feature_ablation_shape_assertions.csv"),
        "adapter_runs": _read(root, "adapter", "gate21_13_adapter_runs.csv"),
        "adapter_by_method": _read(root, "adapter", "gate21_13_adapter_by_method.csv"),
        "adapter_package": _read(root, "adapter_packages", "gate21_13_adapter_package_audit.csv"),
        "system_cost_runs": _read(root, "system_cost", "gate21_13_system_cost_runs.csv"),
        "system_cost_by_method": _read(root, "system_cost", "gate21_13_system_cost_by_method.csv"),
        "cross_dataset_runs": _read(root, "cross_dataset", "gate21_13_cross_dataset_runs.csv"),
        "cross_dataset_by_method": _read(root, "cross_dataset", "gate21_13_cross_dataset_by_method.csv"),
        "cross_dataset_selector_plans": _read(root, "cross_dataset", "gate21_13_cross_dataset_selector_plans.csv"),
    }


def _read(root: Path, component: str, filename: str) -> list[dict[str, str]]:
    rows = read_csv(root / component / filename)
    return rows if rows else read_csv(root / filename)


def _copy_tables_to_summary(root: Path, source_root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    mapping = {
        "official_main": "gate21_13_official_main_by_method.csv",
        "budgeted_selector": "gate21_13_budgeted_selector_by_method.csv",
        "selector_hash_audit": "gate21_13_selector_hash_audit.csv",
        "deterministic_proof": "gate21_13_deterministic_selector_proof.csv",
        "external_tp_runs": "gate21_13_external_tp_runs.csv",
        "external_tp_by_method_budget": "gate21_13_external_tp_by_method_budget.csv",
        "external_tp_budget_fairness": "gate21_13_external_tp_budget_fairness.csv",
        "external_tp_failure_report": "gate21_13_external_tp_failure_report.csv",
        "freehgc_env": "gate21_13_freehgc_env_audit.csv",
        "freehgc_standard_runs": "gate21_13_freehgc_standard_runs.csv",
        "freehgc_standard_by_ratio": "gate21_13_freehgc_standard_by_ratio.csv",
        "freehgc_tp_adapter_audit": "gate21_13_freehgc_tp_adapter_audit.csv",
        "freehgc_tp_runs": "gate21_13_freehgc_tp_runs.csv",
        "metapath": "gate21_13_metapath_tensor_dump.csv",
        "cache": "gate21_13_cache_hash_assertions.csv",
        "metapath_key_diff": "gate21_13_metapath_key_diff.csv",
        "feature_ablation_runs": "gate21_13_feature_ablation_runs.csv",
        "feature_ablation_by_method": "gate21_13_feature_ablation_by_method.csv",
        "feature_shape": "gate21_13_feature_ablation_shape_assertions.csv",
        "adapter_runs": "gate21_13_adapter_runs.csv",
        "adapter_by_method": "gate21_13_adapter_by_method.csv",
        "adapter_package": "gate21_13_adapter_package_audit.csv",
        "system_cost_runs": "gate21_13_system_cost_runs.csv",
        "system_cost_by_method": "gate21_13_system_cost_by_method.csv",
        "cross_dataset_runs": "gate21_13_cross_dataset_runs.csv",
        "cross_dataset_by_method": "gate21_13_cross_dataset_by_method.csv",
        "cross_dataset_selector_plans": "gate21_13_cross_dataset_selector_plans.csv",
        "selector_modes": "gate21_13_selector_modes.csv",
        "selector_pareto": "gate21_13_selector_pareto_frontier.csv",
    }
    for key, filename in mapping.items():
        write_rows(root / filename, tables.get(key, []))
    proof_src = source_root / "freehgc" / "gate21_13_freehgc_tp_failure_proof.json"
    proof_dst = root / "gate21_13_freehgc_tp_failure_proof.json"
    if proof_src.exists() and proof_src.resolve() != proof_dst.resolve():
        shutil.copy2(proof_src, proof_dst)
    elif not proof_dst.exists():
        write_payload(proof_dst, {})


def _combined_by_method(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, key in (
        ("official_main", "official_main"),
        ("external_tp", "external_tp_by_method_budget"),
        ("freehgc_standard", "freehgc_standard_by_ratio"),
        ("freehgc_tp", "freehgc_tp_adapter_audit"),
        ("metapath_cache", "metapath"),
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


def _failure_rows(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failure_keys = {
        "external_tp_runs",
        "external_tp_failure_report",
        "freehgc_env",
        "freehgc_standard_runs",
        "freehgc_tp_adapter_audit",
        "metapath",
        "cache",
        "feature_ablation_runs",
        "adapter_runs",
        "system_cost_runs",
        "cross_dataset_runs",
    }
    for key in failure_keys:
        for row in tables.get(key, []):
            if str(row.get("failure_type", row.get("failure_reason", ""))).strip() or not bool_value(row.get("success", row.get("assertion_pass", True))):
                out = dict(row)
                out["source_table"] = key
                rows.append(out)
    return rows


def _evidence_manifest(root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gate": "21.13",
        "paper_ready_status": payload.get("paper_ready_status"),
        "ICDE_EVIDENCE_READY": payload.get("ICDE_EVIDENCE_READY"),
        "summary_dir": str(root),
        "table_row_counts": {key: len(value) for key, value in tables.items()},
        "strict_ready_guardrails": [
            "smoke rows do not count as complete 5x5",
            "hard failures are not counted as success",
            "empty/NaN tensor hashes cannot pass cache assertions",
            "feature adapters are excluded from official main table",
            "selectors declare uses_test_metrics_for_selection=false",
        ],
    }


def _write_decision_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.13 Decision",
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
        ("P0 selector hash/linkage audit fully passing", payload.get("BUDGETED_SELECTOR_HASH_AUDIT_PASS")),
        ("APV12/APV16 official anchors reproduced", payload.get("OFFICIAL_DBLP_APV12_PASS") and payload.get("OFFICIAL_DBLP_APV16_PASS")),
        ("External TP 5x5 required ready", payload.get("EXTERNAL_TP_5X5_REQUIRED_READY")),
        ("FreeHGC standard 5-seed ready", payload.get("FREEHGC_STANDARD_5SEED_READY")),
        ("FreeHGC TP synthetic ready or hard incompatibility proven", payload.get("FREEHGC_TP_SYNTHETIC_READY_OR_HARD_INCOMPATIBILITY_PROVEN")),
        ("Metapath/cache tensor dump ready", payload.get("METAPATH_TENSOR_DUMP_READY") and payload.get("CACHE_HASH_REAL_PASS")),
        ("Feature ablation task metrics ready", payload.get("FEATURE_ABLATION_TASK_READY")),
        ("APV16 adapter ready", payload.get("APV16_ADAPTER_READY")),
        ("System cost end-to-end ready", payload.get("SYSTEM_COST_END_TO_END_READY")),
        ("ACM or IMDB real HeSF-RCS-auto task result ready", payload.get("CROSS_DATASET_ACM_READY") or payload.get("CROSS_DATASET_IMDB_READY")),
        ("All required summary artifacts emitted", all((root / name).exists() for name in SUMMARY_FILES)),
    ]
    _write_checklist(root / "gate21_13_requirement_checklist.md", "# Gate21.13 Requirement Checklist", checks, payload)
    prompt_checks = [
        (f"Required summary artifact `{name}` exists", True if name == "gate21_13_prompt_completion_checklist.md" else (root / name).exists())
        for name in SUMMARY_FILES
    ]
    prompt_checks.extend((f"Decision flag `{name}` evaluated", name in payload) for name in REQUIRED_DECISION_FLAGS)
    _write_checklist(root / "gate21_13_prompt_completion_checklist.md", "# Gate21.13 Prompt Completion Checklist", prompt_checks, payload)


def _write_checklist(path: Path, title: str, checks: Sequence[tuple[str, Any]], payload: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{payload.get('paper_ready_status')}`", ""]
    lines.extend(f"- [{'x' if bool_value(ok) else ' '}] {label}" for label, ok in checks)
    blockers = payload.get("blocking_issues", [])
    if blockers:
        lines.extend(["", "## Blocking Issues"])
        lines.extend(f"- `{item}`" for item in blockers)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.13 executed baselines and mechanism outputs.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
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
    payload = summarize(result_dir=args.result_dir, out_dir=args.out_dir, fail_on_missing_required=bool(args.fail_on_missing_required))
    print(json.dumps({"paper_ready_status": payload.get("paper_ready_status"), "blocking_issues": payload.get("blocking_issues", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
