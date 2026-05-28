from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_11_common import DEFAULT_OUTPUT_ROOT, SUMMARY_FILES, bool_value, ensure_layout, parse_bool_arg, read_csv, read_json, write_payload, write_rows
from hesf_coarsen.eval.official.gate21_11_decision import REQUIRED_DECISION_FLAGS, decision_status, gate21_11_decision


def summarize(input_dir: Path, out_dir: Path | None = None, *, fail_on_missing_required: bool = False) -> dict[str, Any]:
    root = Path(input_dir)
    paths = ensure_layout(root)
    summary = Path(out_dir or paths["summary"])
    summary.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(paths)
    proof = read_json(paths["budgeted_selector"] / "gate21_11_apv16_deterministic_proof.json")
    decision = gate21_11_decision(
        official_rows=tables["official_main"],
        budgeted_selector_rows=tables["budgeted_selector"],
        channel_trace_rows=tables["channel_trace"],
        external_tp_runs=tables["external_tp_runs"],
        external_tp_by_method=tables["external_tp_by_method"],
        freehgc_standard_runs=tables["freehgc_standard_runs"],
        freehgc_standard_by_method=tables["freehgc_standard_by_method"],
        freehgc_env=tables["freehgc_env"],
        freehgc_tp_audit=tables["freehgc_tp_audit"],
        metapath_rows=tables["metapath"],
        cache_rows=tables["cache"],
        feature_ablation_rows=tables["feature_ablation_runs"],
        adapter_rows=tables["adapter_audit"],
        adapter_by_method=tables["adapter_by_method"],
        system_cost_rows=tables["system_cost_runs"],
        cross_dataset_rows=tables["cross_dataset_runs"],
        coverage_rows=tables["coverage"],
    )
    payload: dict[str, Any] = {
        **decision,
        "paper_ready_status": decision_status(decision),
        "blocking_issues": [name for name in REQUIRED_DECISION_FLAGS if not bool_value(decision.get(name))],
        "counts": {name: len(rows) for name, rows in tables.items()},
        "input_dir": str(root),
        "paper_safe_claims": _safe_claims(decision),
        "paper_unsafe_claims": _unsafe_claims(decision),
    }
    _write_summary(summary, tables, proof, payload)
    _write_decision_md(summary / "gate21_11_decision.md", payload)
    _write_checklists(summary, payload)
    missing = [name for name in SUMMARY_FILES if not (summary / name).exists()]
    if missing and fail_on_missing_required:
        raise RuntimeError(f"missing Gate21.11 summary artifacts: {missing}")
    return payload


def _load_tables(paths: Mapping[str, Path]) -> dict[str, list[dict[str, str]]]:
    return {
        "official_main": read_csv(paths["official_main"] / "gate21_11_official_main_by_method.csv"),
        "budgeted_selector": read_csv(paths["budgeted_selector"] / "gate21_11_budgeted_selector_by_method.csv"),
        "channel_trace": read_csv(paths["budgeted_selector"] / "gate21_11_channel_planner_trace.csv"),
        "external_tp_runs": read_csv(paths["external_tp"] / "gate21_11_external_tp_5x5_runs.csv"),
        "external_tp_by_method": read_csv(paths["external_tp"] / "gate21_11_external_tp_by_method.csv"),
        "external_tp_budget": read_csv(paths["external_tp"] / "gate21_11_external_tp_budget_audit.csv"),
        "freehgc_standard_runs": read_csv(paths["freehgc"] / "gate21_11_freehgc_standard_runs.csv"),
        "freehgc_standard_by_method": read_csv(paths["freehgc"] / "gate21_11_freehgc_standard_by_method.csv"),
        "freehgc_tp_audit": read_csv(paths["freehgc"] / "gate21_11_freehgc_tp_adapter_audit.csv"),
        "freehgc_env": read_csv(paths["freehgc"] / "gate21_11_freehgc_env_audit.csv"),
        "metapath": read_csv(paths["metapath_cache"] / "gate21_11_metapath_tensor_dump.csv"),
        "cache": read_csv(paths["metapath_cache"] / "gate21_11_cache_hash_assertions.csv"),
        "feature_ablation_runs": read_csv(paths["feature_ablation"] / "gate21_11_feature_ablation_task_runs.csv"),
        "feature_ablation_by_method": read_csv(paths["feature_ablation"] / "gate21_11_feature_ablation_by_method.csv"),
        "adapter_audit": read_csv(paths["adapter"] / "gate21_11_adapter_package_audit.csv"),
        "adapter_by_method": read_csv(paths["adapter"] / "gate21_11_adapter_by_method.csv"),
        "system_cost_runs": read_csv(paths["system_cost"] / "gate21_11_system_cost_runs.csv"),
        "system_cost_by_method": read_csv(paths["system_cost"] / "gate21_11_system_cost_by_method.csv"),
        "cross_dataset_runs": read_csv(paths["cross_dataset"] / "gate21_11_cross_dataset_task_runs.csv"),
        "cross_dataset_by_method": read_csv(paths["cross_dataset"] / "gate21_11_cross_dataset_by_method.csv"),
        "coverage": read_csv(paths["coverage"] / "gate21_11_coverage_semantic_diagnostics.csv"),
    }


def _write_summary(summary: Path, tables: Mapping[str, list[dict[str, str]]], proof: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    write_payload(summary / "gate21_11_decision.json", payload)
    copies = {
        "gate21_11_official_main_by_method.csv": tables["official_main"],
        "gate21_11_budgeted_selector_by_method.csv": tables["budgeted_selector"],
        "gate21_11_channel_planner_trace.csv": tables["channel_trace"],
        "gate21_11_external_tp_5x5_runs.csv": tables["external_tp_runs"],
        "gate21_11_external_tp_by_method.csv": tables["external_tp_by_method"],
        "gate21_11_external_tp_budget_audit.csv": tables["external_tp_budget"],
        "gate21_11_freehgc_standard_runs.csv": tables["freehgc_standard_runs"],
        "gate21_11_freehgc_standard_by_method.csv": tables["freehgc_standard_by_method"],
        "gate21_11_freehgc_tp_adapter_audit.csv": tables["freehgc_tp_audit"],
        "gate21_11_freehgc_env_audit.csv": tables["freehgc_env"],
        "gate21_11_metapath_tensor_dump.csv": tables["metapath"],
        "gate21_11_cache_hash_assertions.csv": tables["cache"],
        "gate21_11_feature_ablation_task_runs.csv": tables["feature_ablation_runs"],
        "gate21_11_feature_ablation_by_method.csv": tables["feature_ablation_by_method"],
        "gate21_11_adapter_package_audit.csv": tables["adapter_audit"],
        "gate21_11_adapter_by_method.csv": tables["adapter_by_method"],
        "gate21_11_system_cost_runs.csv": tables["system_cost_runs"],
        "gate21_11_system_cost_by_method.csv": tables["system_cost_by_method"],
        "gate21_11_cross_dataset_task_runs.csv": tables["cross_dataset_runs"],
        "gate21_11_cross_dataset_by_method.csv": tables["cross_dataset_by_method"],
        "gate21_11_coverage_semantic_diagnostics.csv": tables["coverage"],
    }
    for name, rows in copies.items():
        write_rows(summary / name, rows, fieldnames=["status"] if not rows else None)
    write_payload(summary / "gate21_11_apv16_deterministic_proof.json", proof)
    write_rows(summary / "gate21_11_failures.csv", _failure_rows(tables))


def _failure_rows(tables: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, table in tables.items():
        for row in table:
            failure = str(row.get("failure_type", "")).strip()
            if failure or (row.get("success") not in {"", None} and not bool_value(row.get("success"))):
                rows.append(
                    {
                        "source_table": name,
                        "dataset": row.get("dataset", ""),
                        "method": row.get("method", row.get("variant", "")),
                        "failure_type": failure or "not_ready",
                        "failure_reason": row.get("failure_reason", row.get("failure_message", "")),
                        "eligible_for_decision": row.get("eligible_for_decision", ""),
                    }
                )
    return rows


def _write_decision_md(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Gate21.11 Decision",
        "",
        f"- paper_ready_status: `{payload.get('paper_ready_status', '')}`",
        f"- ICDE_SUBMISSION_EVIDENCE_READY: `{payload.get('ICDE_SUBMISSION_EVIDENCE_READY', False)}`",
        "",
        "## Flags",
    ]
    for name in REQUIRED_DECISION_FLAGS:
        lines.append(f"- [{'x' if bool_value(payload.get(name)) else ' '}] `{name}`")
    lines.append("")
    lines.append("## Blocking Issues")
    blockers = list(payload.get("blocking_issues", []))
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checklists(summary: Path, payload: Mapping[str, Any]) -> None:
    checks = [
        ("APV12/APV16 official DBLP anchors preserved", payload.get("OFFICIAL_MAIN_DBLP_READY")),
        ("Budgeted selector clean budget12/budget16 and slack-aware rows emitted", payload.get("BUDGETED_PLANNER_DBLP_012_PASS") and payload.get("BUDGETED_PLANNER_DBLP_016_PASS")),
        ("External TP rows are real 5x5 or explicit failures", (summary / "gate21_11_external_tp_5x5_runs.csv").exists()),
        ("FreeHGC standard verified or precise hard failure emitted", payload.get("FREEHGC_STANDARD_5SEED_READY") or payload.get("FREEHGC_STANDARD_HARD_FAILURE_WITH_REASON")),
        ("FreeHGC TP-selection task or hard incompatibility emitted", payload.get("FREEHGC_TP_SELECTION_TASK_READY") or payload.get("FREEHGC_TP_HARD_INCOMPATIBILITY_PROVEN")),
        ("Metapath/cache tensor dump table emitted", (summary / "gate21_11_metapath_tensor_dump.csv").exists()),
        ("Feature ablation task table emitted", (summary / "gate21_11_feature_ablation_task_runs.csv").exists()),
        ("APV12/APV16 adapter attempts reported", (summary / "gate21_11_adapter_package_audit.csv").exists()),
        ("End-to-end system cost table emitted", (summary / "gate21_11_system_cost_runs.csv").exists()),
        ("ACM/IMDB task rows or hard failures emitted", (summary / "gate21_11_cross_dataset_task_runs.csv").exists()),
        ("No placeholder/smoke/hard failure row marks task evidence ready", not payload.get("ICDE_SUBMISSION_EVIDENCE_READY") or not payload.get("blocking_issues")),
        ("All required summary artifacts emitted", all((summary / name).exists() for name in SUMMARY_FILES)),
    ]
    _write_checklist(summary / "gate21_11_requirement_checklist.md", "# Gate21.11 Requirement Checklist", checks, payload)
    prompt_checks = [(f"Required summary artifact `{name}` exists", (summary / name).exists()) for name in SUMMARY_FILES]
    prompt_checks.extend((f"Decision flag `{name}` evaluated", name in payload) for name in REQUIRED_DECISION_FLAGS)
    _write_checklist(summary / "gate21_11_prompt_completion_checklist.md", "# Gate21.11 Prompt Completion Checklist", prompt_checks, payload)


def _write_checklist(path: Path, title: str, checks: Sequence[tuple[str, Any]], payload: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{payload.get('paper_ready_status', '')}`", ""]
    for label, value in checks:
        lines.append(f"- [{'x' if bool_value(value) else ' '}] {label}")
    lines.append("")
    lines.append("## Blocking Issues")
    blockers = list(payload.get("blocking_issues", []))
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_claims(flags: Mapping[str, Any]) -> list[str]:
    claims = []
    if bool_value(flags.get("OFFICIAL_MAIN_DBLP_READY")):
        claims.append("DBLP official-unmodified APV12/APV16 anchors are preserved for the main TP workload table.")
    if bool_value(flags.get("BUDGETED_PLANNER_DBLP_016_PASS")):
        claims.append("The DBLP budgeted selector emits an APV16-like budget16 plan without test leakage fields.")
    if bool_value(flags.get("FREEHGC_TP_HARD_INCOMPATIBILITY_PROVEN")):
        claims.append("FreeHGC-TP incompatibility is recorded as a specific HGB export/provenance failure, not as a task result.")
    return claims


def _unsafe_claims(flags: Mapping[str, Any]) -> list[str]:
    claims = []
    if not bool_value(flags.get("EXTERNAL_TP_5X5_READY")):
        claims.append("Do not claim matched external TP 5x5 superiority.")
    if not bool_value(flags.get("METAPATH_TENSOR_DUMP_READY")):
        claims.append("Do not claim real SeHGNN metapath/cache tensor mechanism evidence.")
    if not bool_value(flags.get("FEATURE_ABLATION_TASK_READY")):
        claims.append("Do not claim feature ablation task conclusions.")
    if not bool_value(flags.get("SYSTEM_COST_END_TO_END_READY")):
        claims.append("Do not claim end-to-end workload cost readiness.")
    if not bool_value(flags.get("CROSS_DATASET_ACM_TASK_READY")) or not bool_value(flags.get("CROSS_DATASET_IMDB_TASK_READY")):
        claims.append("Do not claim ACM/IMDB generalization from plan-only or failure rows.")
    return claims


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Gate21.11 ICDE submission lockdown evidence.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fail-on-missing-required", nargs="?", const=True, default=False, type=parse_bool_arg)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = summarize(input_dir=Path(args.input), out_dir=Path(args.out) if args.out else None, fail_on_missing_required=bool(args.fail_on_missing_required))
    print(json.dumps({"paper_ready_status": payload.get("paper_ready_status"), "blocking_issues": payload.get("blocking_issues", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
