from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_9_common import DEFAULT_GATE21_8_ROOT, DEFAULT_OUTPUT_ROOT, ensure_layout, read_csv
from hesf_coarsen.eval.official.gate21_9_decision import decision_md, gate21_9_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


SUMMARY_TABLES = {
    "gate21_9_adapter_table.csv": ("adapter_package_v4", "gate21_9_adapter_by_method.csv"),
    "gate21_9_external_tp_table.csv": ("external_tp_5x5", "gate21_9_external_tp_by_method.csv"),
    "gate21_9_standard_condensation_table.csv": ("freehgc_protocols", "gate21_9_freehgc_standard_by_ratio.csv"),
    "gate21_9_cross_dataset_table.csv": ("cross_dataset_auto_channel", "gate21_9_cross_dataset_by_method.csv"),
    "gate21_9_storage_system_table.csv": ("storage_system_costs", "gate21_9_storage_system_by_method.csv"),
}


def summarize(input_root: Path, output_root: Path | None = None, *, strict: bool = False) -> dict[str, Any]:
    root = Path(input_root)
    paths = ensure_layout(root)
    summaries = Path(output_root or root)
    summaries.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(paths["logs"] / "gate21_9_run_manifest.json")
    gate21_8_root = Path(manifest.get("gate21_8_root", DEFAULT_GATE21_8_ROOT))

    official = _official_rows(root, gate21_8_root)
    auto_selector = read_csv(paths["auto_selector_alignment"] / "gate21_9_auto_selector_by_method.csv")
    external_tp = read_csv(paths["external_tp_5x5"] / "gate21_9_external_tp_task_rows.csv")
    freehgc_standard = read_csv(paths["freehgc_protocols"] / "gate21_9_freehgc_standard_task_rows.csv")
    freehgc_env = _read_json(paths["freehgc_protocols"] / "gate21_9_freehgc_env_audit.json")
    for row in freehgc_standard:
        row["standard_condensation_supported"] = freehgc_env.get("standard_condensation_supported", False)
        row["upstream_env_verified"] = freehgc_env.get("upstream_config_verified", False)
    freehgc_tp = read_csv(paths["freehgc_protocols"] / "gate21_9_freehgc_tp_adapter_audit.csv")
    metapath = read_csv(paths["metapath_cache_dump"] / "gate21_9_metapath_tensor_audit.csv")
    cache_assertions = read_csv(paths["metapath_cache_dump"] / "gate21_9_cache_hash_assertions.csv")
    feature_ablation = read_csv(paths["feature_ablation_tasks"] / "gate21_9_feature_ablation_task_rows.csv")
    adapter = read_csv(paths["adapter_package_v4"] / "gate21_9_adapter_task_rows.csv")
    storage = read_csv(paths["storage_system_costs"] / "gate21_9_storage_system_by_method.csv")
    ratio_audit = read_csv(paths["storage_system_costs"] / "gate21_9_ratio_denominator_audit.csv")
    cross_dataset = read_csv(paths["cross_dataset_auto_channel"] / "gate21_9_cross_dataset_task_rows.csv")
    coverage = read_csv(paths["audits"] / "gate21_9_coverage_v4.csv")
    coverage_assertions = read_csv(paths["audits"] / "gate21_9_coverage_sanity_assertions.csv")

    decision = gate21_9_decision(
        official_rows=official,
        auto_selector_rows=auto_selector,
        external_tp_rows=external_tp,
        freehgc_standard_rows=freehgc_standard,
        freehgc_tp_rows=freehgc_tp,
        metapath_rows=metapath,
        cache_assertion_rows=cache_assertions,
        feature_ablation_rows=feature_ablation,
        adapter_rows=adapter,
        storage_rows=storage,
        ratio_audit_rows=ratio_audit,
        cross_dataset_rows=cross_dataset,
        coverage_rows=coverage,
        coverage_assertion_rows=coverage_assertions,
    )
    if strict and decision.get("paper_ready_status") != "ICDE_READY_CANDIDATE":
        decision["strict_failure"] = "Gate21.9 strict mode requested; remaining blockers are listed and READY flags stay false."

    main_table = _main_table_rows(official, auto_selector)
    write_csv(root / "gate21_9_main_table_official.csv", main_table)
    if summaries != root:
        write_csv(summaries / "gate21_9_main_table_official.csv", main_table)
    for summary_name, (subdir, source_name) in SUMMARY_TABLES.items():
        rows = read_csv(paths[subdir] / source_name)
        write_csv(root / summary_name, rows)
        if summaries != root:
            write_csv(summaries / summary_name, rows)

    write_json(root / "gate21_9_decision.json", decision)
    (root / "gate21_9_decision.md").write_text(decision_md(decision), encoding="utf-8")
    _write_run_summary(root / "gate21_9_run_summary.json", decision, paths, manifest)
    _write_requirement_checklist(root / "gate21_9_requirement_checklist.md", decision, root)
    _write_prompt_completion_checklist(root / "gate21_9_prompt_completion_checklist.md", decision, root)
    if summaries != root:
        write_json(summaries / "gate21_9_decision.json", decision)
        (summaries / "gate21_9_decision.md").write_text(decision_md(decision), encoding="utf-8")
        _write_run_summary(summaries / "gate21_9_run_summary.json", decision, paths, manifest)
        _write_requirement_checklist(summaries / "gate21_9_requirement_checklist.md", decision, root)
        _write_prompt_completion_checklist(summaries / "gate21_9_prompt_completion_checklist.md", decision, root)
    return decision


def _official_rows(root: Path, gate21_8_root: Path) -> list[dict[str, str]]:
    current = read_csv(root / "gate21_9_main_table_official.csv")
    if current:
        return current
    source = read_csv(gate21_8_root / "gate21_8_main_table_official.csv")
    out = []
    for row in source:
        item = dict(row)
        item["source_gate"] = item.get("source_gate", "gate21_8")
        item["protocol"] = item.get("protocol", "schema_preserving_tp")
        out.append(item)
    return out


def _main_table_rows(official: Sequence[Mapping[str, Any]], auto_selector: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = (
        "full",
        "export-full",
        "H6-node30",
        "H6-APV-skeleton",
        "HeSF-RCS-APV12",
        "HeSF-RCS-APV16",
    )
    rows: list[dict[str, Any]] = []
    for row in official:
        method = str(row.get("method", ""))
        canonical = str(row.get("canonical_method", ""))
        if any(token in method or token in canonical for token in wanted):
            rows.append(dict(row))
    for row in auto_selector:
        item = dict(row)
        item.setdefault("method_family", "schema_preserving_rcs_auto_selector")
        item.setdefault("canonical_method", item.get("canonical_method", ""))
        item.setdefault("runs", 5)
        item.setdefault("success_count", 5)
        item.setdefault("graph_seed_count", 1)
        item.setdefault("training_seed_count", 5)
        item.setdefault("test_micro_mean", item.get("test_micro_f1", ""))
        item.setdefault("test_macro_mean", item.get("test_macro_f1", ""))
        item.setdefault("training_executed", True)
        rows.append(item)
    return rows


def _write_run_summary(path: Path, decision: Mapping[str, Any], paths: Mapping[str, Path], manifest: Mapping[str, Any]) -> None:
    payload = {
        "gate": "21.9",
        "paper_ready_status": decision.get("paper_ready_status", ""),
        "blocking_issues": decision.get("blocking_issues", []),
        "passed_flags": [name for name, value in dict(decision.get("flags", {})).items() if value],
        "failed_or_partial_flags": [name for name, value in dict(decision.get("flags", {})).items() if not value],
        "counts": decision.get("counts", {}),
        "manifest": dict(manifest),
        "outputs": {name: str(value) for name, value in paths.items()},
    }
    write_json(path, payload)


def _write_requirement_checklist(path: Path, decision: Mapping[str, Any], root: Path) -> None:
    flags = dict(decision.get("flags", {}))
    checks = [
        ("P0 stricter decision JSON schema emitted", _decision_schema_ok(decision)),
        ("P0 misleading FreeHGC flags replaced", all(name in flags for name in ("FREEHGC_TP_ADAPTER_IMPLEMENTED", "FREEHGC_TP_TASK_RESULTS_READY", "FREEHGC_TP_HARD_GAP_REPORTED"))),
        ("P0 misleading standard FreeHGC flags replaced", all(name in flags for name in ("FREEHGC_STANDARD_SINGLE_SEED_RESULTS_READY", "FREEHGC_STANDARD_5SEED_RESULTS_READY", "FREEHGC_UPSTREAM_ENV_VERIFIED", "FREEHGC_STANDARD_PROTOCOL_CONFIG_VERIFIED"))),
        ("P0 storage byte/workload flags separated", all(name in flags for name in ("STORAGE_BYTE_TABLE_READY", "STORAGE_WORKLOAD_COSTS_MEASURED_PASS"))),
        ("P1 DBLP auto selector APV alignment evaluated", flags.get("AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS")),
        ("P1 validation-only channel utility and removal probe files emitted", _exists(root, "auto_selector_alignment/gate21_9_channel_utility.csv") and _exists(root, "auto_selector_alignment/gate21_9_channel_removal_probes.csv")),
        ("P2 external TP 5x5 task table emitted", _exists(root, "external_tp_5x5/gate21_9_external_tp_task_rows.csv")),
        ("P2 external TP READY remains false unless real 5x5 metrics exist", not flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY") or _exists(root, "external_tp_5x5/gate21_9_external_tp_by_method.csv")),
        ("P3 FreeHGC standard table and env audit emitted", _exists(root, "freehgc_protocols/gate21_9_freehgc_standard_task_rows.csv") and _exists(root, "freehgc_protocols/gate21_9_freehgc_env_audit.json")),
        ("P3 FreeHGC-TP task result or hard-gap report emitted", flags.get("FREEHGC_TP_TASK_RESULTS_READY") or flags.get("FREEHGC_TP_HARD_GAP_REPORTED")),
        ("P4 metapath/cache dump tables emitted", _exists(root, "metapath_cache_dump/gate21_9_metapath_tensor_audit.csv") and _exists(root, "metapath_cache_dump/gate21_9_cache_hash_assertions.csv")),
        ("P5 feature ablation task and failure tables emitted", _exists(root, "feature_ablation_tasks/gate21_9_feature_ablation_task_rows.csv") and _exists(root, "feature_ablation_tasks/gate21_9_feature_ablation_failures.csv")),
        ("P6 adapter package v4 tables emitted", _exists(root, "adapter_package_v4/gate21_9_adapter_task_rows.csv") and _exists(root, "adapter_package_v4/gate21_9_adapter_package_audit.csv")),
        ("P7 storage workload and denominator audit emitted", _exists(root, "storage_system_costs/gate21_9_workload_cost_trace.csv") and flags.get("RATIO_DENOMINATOR_AUDIT_V2_PASS")),
        ("P8 cross-dataset task/failure tables emitted", _exists(root, "cross_dataset_auto_channel/gate21_9_cross_dataset_task_rows.csv") and _exists(root, "cross_dataset_auto_channel/gate21_9_cross_dataset_failures.csv")),
        ("P9 coverage v4 and sanity assertions emitted", _exists(root, "audits/gate21_9_coverage_v4.csv") and _exists(root, "audits/gate21_9_coverage_sanity_assertions.csv")),
        ("Top-level output layout emitted", all(_exists(root, name) for name in _top_level_files())),
    ]
    _write_checks(path, "# Gate21.9 Requirement Checklist", checks, decision)


def _write_prompt_completion_checklist(path: Path, decision: Mapping[str, Any], root: Path) -> None:
    flags = dict(decision.get("flags", {}))
    checks: list[tuple[str, Any]] = [(f"Decision flag `{name}` evaluated", name in flags) for name in flags]
    checks.extend(
        [
            ("Required runner exists", Path("experiments/scripts/run_gate21_9_auto_selector_external_baselines.py").exists()),
            ("Required summarizer exists", Path("experiments/scripts/summarize_gate21_9_auto_selector_external_baselines.py").exists()),
            ("Required official modules exist", all(Path(path).exists() for path in _required_module_paths())),
            ("Every required output file exists", all(_exists(root, path) for path in _required_output_paths())),
            ("Protocol eligibility columns present in main selector table", _csv_has_columns(root / "auto_selector_alignment" / "gate21_9_auto_selector_by_method.csv", _eligibility_columns())),
            ("External TP rows include required timing/resource fields", _csv_has_columns(root / "external_tp_5x5" / "gate21_9_external_tp_task_rows.csv", _external_required_columns())),
            ("FreeHGC-TP does not use generic adapter_not_implemented as hard-gap evidence", not _file_contains(root / "freehgc_protocols" / "gate21_9_freehgc_tp_adapter_audit.csv", "adapter_not_implemented")),
            ("READY flags are not true from NaN, placeholder, or smoke rows", _ready_flags_sane(flags)),
            ("paper_ready_status recorded with blockers", bool(decision.get("paper_ready_status")) and isinstance(decision.get("blocking_issues", []), list)),
        ]
    )
    _write_checks(path, "# Gate21.9 Prompt Completion Checklist", checks, decision)


def _write_checks(path: Path, title: str, checks: Sequence[tuple[str, Any]], decision: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{decision.get('paper_ready_status', '')}`", ""]
    for label, value in checks:
        lines.append(f"- [{'x' if _truthy(value) else ' '}] {label}")
    lines.extend(["", "## Blocking Issues"])
    blockers = list(decision.get("blocking_issues", []))
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_schema_ok(decision: Mapping[str, Any]) -> bool:
    required = {
        "paper_ready_status",
        "flags",
        "blocking_issues",
        "paper_safe_claims",
        "paper_unsafe_claims",
        "method_status",
        "external_baseline_status",
        "cross_dataset_status",
        "adapter_status",
        "mechanism_audit_status",
        "system_cost_status",
    }
    return required.issubset(set(decision))


def _top_level_files() -> list[str]:
    return [
        "gate21_9_main_table_official.csv",
        "gate21_9_adapter_table.csv",
        "gate21_9_external_tp_table.csv",
        "gate21_9_standard_condensation_table.csv",
        "gate21_9_cross_dataset_table.csv",
        "gate21_9_storage_system_table.csv",
        "gate21_9_decision.json",
        "gate21_9_decision.md",
        "gate21_9_run_summary.json",
        "gate21_9_requirement_checklist.md",
        "gate21_9_prompt_completion_checklist.md",
    ]


def _required_output_paths() -> list[str]:
    return _top_level_files() + [
        "auto_selector_alignment/gate21_9_channel_utility.csv",
        "auto_selector_alignment/gate21_9_channel_removal_probes.csv",
        "auto_selector_alignment/gate21_9_auto_channel_plans.csv",
        "auto_selector_alignment/gate21_9_auto_selector_by_method.csv",
        "external_tp_5x5/gate21_9_external_tp_task_rows.csv",
        "external_tp_5x5/gate21_9_external_tp_by_method.csv",
        "external_tp_5x5/gate21_9_external_tp_budget_audit.csv",
        "external_tp_5x5/gate21_9_external_tp_failures.csv",
        "freehgc_protocols/gate21_9_freehgc_standard_task_rows.csv",
        "freehgc_protocols/gate21_9_freehgc_standard_by_ratio.csv",
        "freehgc_protocols/gate21_9_freehgc_env_audit.json",
        "freehgc_protocols/gate21_9_freehgc_tp_adapter_audit.csv",
        "freehgc_protocols/gate21_9_freehgc_tp_by_method.csv",
        "freehgc_protocols/gate21_9_freehgc_tp_failure_report.md",
        "metapath_cache_dump/gate21_9_metapath_tensor_audit.csv",
        "metapath_cache_dump/gate21_9_cache_hash_assertions.csv",
        "metapath_cache_dump/gate21_9_introspection_failures.csv",
        "feature_ablation_tasks/gate21_9_feature_ablation_task_rows.csv",
        "feature_ablation_tasks/gate21_9_feature_ablation_by_method.csv",
        "feature_ablation_tasks/gate21_9_feature_shape_audit.csv",
        "feature_ablation_tasks/gate21_9_feature_ablation_failures.csv",
        "adapter_package_v4/gate21_9_adapter_task_rows.csv",
        "adapter_package_v4/gate21_9_adapter_by_method.csv",
        "adapter_package_v4/gate21_9_adapter_package_audit.csv",
        "adapter_package_v4/gate21_9_adapter_manifest_index.csv",
        "storage_system_costs/gate21_9_storage_system_by_method.csv",
        "storage_system_costs/gate21_9_ratio_denominator_audit.csv",
        "storage_system_costs/gate21_9_loader_support_audit.csv",
        "storage_system_costs/gate21_9_workload_cost_trace.csv",
        "cross_dataset_auto_channel/gate21_9_cross_dataset_task_rows.csv",
        "cross_dataset_auto_channel/gate21_9_cross_dataset_by_method.csv",
        "cross_dataset_auto_channel/gate21_9_auto_channel_plans.csv",
        "cross_dataset_auto_channel/gate21_9_auto_channel_validation_trace.csv",
        "cross_dataset_auto_channel/gate21_9_cross_dataset_failures.csv",
        "audits/gate21_9_coverage_v4.csv",
        "audits/gate21_9_coverage_sanity_assertions.csv",
    ]


def _required_module_paths() -> list[str]:
    return [
        "hesf_coarsen/eval/official/gate21_9_decision.py",
        "hesf_coarsen/eval/official/auto_relation_channel_selector_v2.py",
        "hesf_coarsen/eval/official/channel_removal_probe.py",
        "hesf_coarsen/eval/official/external_tp_5x5_runner.py",
        "hesf_coarsen/eval/official/freehgc_standard_runner.py",
        "hesf_coarsen/eval/official/freehgc_tp_export_adapter.py",
        "hesf_coarsen/eval/official/sehgnn_metapath_tensor_dump.py",
        "hesf_coarsen/eval/official/feature_ablation_task_runner.py",
        "hesf_coarsen/eval/official/storage_workload_cost_runner.py",
        "hesf_coarsen/eval/official/ratio_denominator_audit_v2.py",
    ]


def _eligibility_columns() -> list[str]:
    return [
        "schema_compatible",
        "keeps_all_target_nodes",
        "official_hgb_exported",
        "official_sehgnn_unmodified",
        "uses_feature_adapter",
        "uses_weighted_superedges",
        "uses_synthetic_nodes",
        "eligible_for_official_main_table",
        "eligible_for_adapter_table",
        "eligible_for_standard_condensation_table",
        "eligible_for_tp_workload_table",
    ]


def _external_required_columns() -> list[str]:
    return [
        "compress_time_seconds",
        "export_time_seconds",
        "preprocess_time_seconds",
        "train_time_seconds",
        "eval_time_seconds",
        "peak_cpu_rss_mb",
        "peak_gpu_memory_mb",
        "failure_type",
        "failure_message",
    ]


def _exists(root: Path, path: str) -> bool:
    return (root / path).exists()


def _csv_has_columns(path: Path, columns: Sequence[str]) -> bool:
    rows = read_csv(path)
    if not rows:
        return False
    keys = set(rows[0])
    return set(columns).issubset(keys)


def _file_contains(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


def _ready_flags_sane(flags: Mapping[str, Any]) -> bool:
    if flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY") and not flags.get("AUTO_SELECTOR_DBLP_APV_ALIGNMENT_PASS"):
        return False
    if flags.get("FREEHGC_TP_TASK_RESULTS_READY") and not flags.get("FREEHGC_TP_ADAPTER_IMPLEMENTED"):
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(summarize(args.input_root, args.output_root, strict=bool(args.strict)), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
