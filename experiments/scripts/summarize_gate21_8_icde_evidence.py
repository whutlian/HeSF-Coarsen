from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_8_common import DEFAULT_OUTPUT_ROOT, ensure_layout, read_csv
from hesf_coarsen.eval.official.gate21_8_decision import decision_md, gate21_8_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


SUMMARY_TABLES = {
    "gate21_8_main_table_official.csv": ("apv16_5x5", "gate21_8_apv16_by_method.csv"),
    "gate21_8_external_tp_table.csv": ("external_tp_5x5", "gate21_8_external_tp_by_method.csv"),
    "gate21_8_standard_condensation_table.csv": ("freehgc_protocols", "gate21_8_freehgc_standard_by_ratio.csv"),
    "gate21_8_adapter_table.csv": ("adapter_package_v3", "gate21_8_adapter_by_method.csv"),
    "gate21_8_storage_system_table.csv": ("storage_system_costs", "gate21_8_storage_system_by_method.csv"),
    "gate21_8_cross_dataset_table.csv": ("cross_dataset_auto_channel", "gate21_8_cross_dataset_by_method.csv"),
}


def summarize(input_root: Path, output_root: Path | None = None, *, strict: bool = False) -> dict[str, Any]:
    root = Path(input_root)
    paths = ensure_layout(root)
    summaries = Path(output_root or root)
    summaries.mkdir(parents=True, exist_ok=True)

    official = read_csv(paths["apv16_5x5"] / "gate21_8_apv16_by_method.csv")
    apv16_stability = read_csv(paths["apv16_5x5"] / "gate21_8_apv16_graph_seed_stability.csv")
    external_tp = read_csv(paths["external_tp_5x5"] / "gate21_8_external_tp_by_run.csv")
    freehgc_standard = read_csv(paths["freehgc_protocols"] / "gate21_8_freehgc_standard_by_run.csv")
    freehgc_tp = read_csv(paths["freehgc_protocols"] / "gate21_8_freehgc_tp_by_run.csv")
    metapath = read_csv(paths["metapath_cache_dump"] / "gate21_8_metapath_tensor_audit.csv")
    cache_assertions = read_csv(paths["metapath_cache_dump"] / "gate21_8_cache_hash_assertions.csv")
    feature_ablation = read_csv(paths["feature_ablation_tasks"] / "gate21_8_feature_ablation_by_run.csv")
    adapter = read_csv(paths["adapter_package_v3"] / "gate21_8_adapter_by_run.csv")
    storage = read_csv(paths["storage_system_costs"] / "gate21_8_storage_system_by_method.csv")
    ratio_audit = read_csv(paths["storage_system_costs"] / "gate21_8_ratio_denominator_audit.csv")
    cross_dataset = read_csv(paths["cross_dataset_auto_channel"] / "gate21_8_cross_dataset_by_run.csv")

    decision = gate21_8_decision(
        official_rows=official,
        apv16_stability_rows=apv16_stability,
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
    )
    if strict and decision.get("paper_ready_status") != "ICDE_EVIDENCE_READY":
        decision["strict_failure"] = "Gate21.8 strict mode requested but paper_ready_status is not ICDE_EVIDENCE_READY."

    for summary_name, (subdir, source_name) in SUMMARY_TABLES.items():
        rows = read_csv(paths[subdir] / source_name)
        write_csv(root / summary_name, rows)
        if summaries != root:
            write_csv(summaries / summary_name, rows)

    write_json(root / "gate21_8_decision.json", decision)
    (root / "gate21_8_decision.md").write_text(decision_md(decision), encoding="utf-8")
    _write_run_summary(root / "gate21_8_run_summary.json", decision, paths)
    _write_requirement_checklist(root / "gate21_8_requirement_checklist.md", decision, root)
    _write_prompt_completion_checklist(root / "gate21_8_prompt_completion_checklist.md", decision, root)

    if summaries != root:
        write_json(summaries / "gate21_8_decision.json", decision)
        (summaries / "gate21_8_decision.md").write_text(decision_md(decision), encoding="utf-8")
        _write_run_summary(summaries / "gate21_8_run_summary.json", decision, paths)
        _write_requirement_checklist(summaries / "gate21_8_requirement_checklist.md", decision, root)
        _write_prompt_completion_checklist(summaries / "gate21_8_prompt_completion_checklist.md", decision, root)
    return decision


def _write_run_summary(path: Path, decision: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    payload = {
        "gate": "21.8",
        "paper_ready_status": decision.get("paper_ready_status", ""),
        "blocking_issues": decision.get("blocking_issues", []),
        "passed_flags": [name for name, value in dict(decision.get("flags", {})).items() if value],
        "failed_or_partial_flags": [name for name, value in dict(decision.get("flags", {})).items() if not value],
        "counts": decision.get("counts", {}),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_json(path, payload)


def _write_requirement_checklist(path: Path, decision: Mapping[str, Any], root: Path) -> None:
    flags = dict(decision.get("flags", {}))
    checks = [
        ("Output root `outputs/gate21_8_icde_evidence` layout created", root.name == "gate21_8_icde_evidence"),
        ("Every major subdirectory has README", all((root / name / "README.md").exists() for name in _major_subdirs())),
        ("APV12 official main DBLP evidence preserved", flags.get("OFFICIAL_MAIN_DBLP_APV12_READY")),
        ("APV16 training seed stability audited", flags.get("OFFICIAL_MAIN_DBLP_APV16_TRAINING_SEED_STABLE")),
        ("APV16 graph seed 5x5 or deterministic proof audited", flags.get("OFFICIAL_MAIN_DBLP_APV16_GRAPH_SEED_STABLE")),
        ("External TP smoke task results distinguished from 5x5", flags.get("EXTERNAL_TP_SMOKE_TASK_RESULTS_READY") and not flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY")),
        ("External TP required 5x5 readiness evaluated", (root / "external_tp_5x5" / "gate21_8_external_tp_by_run.csv").exists()),
        ("FreeHGC standard and TP protocols separated", flags.get("FREEHGC_STANDARD_PROTOCOL_VERIFIED") and (root / "freehgc_protocols" / "gate21_8_freehgc_tp_by_run.csv").exists()),
        ("FreeHGC upstream clone/env audited; no self implementation", (root / "freehgc_protocols" / "gate21_8_freehgc_env_audit.json").exists()),
        ("Metapath tensor introspection table emitted", (root / "metapath_cache_dump" / "gate21_8_metapath_tensor_audit.csv").exists()),
        ("Cache hash assertions emitted and empty-hash failure explicit", (root / "metapath_cache_dump" / "gate21_8_cache_hash_assertions.csv").exists()),
        ("Feature ablation shape safety emitted", flags.get("FEATURE_ABLATION_SHAPE_SAFE_PASS")),
        ("Adapter package v3 manifests emitted", (root / "adapter_package_v3" / "gate21_8_adapter_manifest_index.csv").exists()),
        ("Storage/system cost rows emitted", (root / "storage_system_costs" / "gate21_8_storage_system_by_method.csv").exists()),
        ("Ratio denominator audit emitted", (root / "storage_system_costs" / "gate21_8_ratio_denominator_audit.csv").exists()),
        ("Cross-dataset auto-channel plans emitted", flags.get("CROSS_DATASET_AUTO_CHANNEL_PLAN_READY")),
        ("Decision JSON/MD emitted", (root / "gate21_8_decision.json").exists() and (root / "gate21_8_decision.md").exists()),
        ("Partial evidence status used when required evidence is missing", decision.get("paper_ready_status") in {"ICDE_EVIDENCE_READY", "ICDE_EVIDENCE_PARTIAL", "NOT_READY"}),
    ]
    _write_checks(path, "# Gate21.8 Requirement Checklist", checks, decision)


def _write_prompt_completion_checklist(path: Path, decision: Mapping[str, Any], root: Path) -> None:
    flags = dict(decision.get("flags", {}))
    checks: list[tuple[str, Any]] = [(f"Decision flag `{name}` evaluated", name in flags) for name in flags]
    checks.extend(
        [
            ("Required CLI runner exists", Path("experiments/scripts/run_gate21_8_icde_evidence.py").exists()),
            ("Required CLI summarizer exists", Path("experiments/scripts/summarize_gate21_8_icde_evidence.py").exists()),
            ("Dry-run path writes manifests without task metrics", (root / "audits" / "gate21_8_dry_run_component_manifest.csv").exists() or not _is_dry_run(root)),
            ("Quick/smoke rows cannot mark 5x5 true", not (_is_quick(root) and flags.get("EXTERNAL_TP_5X5_TASK_RESULTS_READY"))),
            ("Standard condensation is not treated as TP", _protocols_separated(root)),
            ("FreeHGC is cloned/audited from GitHub rather than self-implemented", _freehgc_clone_audited(root)),
            ("APV16 graph-seed decision uses 5x5 or deterministic-proof semantics", (root / "apv16_5x5" / "gate21_8_apv16_graph_seed_stability.csv").exists()),
            ("External TP budget alignment audit exists", (root / "external_tp_5x5" / "gate21_8_external_tp_budget_audit.csv").exists()),
            ("FreeHGC-TP metrics or hard incompatibility report exists", (root / "freehgc_protocols" / "gate21_8_freehgc_tp_failure_report.md").exists()),
            ("Coverage v3 audit emitted", (root / "audits" / "gate21_8_coverage_v3.csv").exists()),
            ("Main summary tables emitted", all((root / name).exists() for name in SUMMARY_TABLES)),
            ("paper_ready_status recorded", bool(decision.get("paper_ready_status"))),
        ]
    )
    _write_checks(path, "# Gate21.8 Prompt Completion Checklist", checks, decision)


def _write_checks(path: Path, title: str, checks: Sequence[tuple[str, Any]], decision: Mapping[str, Any]) -> None:
    lines = [title, "", f"- paper_ready_status: `{decision.get('paper_ready_status', '')}`", ""]
    for label, value in checks:
        lines.append(f"- [{'x' if _truthy(value) else ' '}] {label}")
    blockers = list(decision.get("blocking_issues", []))
    lines.extend(["", "## Missing Or Blocking Evidence"])
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _major_subdirs() -> list[str]:
    return [
        "apv16_5x5",
        "external_tp_5x5",
        "freehgc_protocols",
        "metapath_cache_dump",
        "feature_ablation_tasks",
        "adapter_package_v3",
        "storage_system_costs",
        "cross_dataset_auto_channel",
        "audits",
        "logs",
    ]


def _is_dry_run(root: Path) -> bool:
    manifest = root / "logs" / "gate21_8_run_manifest.json"
    if not manifest.exists():
        return False
    return bool(json.loads(manifest.read_text(encoding="utf-8")).get("dry_run"))


def _is_quick(root: Path) -> bool:
    manifest = root / "logs" / "gate21_8_run_manifest.json"
    if not manifest.exists():
        return False
    return bool(json.loads(manifest.read_text(encoding="utf-8")).get("quick"))


def _protocols_separated(root: Path) -> bool:
    standard = read_csv(root / "freehgc_protocols" / "gate21_8_freehgc_standard_by_run.csv")
    tp = read_csv(root / "freehgc_protocols" / "gate21_8_freehgc_tp_by_run.csv")
    return bool(standard or tp) and all(str(row.get("protocol", "")) == "standard_condensation" for row in standard) and all(
        str(row.get("protocol", "")) == "schema_preserving_tp" for row in tp
    )


def _freehgc_clone_audited(root: Path) -> bool:
    path = root / "freehgc_protocols" / "gate21_8_freehgc_env_audit.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("is_git_clone")) and data.get("repo_url") == "https://github.com/GooLiang/FreeHGC"


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
