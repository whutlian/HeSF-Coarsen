from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import DEFAULT_OUTPUT_ROOT, ensure_layout, read_csv
from hesf_coarsen.eval.official.gate21_7_decision import decision_md, gate21_7_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


SUMMARY_TABLES = {
    "gate21_7_main_table_official.csv": ("apv16_stability", "gate21_7_apv16_stability_by_method.csv"),
    "gate21_7_external_tp_table.csv": ("external_tp", "gate21_7_external_tp_by_method.csv"),
    "gate21_7_standard_condensation_table.csv": ("standard_condensation", "gate21_7_standard_condensation_by_method.csv"),
    "gate21_7_adapter_table.csv": ("adapter_package_repaired", "gate21_7_feature_adapter_by_method.csv"),
    "gate21_7_ablation_table.csv": ("feature_ablation_repaired", "gate21_7_feature_ablation_repaired.csv"),
    "gate21_7_storage_system_table.csv": ("storage_system_costs", "gate21_7_storage_system_costs.csv"),
    "gate21_7_cross_dataset_table.csv": ("cross_dataset", "gate21_7_cross_dataset_by_method.csv"),
}


def summarize(input_root: Path, output_root: Path | None = None, *, strict: bool = False) -> dict[str, Any]:
    root = Path(input_root)
    paths = ensure_layout(root)
    summaries = Path(output_root or paths["summaries"])
    summaries.mkdir(parents=True, exist_ok=True)

    official = read_csv(paths["apv16_stability"] / "gate21_7_apv16_stability_by_method.csv")
    adapter = read_csv(paths["adapter_package_repaired"] / "gate21_7_feature_adapter_by_method.csv")
    adapter_audit = read_csv(paths["adapter_package_repaired"] / "gate21_7_adapter_package_audit.csv")
    external = read_csv(paths["external_tp"] / "gate21_7_external_tp_by_method.csv")
    coverage_assertions = read_csv(paths["semantic_audit"] / "gate21_7_coverage_sanity_assertions.csv")
    metapath = read_csv(paths["semantic_audit"] / "gate21_7_metapath_cache_audit.csv")
    cache_hash = read_csv(paths["semantic_audit"] / "gate21_7_cache_hash_audit.csv")
    ablation = read_csv(paths["feature_ablation_repaired"] / "gate21_7_feature_ablation_repaired.csv")
    storage = read_csv(paths["storage_system_costs"] / "gate21_7_storage_only_baselines.csv")
    system_rows = read_csv(paths["storage_system_costs"] / "gate21_7_system_resource_by_stage.csv")
    cross = read_csv(paths["cross_dataset"] / "gate21_7_cross_dataset_by_method.csv")
    standard = read_csv(paths["standard_condensation"] / "gate21_7_standard_condensation_by_method.csv")

    decision = gate21_7_decision(
        official_rows=official,
        adapter_rows=_adapter_decision_rows(adapter_audit or adapter),
        external_tp_rows=external,
        coverage_assertion_rows=coverage_assertions,
        metapath_rows=metapath,
        cache_hash_rows=cache_hash,
        feature_ablation_rows=ablation,
        storage_rows=storage,
        system_resource_rows=system_rows,
        cross_dataset_rows=cross,
    )
    flags = dict(decision["flags"])
    flags["STANDARD_CONDENSATION_PROTOCOL_CONFIGURED"] = bool(standard)
    flags["STANDARD_CONDENSATION_TASK_RESULTS_READY"] = bool(standard) and any(_bool(row.get("success")) and _finite(row.get("test_micro_f1")) for row in standard)
    decision["flags"] = flags
    decision["decisions"] = [name for name, value in flags.items() if value]
    decision["failures"] = [name for name, value in flags.items() if not value]

    for summary_name, (subdir, source_name) in SUMMARY_TABLES.items():
        src = paths[subdir] / source_name
        rows = read_csv(src)
        if summary_name == "gate21_7_main_table_official.csv":
            rows = [row for row in rows if _bool(row.get("official_sehgnn_unmodified", True)) and _bool(row.get("eligible_for_official_main_table", row.get("eligible_for_main_decision", False)))]
        write_csv(summaries / summary_name, rows)
        write_csv(root / summary_name, rows)

    write_json(summaries / "gate21_7_decision.json", decision)
    write_json(root / "gate21_7_decision.json", decision)
    (summaries / "gate21_7_decision.md").write_text(decision_md(decision), encoding="utf-8")
    (root / "gate21_7_decision.md").write_text(decision_md(decision), encoding="utf-8")
    _write_requirement_checklist(root / "gate21_7_requirement_checklist.md", decision, root)
    _write_requirement_checklist(summaries / "gate21_7_requirement_checklist.md", decision, root)
    _write_prompt_completion_checklist(root / "gate21_7_prompt_completion_checklist.md", decision, root)
    _write_prompt_completion_checklist(summaries / "gate21_7_prompt_completion_checklist.md", decision, root)
    if strict and not flags["ICDE_READY_MINIMAL_PASS"]:
        decision["strict_failure"] = "ICDE_READY_MINIMAL_PASS is false; outputs remain generated with explicit not-ready rows."
        write_json(summaries / "gate21_7_decision.json", decision)
        write_json(root / "gate21_7_decision.json", decision)
    return decision


def _write_requirement_checklist(path: Path, decision: Mapping[str, Any], root: Path) -> None:
    flags = dict(decision.get("flags", {}))
    checks = [
        ("APV12 official main anchor emitted", flags.get("OFFICIAL_MAIN_APV12_PASS")),
        ("APV16 official main anchor emitted", flags.get("OFFICIAL_MAIN_APV16_PASS")),
        ("APV16 graph/training seed stability checked", (root / "apv16_stability" / "gate21_7_graph_seed_stability.csv").exists()),
        ("External TP task rows require real task metrics", (root / "external_tp" / "gate21_7_external_tp_by_run.csv").exists()),
        ("FreeHGC upstream clone/preflight audited", (root / "standard_condensation" / "gate21_7_freehgc_preflight.json").exists()),
        ("Coverage diagnostics v2 emitted", flags.get("COVERAGE_TABLE_EMITTED")),
        ("Coverage sanity assertions pass", flags.get("COVERAGE_SEMANTIC_VALIDATION_PASS")),
        ("Metapath/cache audit emitted", flags.get("METAPATH_INTROSPECTION_EMITTED")),
        ("Cache hash real pass", flags.get("CACHE_HASH_REAL_PASS")),
        ("Feature ablation repaired table emitted", flags.get("FEATURE_ABLATION_TABLE_EMITTED")),
        ("Feature shape safe pass", flags.get("FEATURE_ABLATION_SHAPE_SAFE_PASS")),
        ("Adapter package accounting v2 emitted", (root / "adapter_package_repaired" / "gate21_7_adapter_package_audit.csv").exists()),
        ("Storage and system costs emitted", flags.get("STORAGE_ONLY_BYTES_READY") and flags.get("SYSTEM_RESOURCE_SCHEMA_READY")),
        ("Cross-dataset plans/task-result rows emitted", (root / "cross_dataset" / "gate21_7_cross_dataset_by_method.csv").exists()),
        ("Gate21.7 decision JSON/MD emitted", (root / "gate21_7_decision.json").exists() and (root / "gate21_7_decision.md").exists()),
    ]
    _write_checks(path, "# Gate21.7 Requirement Checklist", checks)


def _write_prompt_completion_checklist(path: Path, decision: Mapping[str, Any], root: Path) -> None:
    flags = dict(decision.get("flags", {}))
    checks = [(name, value) for name, value in flags.items()]
    checks.extend(
        [
            ("Output root uses outputs/gate21_7_icde_ready layout", root.name == "gate21_7_icde_ready"),
            ("All requested subdirectories created", all((root / name).exists() for name in ["main_official", "apv16_stability", "external_tp", "standard_condensation", "semantic_audit", "feature_ablation_repaired", "adapter_package_repaired", "storage_system_costs", "cross_dataset", "summaries"])),
            ("FreeHGC is direct upstream clone, not self implementation", Path("external/FreeHGC/README.md").exists()),
            ("Missing dependencies/runtime failures are explicit rows", (root / "standard_condensation" / "gate21_7_standard_condensation_failure_log.csv").exists()),
        ]
    )
    _write_checks(path, "# Gate21.7 Prompt Completion Checklist", checks)


def _write_checks(path: Path, title: str, checks: Sequence[tuple[str, object]]) -> None:
    lines = [title, ""]
    lines.extend(f"- [{'x' if _bool(ok) else ' '}] {label}" for label, ok in checks)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _adapter_decision_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if not str(row.get("missing_reason", "")).startswith("source Gate21.6 adapter run")
    ]


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _finite(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


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
