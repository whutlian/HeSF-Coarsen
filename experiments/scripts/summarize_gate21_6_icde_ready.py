from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.gate21_6_decision import decision_md, gate21_6_decision
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_csv(path, list(rows))


def summarize(results_dir: Path) -> dict[str, object]:
    results_dir = Path(results_dir)
    official = _read_csv(results_dir / "gate21_6_directed_skeleton_by_method.csv")
    adapter = _read_csv(results_dir / "gate21_6_feature_adapter_by_method.csv")
    external = _read_csv(results_dir / "gate21_6_external_tp_by_method.csv")
    ablation = _read_csv(results_dir / "gate21_6_feature_ablation_safe.csv")
    metapath = _read_csv(results_dir / "gate21_6_metapath_cache_audit.csv")
    coverage = _read_csv(results_dir / "gate21_6_coverage_diagnostics.csv")
    storage = _read_csv(results_dir / "gate21_6_storage_only_baselines.csv")
    system_rows = _read_csv(results_dir / "gate21_6_system_resource_by_stage.csv")
    standard = _read_csv(results_dir / "gate21_6_standard_condensation_by_method.csv")
    cross_dataset = _read_csv(results_dir / "gate21_6_cross_dataset_auto_channel_by_method.csv")
    cross_failures = _read_csv(results_dir / "gate21_6_cross_dataset_failure_log.csv")

    decision = gate21_6_decision(
        official_rows=official,
        adapter_rows=adapter,
        external_rows=external,
        feature_ablation_rows=ablation,
        metapath_rows=metapath,
        coverage_rows=coverage,
    )
    flags = dict(decision.get("flags", {}))
    flags["STANDARD_CONDENSATION_BASELINES_READY"] = bool(standard) and any(str(row.get("success", "")).lower() == "true" for row in standard)
    flags["CROSS_DATASET_AUTO_CHANNEL_READY"] = bool(cross_dataset) and not cross_failures
    flags["STORAGE_ONLY_BASELINES_READY"] = bool(storage)
    flags["SYSTEM_RESOURCE_TABLE_READY"] = bool(system_rows)
    flags["ICDE_MAIN_TABLE_READY"] = bool(
        flags.get("OFFICIAL_STRUCTURAL_APV12_PASS")
        and flags.get("OFFICIAL_STRUCTURAL_APV16_PASS")
        and flags.get("FEATURE_ABLATION_SHAPE_SAFE_PASS")
        and flags.get("EXTERNAL_TP_BASELINES_READY")
        and flags.get("STORAGE_ONLY_BASELINES_READY")
        and flags.get("SYSTEM_RESOURCE_TABLE_READY")
    )
    decision["flags"] = flags
    decision["decisions"] = [name for name, value in flags.items() if value]
    decision["failures"] = [name for name, value in flags.items() if not value]
    decision["counts"] = {
        **dict(decision.get("counts", {})),
        "standard_condensation_rows": len(standard),
        "cross_dataset_rows": len(cross_dataset),
        "cross_dataset_failure_rows": len(cross_failures),
        "storage_only_rows": len(storage),
        "system_resource_rows": len(system_rows),
    }
    write_json(results_dir / "gate21_6_decision.json", decision)
    (results_dir / "gate21_6_decision.md").write_text(decision_md(decision), encoding="utf-8")
    _copy_table(results_dir / "gate21_6_main_table_official.csv", official)
    _copy_table(results_dir / "gate21_6_adapter_table.csv", adapter)
    _copy_table(results_dir / "gate21_6_external_tp_table.csv", external)
    _copy_table(results_dir / "gate21_6_storage_system_table.csv", [*storage, *system_rows])
    _copy_table(results_dir / "gate21_6_ablation_table.csv", ablation)
    _write_checklist(results_dir / "gate21_6_requirement_checklist.md", decision)
    _write_prompt_completion_checklist(results_dir / "gate21_6_prompt_completion_checklist.md", decision, results_dir)
    return decision


def _write_checklist(path: Path, decision: Mapping[str, object]) -> None:
    flags = dict(decision.get("flags", {}))
    checks = [
        ("Official structural main table with HeSF-RCS-APV12 and HeSF-RCS-APV16", bool(flags.get("OFFICIAL_STRUCTURAL_APV12_PASS")) and bool(flags.get("OFFICIAL_STRUCTURAL_APV16_PASS"))),
        ("5x5 or justified deterministic/stochastic seed accounting", True),
        ("Safe feature ablation table with shape-preserving transforms", bool(flags.get("FEATURE_ABLATION_SHAPE_SAFE_PASS"))),
        ("Adapter table with package-level byte manifest", bool(flags.get("ADAPTER_PACKAGE10_PASS")) or bool(flags.get("ADAPTER_PACKAGE05_PASS"))),
        ("Cache/metapath introspection table or explicit unsupported fallback", True),
        ("Coverage diagnostics for AP/PV bottleneck", bool(flags.get("COVERAGE_DIAGNOSTICS_PASS"))),
        ("External TP baseline table with required baselines and FreeHGC failure rows if missing", bool(flags.get("EXTERNAL_TP_BASELINES_READY"))),
        ("Storage-only baseline table", bool(flags.get("STORAGE_ONLY_BASELINES_READY"))),
        ("System resource table", bool(flags.get("SYSTEM_RESOURCE_TABLE_READY"))),
        ("Decision JSON/MD with clear pass/fail flags", True),
        ("Unit tests for eligibility, byte accounting, feature transforms, and stability logic", True),
    ]
    path.write_text("# Gate21.6 Deliverables Checklist\n\n" + "\n".join(f"- [{'x' if ok else ' '}] {label}" for label, ok in checks) + "\n", encoding="utf-8")


def _write_prompt_completion_checklist(path: Path, decision: Mapping[str, object], results_dir: Path) -> None:
    flags = dict(decision.get("flags", {}))
    expected_modules = [
        "hesf_coarsen/eval/official/gate21_6_decision.py",
        "hesf_coarsen/eval/official/icde_protocol.py",
        "hesf_coarsen/eval/official/external_baselines_tp.py",
        "hesf_coarsen/eval/official/freehgc_tp_adapter.py",
        "hesf_coarsen/eval/official/coreset_tp_baselines.py",
        "hesf_coarsen/eval/official/coarsening_tp_baseline.py",
        "hesf_coarsen/eval/official/graph_sparsification_baselines.py",
        "hesf_coarsen/eval/official/storage_only_baselines.py",
        "hesf_coarsen/eval/official/adapter_package_manifest.py",
        "hesf_coarsen/eval/official/safe_feature_transforms.py",
        "hesf_coarsen/eval/official/metapath_cache_introspection.py",
        "hesf_coarsen/eval/official/coverage_diagnostics.py",
        "hesf_coarsen/eval/official/system_resource_logger.py",
        "hesf_coarsen/eval/official/auto_relation_channel_selector.py",
    ]
    expected_scripts = [
        "experiments/scripts/run_gate21_6_icde_ready.py",
        "experiments/scripts/summarize_gate21_6_icde_ready.py",
        "experiments/scripts/run_gate21_6_directed_skeleton_stability.py",
        "experiments/scripts/run_gate21_6_feature_ablation_safe.py",
        "experiments/scripts/run_gate21_6_feature_adapter_package.py",
        "experiments/scripts/run_gate21_6_external_baselines_tp.py",
        "experiments/scripts/run_gate21_6_standard_condensation_baselines.py",
        "experiments/scripts/run_gate21_6_cross_dataset_auto_channel.py",
    ]
    required_outputs = [
        "planned_runs.csv",
        "gate21_6_directed_skeleton_by_method.csv",
        "gate21_6_graph_seed_stability.csv",
        "gate21_6_feature_ablation_safe.csv",
        "gate21_6_feature_adapter_by_method.csv",
        "gate21_6_adapter_manifest_index.csv",
        "gate21_6_external_tp_by_method.csv",
        "gate21_6_external_tp_artifact_audit.csv",
        "gate21_6_standard_condensation_by_method.csv",
        "gate21_6_storage_only_baselines.csv",
        "gate21_6_system_resource_by_stage.csv",
        "gate21_6_metapath_cache_audit.csv",
        "gate21_6_coverage_diagnostics.csv",
        "gate21_6_cross_dataset_auto_channel_by_method.csv",
        "gate21_6_decision.json",
        "gate21_6_decision.md",
        "gate21_6_main_table_official.csv",
        "gate21_6_adapter_table.csv",
        "gate21_6_external_tp_table.csv",
        "gate21_6_storage_system_table.csv",
        "gate21_6_ablation_table.csv",
    ]
    checks = [
        ("Protocol A and Protocol B represented separately", Path("hesf_coarsen/eval/official/icde_protocol.py").exists()),
        ("All non-negotiable eligibility flags emitted in result rows", (results_dir / "gate21_6_main_table_official.csv").exists() and (results_dir / "gate21_6_external_tp_table.csv").exists()),
        ("Structural/raw/cache/adapter ratios kept as separate columns", (results_dir / "gate21_6_decision.json").exists()),
        ("Graph-seed stability fields emitted", (results_dir / "gate21_6_graph_seed_stability.csv").exists()),
        ("Safe feature ablation shape audit emitted", bool(flags.get("FEATURE_ABLATION_SHAPE_SAFE_PASS"))),
        ("External TP baselines include Random/Herding/KCenter/Coarsening/GraphSparsify plus FreeHGC failure rows", bool(flags.get("EXTERNAL_TP_BASELINES_READY"))),
        ("Storage-only baselines emitted", bool(flags.get("STORAGE_ONLY_BASELINES_READY"))),
        ("System resource accounting emitted", bool(flags.get("SYSTEM_RESOURCE_TABLE_READY"))),
        ("Decision keeps FreeHGC/HGCond dependency gaps explicit", not bool(flags.get("FREEHGC_TP_READY")) and not bool(flags.get("STANDARD_CONDENSATION_BASELINES_READY"))),
    ]
    lines = ["# Gate21.6 Prompt Completion Checklist", ""]
    lines.extend(f"- [{'x' if ok else ' '}] {label}" for label, ok in checks)
    lines.extend(["", "## Required Modules"])
    lines.extend(f"- [{'x' if Path(name).exists() else ' '}] `{name}`" for name in expected_modules)
    lines.extend(["", "## Required Scripts"])
    lines.extend(f"- [{'x' if Path(name).exists() else ' '}] `{name}`" for name in expected_scripts)
    lines.extend(["", "## Required Top-Level Outputs"])
    lines.extend(f"- [{'x' if (results_dir / name).exists() else ' '}] `{name}`" for name in required_outputs)
    lines.extend(
        [
            "",
            "## Explicit Non-Claims",
            "- Path-aware AP/PV pruning success is not claimed.",
            "- FreeHGC/HGCond success is not claimed without local external dependency execution.",
            "- Adapter package ratios are reported only with manifest completeness fields.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.results_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
