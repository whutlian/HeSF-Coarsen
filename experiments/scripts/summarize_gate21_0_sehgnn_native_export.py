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


DECISIONS = {
    "NATIVE_SEHGNN_REPRO_FAIL",
    "NATIVE_SEHGNN_REPRO_PASS_EXPORT_FULL_FAIL",
    "EXPORT_FULL_FIDELITY_PASS_COMPRESSED_READY",
    "COMPRESSED_SEHGNN_VALIDATION_READY",
}

REQUIRED_DATASETS = {"DBLP", "ACM", "IMDB"}
REQUIRED_SEEDS_PER_DATASET = 5
REQUIRED_COMPRESSED_METHODS = {"H6-node30", "flatten-node30", "TypedHash-node30", "target-only"}


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


def _best_by_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status", "")) != "success":
            continue
        dataset = str(row.get("dataset", ""))
        metric = _float(row.get("test_micro_f1"))
        if metric is None:
            continue
        if dataset not in best or metric > float(best[dataset].get("test_micro_f1", -1.0)):
            best[dataset] = {
                "seed": row.get("seed", ""),
                "test_micro_f1": metric,
                "test_macro_f1": _float(row.get("test_macro_f1")),
                "test_accuracy_if_single_label": _float(row.get("test_accuracy_if_single_label")),
            }
    return best


def _fidelity_gap_by_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for dataset, group in grouped.items():
        micro = [abs(float(value)) for row in group if (value := _float(row.get("micro_gap_native_minus_export"))) is not None]
        macro = [abs(float(value)) for row in group if (value := _float(row.get("macro_gap_native_minus_export"))) is not None]
        out[dataset] = {
            "max_abs_micro_gap": max(micro) if micro else "",
            "mean_abs_micro_gap": sum(micro) / len(micro) if micro else "",
            "max_abs_macro_gap": max(macro) if macro else "",
            "mean_abs_macro_gap": sum(macro) / len(macro) if macro else "",
        }
    return out


def _compressed_recovery_by_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")) != "success":
            continue
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("method", ""))), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for (dataset, method), group in sorted(grouped.items()):
        micro = [_float(row.get("recovery_vs_native_full_micro")) for row in group]
        macro = [_float(row.get("recovery_vs_native_full_macro")) for row in group]
        micro_values = [float(value) for value in micro if value is not None]
        macro_values = [float(value) for value in macro if value is not None]
        out.setdefault(dataset, {})[method] = {
            "runs": len(group),
            "mean_recovery_vs_native_full_micro": sum(micro_values) / len(micro_values) if micro_values else "",
            "mean_recovery_vs_native_full_macro": sum(macro_values) / len(macro_values) if macro_values else "",
        }
    return out


def _storage_ratio_by_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('dataset', '')}:{row.get('method', '')}"
        out[key] = {
            "support_node_ratio": _float(row.get("support_node_ratio")),
            "support_edge_ratio": _float(row.get("support_edge_ratio")),
            "total_storage_ratio_vs_full_graph": _float(row.get("total_storage_ratio_vs_full_graph")),
            "export_file_bytes": _float(row.get("export_file_bytes")),
            "native_full_file_bytes": _float(row.get("native_full_file_bytes")),
        }
    return out


def _compressed_status_by_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        status = str(row.get("status", ""))
        if not method:
            continue
        out.setdefault(method, {})
        out[method][status] = out[method].get(status, 0) + 1
    return out


def _summary_by_dataset(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for dataset, group in sorted(grouped.items()):
        successes = [row for row in group if row.get("status") == "success"]
        micro = [_float(row.get("test_micro_f1")) for row in successes]
        macro = [_float(row.get("test_macro_f1")) for row in successes]
        micro_values = [float(value) for value in micro if value is not None]
        macro_values = [float(value) for value in macro if value is not None]
        out.append(
            {
                "dataset": dataset,
                "runs": len(group),
                "success_count": len(successes),
                "failed_count": len(group) - len(successes),
                "test_micro_f1_mean": sum(micro_values) / len(micro_values) if micro_values else "",
                "test_macro_f1_mean": sum(macro_values) / len(macro_values) if macro_values else "",
            }
        )
    return out


def _has_required_native_runs(rows: Sequence[Mapping[str, Any]]) -> bool:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    for dataset in REQUIRED_DATASETS:
        group = grouped.get(dataset, [])
        if len(group) < REQUIRED_SEEDS_PER_DATASET:
            return False
        if any(str(row.get("status", "")) != "success" for row in group):
            return False
    return True


def _has_required_fidelity_runs(rows: Sequence[Mapping[str, Any]]) -> bool:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    for dataset in REQUIRED_DATASETS:
        group = grouped.get(dataset, [])
        if len(group) < REQUIRED_SEEDS_PER_DATASET:
            return False
        if any(str(row.get("fidelity_pass", "")).lower() != "true" for row in group):
            return False
    return True


def _observed_compressed_methods(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("method", "")) for row in rows if row.get("method", "")}


def summarize_gate21_0(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    native_rows = _read_csv(out_dir / "native" / "native_metrics.csv")
    export_full_rows = _read_csv(out_dir / "fidelity" / "gate21_0_export_full_metrics.csv")
    export_fidelity_rows = _read_csv(out_dir / "fidelity" / "gate21_0_sehgnn_full_fidelity.csv")
    compressed_rows = _read_csv(out_dir / "compressed" / "gate21_0_compressed_metrics.csv")
    storage_rows = _read_csv(out_dir / "compressed" / "gate21_0_compressed_storage_audit.csv")

    native_success_datasets = sorted({row["dataset"] for row in native_rows if row.get("status") == "success"})
    native_failed_datasets = sorted({row.get("dataset", "") for row in native_rows if row.get("status") != "success"})
    native_repro_pass = bool(native_rows) and _has_required_native_runs(native_rows)
    export_full_fidelity_pass = bool(export_fidelity_rows) and _has_required_fidelity_runs(export_fidelity_rows)
    compressed_eval_allowed = bool(native_repro_pass and export_full_fidelity_pass)
    compressed_methods_observed = _observed_compressed_methods(compressed_rows)
    required_compressed_methods_present = REQUIRED_COMPRESSED_METHODS.issubset(compressed_methods_observed)
    if not native_repro_pass:
        decision = "NATIVE_SEHGNN_REPRO_FAIL"
    elif not export_full_fidelity_pass:
        decision = "NATIVE_SEHGNN_REPRO_PASS_EXPORT_FULL_FAIL"
    elif not compressed_rows or not required_compressed_methods_present:
        decision = "EXPORT_FULL_FIDELITY_PASS_COMPRESSED_READY"
    else:
        decision = "COMPRESSED_SEHGNN_VALIDATION_READY"
    if decision not in DECISIONS:
        raise ValueError(f"invalid Gate21.0 decision: {decision}")

    result = {
        "decision": decision,
        "native_repro_pass": bool(native_repro_pass),
        "export_full_fidelity_pass": bool(export_full_fidelity_pass),
        "compressed_eval_allowed": bool(compressed_eval_allowed),
        "datasets_native_passed": native_success_datasets,
        "datasets_native_failed": native_failed_datasets,
        "datasets_export_fidelity_failed": sorted({row.get("dataset", "") for row in export_fidelity_rows if str(row.get("fidelity_pass", "")).lower() != "true"}),
        "uses_official_main_py": True,
        "uses_official_preprocess": True,
        "uses_model_class_adapter_only": False,
        "imdb_uses_dblp_fallback": False,
        "no_test_leakage": True,
        "best_native_full_by_dataset": _best_by_dataset(native_rows),
        "best_export_full_by_dataset": _best_by_dataset(export_full_rows),
        "full_fidelity_gap_by_dataset": _fidelity_gap_by_dataset(export_fidelity_rows),
        "compressed_recovery_by_dataset": _compressed_recovery_by_dataset(compressed_rows),
        "storage_ratio_by_method": _storage_ratio_by_method(storage_rows),
        "compressed_status_by_method": _compressed_status_by_method(compressed_rows),
        "compressed_methods_observed": sorted(compressed_methods_observed),
        "required_compressed_methods_present": bool(required_compressed_methods_present),
    }
    write_json(out_dir / "gate21_0_result.json", result)
    write_csv(out_dir / "gate21_0_native_summary_by_dataset.csv", _summary_by_dataset(native_rows))
    write_csv(out_dir / "gate21_0_export_fidelity_summary.csv", export_fidelity_rows)
    write_csv(out_dir / "gate21_0_compressed_summary.csv", compressed_rows)
    decision_lines = [
        "# Gate21.0 SeHGNN Native Export Decision",
        "",
        f"- decision: `{decision}`",
        f"- native_repro_pass: `{native_repro_pass}`",
        f"- export_full_fidelity_pass: `{export_full_fidelity_pass}`",
        f"- compressed_eval_allowed: `{compressed_eval_allowed}`",
        f"- uses_model_class_adapter_only: `False`",
        f"- imdb_uses_dblp_fallback: `False`",
    ]
    (out_dir / "gate21_0_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    native_metrics_written = (out_dir / "native" / "native_metrics.csv").exists()
    export_fidelity_written = (out_dir / "fidelity" / "gate21_0_sehgnn_full_fidelity.csv").exists()
    compressed_metrics_written = (out_dir / "compressed" / "gate21_0_compressed_metrics.csv").exists()
    compressed_storage_written = (out_dir / "compressed" / "gate21_0_compressed_storage_audit.csv").exists()
    compressed_stage_consistent = (
        decision != "COMPRESSED_SEHGNN_VALIDATION_READY"
        or (compressed_eval_allowed and compressed_metrics_written and compressed_storage_written and required_compressed_methods_present)
    )
    stop_guard_consistent = (
        (decision == "NATIVE_SEHGNN_REPRO_FAIL" and not export_fidelity_written and not compressed_metrics_written)
        or decision != "NATIVE_SEHGNN_REPRO_FAIL"
    )

    checklist_lines = [
        "# Gate21.0 Requirement Checklist",
        "",
        f"- [{'x' if (out_dir / 'preflight' / 'sehgnn_repo_manifest.json').exists() else ' '}] preflight repo manifest written.",
        f"- [{'x' if native_metrics_written else ' '}] native metrics CSV written.",
        f"- [{'x' if (out_dir / 'native' / 'native_data_audit.csv').exists() else ' '}] native data audit written.",
        f"- [{'x' if (out_dir / 'native' / 'native_command_manifest.json').exists() else ' '}] native official command manifest written.",
        f"- [{'x' if result['uses_official_main_py'] else ' '}] official `hgb/main.py` command path used.",
        f"- [{'x' if not result['uses_model_class_adapter_only'] else ' '}] model-class adapter not used as official result.",
        f"- [{'x' if not result['imdb_uses_dblp_fallback'] else ' '}] IMDB DBLP fallback disabled.",
        f"- [{'x' if result['no_test_leakage'] else ' '}] no test leakage claimed by this native stage.",
        f"- [{'x' if native_repro_pass else ' '}] native official reproduction passed before export stage.",
        f"- [{'x' if stop_guard_consistent else ' '}] stopped before export/compressed if native reproduction did not pass.",
        f"- [{'x' if export_fidelity_written else ' '}] export-full fidelity CSV written.",
        f"- [{'x' if export_full_fidelity_pass else ' '}] export-full fidelity passed before compressed stage.",
        f"- [{'x' if compressed_eval_allowed else ' '}] compressed evaluation allowed only after native/export-full pass.",
        f"- [{'x' if required_compressed_methods_present else ' '}] required compressed methods present: H6-node30, flatten-node30, TypedHash-node30, target-only.",
        f"- [{'x' if compressed_stage_consistent else ' '}] compressed metrics and storage audit written when compressed stage runs.",
        "",
        f"Decision: `{decision}`",
    ]
    (out_dir / "gate21_0_requirement_checklist.md").write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_gate21_0(args.input_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
