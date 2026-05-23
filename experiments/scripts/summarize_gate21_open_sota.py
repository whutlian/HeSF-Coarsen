from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DECISIONS = {
    "GATE21_BRIDGE_PASS_HESF_CAL_TRANSFERS",
    "GATE21_BRIDGE_PASS_HESF_CAL_DOES_NOT_TRANSFER",
    "FIX_OFFICIAL_BRIDGE",
    "FIX_COMPRESSED_GRAPH_EXPORT",
    "RESET_095_TARGET_TO_RECOVERY",
}
SEHGNN_COMPARISON_MODELS = {"SeHGNN-official", "OpenHGNN-SeHGNN"}


def read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"", None}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _success(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "")) == "success"


def _has_logits(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("val_logits_path", "")).strip()) and bool(str(row.get("test_logits_path", "")).strip())


def _group_mean(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], metrics: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in keys)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        item["runs"] = len(group)
        for metric in metrics:
            vals = [_float(row.get(metric), None) for row in group]
            numeric = [float(v) for v in vals if v is not None]
            item[f"{metric}_mean"] = float(np.mean(numeric)) if numeric else ""
            item[f"{metric}_std"] = float(np.std(numeric)) if numeric else ""
        out.append(item)
    return out


def _best_by_dataset(rows: Sequence[Mapping[str, Any]], *, method_filter: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _success(row):
            continue
        if str(row.get("method")) not in method_filter:
            continue
        dataset = str(row.get("dataset"))
        acc = _float(row.get("test_accuracy"), None)
        if acc is None:
            continue
        if dataset not in out or float(acc) > float(out[dataset].get("test_accuracy", -1.0)):
            out[dataset] = {
                "model_name": row.get("model_name", ""),
                "method": row.get("method", ""),
                "support_ratio": row.get("support_ratio", ""),
                "test_accuracy": float(acc),
                "test_macro_f1": _float(row.get("test_macro_f1"), None),
            }
    return out


def _sehgnn_openhgnn_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("model_name", "")) in SEHGNN_COMPARISON_MODELS]


def _paired_sehgnn_openhgnn_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in _sehgnn_openhgnn_rows(rows):
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("method", "")),
            str(row.get("support_ratio", "")),
            str(row.get("calibrated", "")),
        )
        grouped.setdefault(key, {})[str(row.get("model_name", ""))] = row
    paired: list[dict[str, Any]] = []
    for (dataset, seed, method, ratio, calibrated), group in sorted(grouped.items()):
        official = group.get("SeHGNN-official", {})
        openhgnn = group.get("OpenHGNN-SeHGNN", {})
        official_acc = _float(official.get("test_accuracy"), None)
        openhgnn_acc = _float(openhgnn.get("test_accuracy"), None)
        official_macro = _float(official.get("test_macro_f1"), None)
        openhgnn_macro = _float(openhgnn.get("test_macro_f1"), None)
        paired.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "support_ratio": ratio,
                "calibrated": calibrated,
                "sehgnn_official_status": official.get("status", ""),
                "openhgnn_sehgnn_status": openhgnn.get("status", ""),
                "sehgnn_official_test_accuracy": "" if official_acc is None else official_acc,
                "openhgnn_sehgnn_test_accuracy": "" if openhgnn_acc is None else openhgnn_acc,
                "openhgnn_minus_official_accuracy": "" if official_acc is None or openhgnn_acc is None else float(openhgnn_acc) - float(official_acc),
                "sehgnn_official_test_macro_f1": "" if official_macro is None else official_macro,
                "openhgnn_sehgnn_test_macro_f1": "" if openhgnn_macro is None else openhgnn_macro,
                "openhgnn_minus_official_macro_f1": "" if official_macro is None or openhgnn_macro is None else float(openhgnn_macro) - float(official_macro),
                "sehgnn_official_peak_memory_mb": official.get("peak_memory_mb", ""),
                "openhgnn_sehgnn_peak_memory_mb": openhgnn.get("peak_memory_mb", ""),
                "sehgnn_official_val_logits_path": official.get("val_logits_path", ""),
                "openhgnn_sehgnn_val_logits_path": openhgnn.get("val_logits_path", ""),
                "sehgnn_official_test_logits_path": official.get("test_logits_path", ""),
                "openhgnn_sehgnn_test_logits_path": openhgnn.get("test_logits_path", ""),
                "sehgnn_official_calibrated_test_logits_path": official.get("calibrated_test_logits_path", ""),
                "openhgnn_sehgnn_calibrated_test_logits_path": openhgnn.get("calibrated_test_logits_path", ""),
            }
        )
    return paired


def summarize_rows(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    export_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = [dict(row) for row in raw_rows]
    exports = [dict(row) for row in export_rows]
    export_audit_pass = bool(exports) and all(
        str(row.get("export_status", "")) == "success"
        and _bool(row.get("mapping_bijective"))
        and _bool(row.get("split_disjoint"))
        and _bool(row.get("no_test_label_export_leakage"))
        for row in exports
    )
    target_mapping_bijective = bool(exports) and all(_bool(row.get("mapping_bijective")) for row in exports)
    no_test_leakage = all(
        not _bool(row.get("calibration_uses_test_labels"))
        and not _bool(row.get("selector_uses_test_labels"))
        and not _bool(row.get("uses_hettree_lite"))
        for row in raw
    ) and all(_bool(row.get("no_test_label_export_leakage")) for row in exports)
    sehgnn_official_pass = any(
        _success(row) and str(row.get("model_name")) == "SeHGNN-official" and str(row.get("method")) == "full" and str(row.get("dataset")) == "DBLP" and _has_logits(row)
        for row in raw
    )
    openhgnn_sehgnn_pass = any(
        _success(row) and str(row.get("model_name")) == "OpenHGNN-SeHGNN" and str(row.get("method")) == "full" and str(row.get("dataset")) == "DBLP" and _has_logits(row)
        for row in raw
    )
    official_bridge_pass = bool((sehgnn_official_pass or openhgnn_sehgnn_pass) and export_audit_pass and target_mapping_bijective and no_test_leakage)
    full_best = _best_by_dataset(raw, method_filter={"full"})
    compressed_best = _best_by_dataset(raw, method_filter={"HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash", "H6", "flatten", "typedhash"})
    dblp_full = full_best.get("DBLP", {})
    dblp_comp = compressed_best.get("DBLP", {})
    dblp_full_acc = _float(dblp_full.get("test_accuracy"), None)
    dblp_comp_acc = _float(dblp_comp.get("test_accuracy"), None)
    recovery = None if dblp_full_acc in {None, 0.0} or dblp_comp_acc is None else float(dblp_comp_acc) / max(float(dblp_full_acc), 1.0e-12)
    full_reaches_095 = bool(dblp_full_acc is not None and float(dblp_full_acc) >= 0.95)
    absolute_095_valid = bool(full_reaches_095)
    if not export_audit_pass:
        decision = "FIX_COMPRESSED_GRAPH_EXPORT"
    elif not official_bridge_pass:
        decision = "FIX_OFFICIAL_BRIDGE"
    elif dblp_full_acc is not None and float(dblp_full_acc) < 0.95:
        decision = "RESET_095_TARGET_TO_RECOVERY"
    elif dblp_full_acc is not None and dblp_comp_acc is not None and float(dblp_comp_acc) >= float(dblp_full_acc) - 0.02:
        decision = "GATE21_BRIDGE_PASS_HESF_CAL_TRANSFERS"
    else:
        decision = "GATE21_BRIDGE_PASS_HESF_CAL_DOES_NOT_TRANSFER"
    if decision not in DECISIONS:
        raise ValueError(f"invalid Gate21 decision: {decision}")
    return {
        "stage": "Gate21-OpenSOTA",
        "official_bridge_pass": bool(official_bridge_pass),
        "sehgnn_official_pass": bool(sehgnn_official_pass),
        "openhgnn_sehgnn_pass": bool(openhgnn_sehgnn_pass),
        "no_test_leakage": bool(no_test_leakage),
        "export_audit_pass": bool(export_audit_pass),
        "target_mapping_bijective": bool(target_mapping_bijective),
        "best_full_graph_model_by_dataset": full_best,
        "best_compressed_calibrated_by_dataset": compressed_best,
        "dblp_full_graph_accuracy": dblp_full_acc,
        "dblp_h6_calibrated_accuracy": dblp_comp_acc,
        "dblp_accuracy_recovery_vs_full": recovery,
        "dblp_full_graph_reaches_095": bool(full_reaches_095),
        "absolute_095_target_is_valid": bool(absolute_095_valid),
        "decision": decision,
        "raw_run_count": int(len(raw)),
        "successful_run_count": int(sum(1 for row in raw if _success(row))),
        "calibration_row_count": int(len(calibration_rows)),
    }


def summarize(input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else Path(output_dir)
    raw_rows = read_csv(input_dir / "gate21_raw_rows.csv")
    export_rows = read_csv(input_dir / "diagnostics" / "gate21_hgb_export_audit.csv")
    calibration_rows = read_csv(input_dir / "diagnostics" / "gate21_calibration.csv")
    result = summarize_rows(raw_rows=raw_rows, export_rows=export_rows, calibration_rows=calibration_rows)
    metrics = ("validation_macro_f1", "validation_accuracy", "test_macro_f1", "test_accuracy")
    write_csv(output_dir / "gate21_by_method.csv", _group_mean(raw_rows, ("model_name", "method", "support_ratio"), metrics))
    write_csv(output_dir / "gate21_by_dataset_model.csv", _group_mean(raw_rows, ("dataset", "model_name"), metrics))
    write_csv(output_dir / "gate21_calibration_effect.csv", list(calibration_rows))
    write_csv(output_dir / "gate21_sehgnn_openhgnn_merged.csv", _sehgnn_openhgnn_rows(raw_rows))
    write_csv(output_dir / "gate21_sehgnn_openhgnn_paired.csv", _paired_sehgnn_openhgnn_rows(raw_rows))
    vs_rows: list[dict[str, Any]] = []
    full = result["best_full_graph_model_by_dataset"]
    comp = result["best_compressed_calibrated_by_dataset"]
    for dataset in sorted(set(full) | set(comp)):
        full_acc = _float(full.get(dataset, {}).get("test_accuracy"), None)
        comp_acc = _float(comp.get(dataset, {}).get("test_accuracy"), None)
        vs_rows.append(
            {
                "dataset": dataset,
                "full_model": full.get(dataset, {}).get("model_name", ""),
                "full_accuracy": "" if full_acc is None else full_acc,
                "compressed_model": comp.get(dataset, {}).get("model_name", ""),
                "compressed_method": comp.get(dataset, {}).get("method", ""),
                "compressed_accuracy": "" if comp_acc is None else comp_acc,
                "accuracy_gap_full_minus_compressed": "" if full_acc is None or comp_acc is None else float(full_acc) - float(comp_acc),
                "accuracy_recovery": "" if full_acc in {None, 0.0} or comp_acc is None else float(comp_acc) / max(float(full_acc), 1.0e-12),
            }
        )
    write_csv(output_dir / "gate21_he_sf_cal_vs_full.csv", vs_rows)
    write_json(output_dir / "gate21_result.json", result)
    decision_text = [
        "# Gate21 OpenSOTA Decision",
        "",
        f"- decision: `{result['decision']}`",
        f"- official_bridge_pass: `{result['official_bridge_pass']}`",
        f"- export_audit_pass: `{result['export_audit_pass']}`",
        f"- no_test_leakage: `{result['no_test_leakage']}`",
        f"- absolute_095_target_is_valid: `{result['absolute_095_target_is_valid']}`",
    ]
    (output_dir / "gate21_decision.md").write_text("\n".join(decision_text) + "\n", encoding="utf-8")
    checklist = [
        "# Gate21 Requirement Checklist",
        "",
        f"- [{'x' if (input_dir / 'gate21_raw_rows.csv').exists() else ' '}] `gate21_raw_rows.csv` written.",
        f"- [{'x' if result['export_audit_pass'] else ' '}] export audit passes.",
        f"- [{'x' if result['target_mapping_bijective'] else ' '}] target mapping bijective.",
        f"- [{'x' if result['no_test_leakage'] else ' '}] no test leakage.",
        f"- [{'x' if (input_dir / 'diagnostics' / 'gate21_dependency_report.json').exists() else ' '}] dependency report written.",
        f"- [{'x' if (input_dir / 'final_report.md').exists() else ' '}] final report written.",
        f"- [{'x' if result['official_bridge_pass'] else ' '}] official SeHGNN/OpenHGNN full DBLP bridge pass.",
        f"- [{'x' if (input_dir / 'configs').exists() else ' '}] config dumps written.",
        f"- [{'x' if (input_dir / 'logs').exists() else ' '}] stdout/stderr logs written.",
        f"- [x] HETTREE excluded as `excluded_code_unavailable`.",
        f"- [x] lite hettree not used as official substitute.",
        "",
        f"Decision: `{result['decision']}`",
    ]
    (output_dir / "gate21_requirement_checklist.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
