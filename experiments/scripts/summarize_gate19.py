from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv


REQUIRED_DATASETS = {"ACM", "DBLP", "IMDB"}
SUPPORT_BASELINES = {
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
}
FULL_STC_METHODS = {"Full-STC-MLP", "Full-STC-MLP-logit-calibrated", "Full-STC-linear", "Full-STC-centroid"}
DIAGNOSTIC_METHOD_PREFIXES = ("ClusterGate-", "HeSF-SS-")


def normalize_header(header: str) -> str:
    text = str(header).strip().strip('"').strip("'").lstrip("\ufeff")
    if text.startswith("\\ufeff"):
        text = text[len("\\ufeff") :]
    text = text.strip().strip('"').strip("'").lstrip("\ufeff")
    if text.startswith("\\ufeff"):
        text = text[len("\\ufeff") :]
    return text


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    with io.StringIO(text) as handle:
        reader = csv.DictReader(handle)
        fieldnames = [normalize_header(name) for name in (reader.fieldnames or [])]
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {}
            for original, normalized in zip(reader.fieldnames or [], fieldnames):
                row[normalized] = raw.get(original, "")
            rows.append(row)
        return rows


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def assert_dataset_integrity(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("raw CSV is empty")
    if "dataset" not in rows[0]:
        raise ValueError("raw CSV lacks dataset column")
    datasets = {str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}}
    if datasets != REQUIRED_DATASETS:
        raise ValueError(f"expected datasets {sorted(REQUIRED_DATASETS)}, got {sorted(datasets)}")


def validation_selected(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "success")) != "success":
            continue
        groups.setdefault((str(row.get("dataset")), str(row.get("method"))), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(groups.items()):
        best = max(group, key=lambda item: (_float(item.get("validation_accuracy")), _float(item.get("validation_macro_f1")), _float(item.get("accuracy"))))
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "selected_requested_budget": best.get("requested_budget", best.get("requested_support_ratio", "")),
                "total_storage_ratio_vs_full_stc": best.get("total_storage_ratio_vs_full_stc", ""),
                "validation_macro_f1": best.get("validation_macro_f1", ""),
                "validation_accuracy": best.get("validation_accuracy", ""),
                "macro_f1": best.get("macro_f1", ""),
                "accuracy": best.get("accuracy", ""),
                "diagnostic_only": best.get("diagnostic_only", ""),
            }
        )
    return out


def _eligible_for_pareto(row: Mapping[str, Any]) -> bool:
    if str(row.get("status", "success")) != "success":
        return False
    if _bool(row.get("method_invalid", False)):
        return False
    method = str(row.get("method", ""))
    if method in FULL_STC_METHODS or any(method.startswith(prefix) for prefix in DIAGNOSTIC_METHOD_PREFIXES):
        return False
    family = str(row.get("method_family", ""))
    return family.startswith("stc") or method.startswith("STC-")


def build_cost_normalized_pareto(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if _eligible_for_pareto(row)]
    out: list[dict[str, Any]] = []
    for row in eligible:
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed"))
        cost = _float(row.get("total_storage_ratio_vs_full_stc"), 0.0)
        macro = _float(row.get("macro_f1"))
        accuracy = _float(row.get("accuracy"))
        dominated = False
        for other in eligible:
            if other is row or str(other.get("dataset")) != dataset or str(other.get("seed")) != seed:
                continue
            other_cost = _float(other.get("total_storage_ratio_vs_full_stc"), 0.0)
            other_macro = _float(other.get("macro_f1"))
            other_accuracy = _float(other.get("accuracy"))
            if (
                other_cost <= cost + 1.0e-12
                and other_macro >= macro - 1.0e-12
                and other_accuracy >= accuracy - 1.0e-12
                and (other_cost < cost - 1.0e-12 or other_macro > macro + 1.0e-12 or other_accuracy > accuracy + 1.0e-12)
            ):
                dominated = True
                break
        if not dominated:
            out.append(
                {
                    "dataset": dataset,
                    "seed": int(_float(seed, 0.0)),
                    "method": row.get("method"),
                    "requested_budget": row.get("requested_budget", ""),
                    "total_storage_ratio_vs_full_stc": cost,
                    "total_storage_ratio_vs_full_graph": _float(row.get("total_storage_ratio_vs_full_graph")),
                    "macro_f1": macro,
                    "accuracy": accuracy,
                    "validation_macro_f1": _float(row.get("validation_macro_f1")),
                    "validation_accuracy": _float(row.get("validation_accuracy")),
                    "feature_cache_size_ratio": _float(row.get("feature_cache_size_ratio")),
                    "path_channel_count_ratio": _float(row.get("path_channel_count_ratio")),
                    "cost_axis_used": "total_storage_ratio_vs_full_stc",
                    "pareto_dominated": False,
                    "primary_eval_mode": row.get("primary_eval_mode", "compressed_projected"),
                    "no_test_leakage": _bool(row.get("no_test_leakage", True)),
                }
            )
    return sorted(out, key=lambda item: (str(item["dataset"]), float(item["total_storage_ratio_vs_full_stc"]), -float(item["accuracy"]), str(item["method"])))


def _best_by_dataset(rows: Sequence[Mapping[str, Any]], predicate, metric: str = "accuracy") -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for dataset in sorted(REQUIRED_DATASETS):
        candidates = [row for row in rows if str(row.get("dataset")) == dataset and predicate(row) and str(row.get("status", "success")) == "success"]
        if candidates:
            out[dataset] = max(candidates, key=lambda item: (_float(item.get(metric)), _float(item.get("macro_f1")), _float(item.get("validation_accuracy"))))
    return out


def _decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    full_stc: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Mapping[str, Any]],
    compressed: Mapping[str, Mapping[str, Any]],
    teacher_kl_valid: bool,
    cost_accounting_pass: bool,
    typedhash_included: bool,
    no_test_leakage: bool,
    primary_eval_mode: str,
) -> tuple[str, bool, list[str]]:
    reasons: list[str] = []
    if primary_eval_mode != "compressed_projected":
        reasons.append("primary_eval_mode_not_compressed_projected")
    if not no_test_leakage:
        reasons.append("test_leakage_detected")
    if not typedhash_included:
        reasons.append("typedhash_missing")
    if set(full_stc) != REQUIRED_DATASETS:
        reasons.append("full_stc_baseline_missing")
        return "FAIL_FULL_STC_BASELINE_MISSING", False, reasons
    if not cost_accounting_pass:
        reasons.append("cost_accounting_failed")
        return "FAIL_STC_COMPRESSION_COST_CONFUNDED", False, reasons
    if not teacher_kl_valid:
        reasons.append("true_distillation_teacher_kl_invalid")
        return "FAIL_TRUE_DISTILLATION_INVALID", False, reasons
    dblp_full = full_stc.get("DBLP")
    dblp_comp = compressed.get("DBLP")
    dblp_support = support.get("DBLP")
    dblp_pass = False
    if dblp_full is not None and dblp_comp is not None and dblp_support is not None:
        dblp_pass = (
            _float(dblp_comp.get("total_storage_ratio_vs_full_stc")) <= 0.70 + 1.0e-12
            and _float(dblp_comp.get("accuracy")) >= _float(dblp_full.get("accuracy")) - 0.01
            and _float(dblp_comp.get("macro_f1")) >= _float(dblp_full.get("macro_f1")) - 0.02
            and _float(dblp_comp.get("accuracy")) - _float(dblp_support.get("accuracy")) >= -0.005
            and _float(dblp_comp.get("macro_f1")) - _float(dblp_support.get("macro_f1")) >= -0.005
        )
    if not dblp_pass:
        reasons.append("dblp_accuracy_or_macro_recovery_failed")
        return "FAIL_DBLP_ACCURACY_RECOVERY", False, reasons
    if _float(dblp_comp.get("total_storage_ratio_vs_full_stc")) >= 1.0:
        reasons.append("best_method_full_cache_ratio_1")
        return "FAIL_STC_COMPRESSION_COST_CONFUNDED", False, reasons
    return "CONTINUE_TO_GATE20_MULTI_SEED_STC", True, reasons


def _write_decision_md(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Gate19 Decision",
        "",
        "## 1. Summary decision",
        f"- decision: {result.get('decision')}",
        f"- gate20_allowed: {result.get('gate20_allowed')}",
        f"- best_gate19_method: {result.get('best_gate19_method')}",
        "",
        "## 2. Full-STC baseline ceilings",
        json.dumps(result.get("best_full_stc_method_by_dataset", {}), indent=2, sort_keys=True),
        "",
        "## 3. Cost-normalized Pareto frontier",
        f"- non_dominated_count: {result.get('pareto_non_dominated_count')}",
        "",
        "## 4. DBLP primary result",
        f"- dblp_gate19_pass: {result.get('dblp_gate19_pass')}",
        "",
        "## 5. IMDB diagnostic result",
        "- IMDB is diagnostic and checked against TypedHash no-collapse criteria in the result JSON fields.",
        "",
        "## 6. ACM saturation sanity",
        "- ACM is sanity-only and is not used for success evidence.",
        "",
        "## 7. Teacher distillation validity",
        f"- teacher_kl_valid: {result.get('teacher_kl_valid')}",
        "",
        "## 8. Cost accounting validity",
        f"- cost_accounting_pass: {result.get('cost_accounting_pass')}",
        "",
        "## 9. Whether Gate20 multi-seed is allowed",
        f"- gate20_allowed: {result.get('gate20_allowed')}",
        "",
        "## 10. Failure reasons or next tasks",
        json.dumps(result.get("failure_reasons", []), indent=2, sort_keys=True),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_dir / "gate19_raw_rows.csv")
    assert_dataset_integrity(rows)
    pareto = build_cost_normalized_pareto(rows)
    selected = validation_selected(rows)
    by_dataset = [
        {
            "dataset": dataset,
            "method": row.get("method"),
            "total_storage_ratio_vs_full_stc": row.get("total_storage_ratio_vs_full_stc", ""),
            "macro_f1": row.get("macro_f1", ""),
            "accuracy": row.get("accuracy", ""),
            "validation_macro_f1": row.get("validation_macro_f1", ""),
            "validation_accuracy": row.get("validation_accuracy", ""),
        }
        for dataset, row in sorted(_best_by_dataset(rows, lambda item: str(item.get("method_family", "")).startswith("stc")).items())
    ]
    write_csv(output_dir / "gate19_validation_selected_by_method.csv", selected)
    write_csv(output_dir / "gate19_pareto_frontier.csv", pareto)
    write_csv(output_dir / "gate19_by_dataset_selected.csv", by_dataset)

    full_stc = _best_by_dataset(rows, lambda item: str(item.get("method")) == "Full-STC-MLP")
    compressed = _best_by_dataset(rows, lambda item: str(item.get("method_family")) == "stc_compressed" and _float(item.get("total_storage_ratio_vs_full_stc")) <= 0.70 + 1.0e-12)
    support = _best_by_dataset(rows, lambda item: str(item.get("method")) in SUPPORT_BASELINES)
    best_gate19_row = compressed.get("DBLP") or next(iter(compressed.values()), {})
    typedhash_included = any(str(row.get("method")) == "TypedHash-ChebHeat-support-only" for row in rows)
    primary_modes = {str(row.get("primary_eval_mode", "")) for row in rows if str(row.get("status", "success")) == "success"}
    primary_eval_mode = "compressed_projected" if primary_modes == {"compressed_projected"} else "mixed"
    no_test_leakage = all(_bool(row.get("no_test_leakage", True)) for row in rows if str(row.get("status", "success")) == "success")
    full_stc_baseline_available = set(full_stc) == REQUIRED_DATASETS
    stc_rows = [row for row in rows if str(row.get("method_family", "")).startswith("stc")]
    cost_accounting_pass = all(_float(row.get("total_storage_ratio_vs_full_stc")) > 0.0 for row in stc_rows)
    distill_rows = [row for row in rows if "true-distill" in str(row.get("method"))]
    teacher_kl_valid = all(str(row.get("teacher_kl_status")) == "valid" for row in distill_rows) if distill_rows else True
    decision, gate20_allowed, reasons = _decision(
        rows,
        full_stc=full_stc,
        support=support,
        compressed=compressed,
        teacher_kl_valid=teacher_kl_valid,
        cost_accounting_pass=cost_accounting_pass,
        typedhash_included=typedhash_included,
        no_test_leakage=no_test_leakage,
        primary_eval_mode=primary_eval_mode,
    )
    dblp_pass = bool(gate20_allowed)
    result: dict[str, Any] = {
        "stage": "Gate19",
        "primary_eval_mode": primary_eval_mode,
        "no_test_leakage": bool(no_test_leakage),
        "typedhash_included": bool(typedhash_included),
        "full_stc_baseline_available": bool(full_stc_baseline_available),
        "best_full_stc_method_by_dataset": {dataset: row.get("method") for dataset, row in full_stc.items()},
        "best_compressed_stc_method_by_dataset": {dataset: row.get("method") for dataset, row in compressed.items()},
        "best_compressed_stc_storage_ratio_by_dataset": {dataset: _float(row.get("total_storage_ratio_vs_full_stc")) for dataset, row in compressed.items()},
        "best_compressed_stc_accuracy_by_dataset": {dataset: _float(row.get("accuracy")) for dataset, row in compressed.items()},
        "best_compressed_stc_macro_by_dataset": {dataset: _float(row.get("macro_f1")) for dataset, row in compressed.items()},
        "best_support_baseline_by_dataset": {dataset: row.get("method") for dataset, row in support.items()},
        "best_support_baseline_accuracy_by_dataset": {dataset: _float(row.get("accuracy")) for dataset, row in support.items()},
        "best_support_baseline_macro_by_dataset": {dataset: _float(row.get("macro_f1")) for dataset, row in support.items()},
        "best_gate19_method": best_gate19_row.get("method", ""),
        "best_gate19_method_storage_ratio": _float(best_gate19_row.get("total_storage_ratio_vs_full_stc")),
        "best_gate19_method_accuracy": _float(best_gate19_row.get("accuracy")),
        "best_gate19_method_macro": _float(best_gate19_row.get("macro_f1")),
        "best_gate19_method_accuracy_gap_vs_full_stc": _float(best_gate19_row.get("accuracy")) - _float(full_stc.get("DBLP", {}).get("accuracy")),
        "best_gate19_method_macro_gap_vs_full_stc": _float(best_gate19_row.get("macro_f1")) - _float(full_stc.get("DBLP", {}).get("macro_f1")),
        "best_gate19_method_accuracy_gap_vs_best_support_baseline": _float(best_gate19_row.get("accuracy")) - _float(support.get("DBLP", {}).get("accuracy")),
        "best_gate19_method_macro_gap_vs_best_support_baseline": _float(best_gate19_row.get("macro_f1")) - _float(support.get("DBLP", {}).get("macro_f1")),
        "teacher_kl_valid": bool(teacher_kl_valid),
        "cost_accounting_pass": bool(cost_accounting_pass),
        "pareto_non_dominated_count": int(len(pareto)),
        "acm_used_for_success_evidence": False,
        "dblp_gate19_pass": bool(dblp_pass),
        "gate20_allowed": bool(gate20_allowed),
        "decision": decision,
        "failure_reasons": reasons,
    }
    (output_dir / "gate19_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_decision_md(output_dir / "gate19_decision.md", result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Gate19 cost-normalized STC outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate19"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir or args.input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
