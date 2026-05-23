from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import write_csv


STRONG_BASELINES = {
    "random-support-only",
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
}
EXCLUDED_FOR_DECISION = {
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    "HeSF-SS-H6-fill-only",
    "HeSF-SS-full-residual-prototype-upperbound",
}
OLD_ABLATIONS = {
    "HeSF-SS-validation-H6-fill",
    "HeSF-SS-H6-fill-only",
    "HeSF-SS-random-fill-after-validation",
    "HeSF-SS-validation-H6-fill-acc0.25",
    "HeSF-SS-validation-H6-fill-acc0.50",
    "HeSF-SS-validation-H6-fill-acc1.00",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _ratio(row: Mapping[str, Any], key: str = "requested_support_ratio") -> float:
    return round(_float(row.get(key), 0.0), 10)


def _status_ok(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "success")) == "success"


def _is_eligible(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method", ""))
    if method in STRONG_BASELINES or method in EXCLUDED_FOR_DECISION or method in OLD_ABLATIONS:
        return False
    if method.startswith("HeSF-") and not _bool(row.get("diagnostic_only", False)):
        return _status_ok(row)
    return _bool(row.get("eligible_for_main_decision", False)) and _status_ok(row)


def _best_baseline(row: Mapping[str, Any], baselines: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    dataset = str(row.get("dataset"))
    seed = str(row.get("seed"))
    requested = _ratio(row)
    candidates = [
        base
        for base in baselines
        if str(base.get("dataset")) == dataset
        and str(base.get("seed")) == seed
        and _status_ok(base)
        and _float(base.get("actual_support_ratio", base.get("realized_support_ratio", base.get("requested_support_ratio"))), 0.0) <= requested + 1.0e-9
    ]
    if not candidates:
        candidates = [
            base
            for base in baselines
            if str(base.get("dataset")) == dataset and str(base.get("seed")) == seed and _ratio(base) == requested and _status_ok(base)
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_float(item.get("macro_f1")), _float(item.get("accuracy"))))


def _pareto_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = [row for row in rows if str(row.get("method")) in STRONG_BASELINES]
    eligible = [row for row in rows if _is_eligible(row)]
    out: list[dict[str, Any]] = []
    for row in eligible:
        method = str(row.get("method"))
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed"))
        requested = _ratio(row)
        actual = _float(row.get("actual_support_ratio", row.get("realized_support_ratio", row.get("requested_support_ratio"))), 0.0)
        macro = _float(row.get("macro_f1"))
        acc = _float(row.get("accuracy"))
        dominated = False
        for other in eligible:
            if other is row:
                continue
            if str(other.get("dataset")) != dataset or str(other.get("seed")) != seed or _ratio(other) != requested:
                continue
            other_actual = _float(other.get("actual_support_ratio", other.get("realized_support_ratio", other.get("requested_support_ratio"))), 0.0)
            other_macro = _float(other.get("macro_f1"))
            other_acc = _float(other.get("accuracy"))
            if other_actual <= actual + 1.0e-9 and other_macro >= macro - 1.0e-12 and other_acc >= acc - 1.0e-12 and (other_macro > macro + 1.0e-12 or other_acc > acc + 1.0e-12 or other_actual < actual - 1.0e-12):
                dominated = True
                break
        baseline = _best_baseline(row, baselines)
        base_macro = _float(baseline.get("macro_f1")) if baseline is not None else 0.0
        base_acc = _float(baseline.get("accuracy")) if baseline is not None else 0.0
        out.append(
            {
                "dataset": dataset,
                "seed": int(_float(seed, 0.0)),
                "method": method,
                "requested_support_ratio": requested,
                "actual_support_ratio": actual,
                "effective_support_node_ratio": _float(row.get("effective_support_node_ratio", actual), actual),
                "represented_support_context_ratio": _float(row.get("represented_support_context_ratio", actual), actual),
                "macro_f1": macro,
                "accuracy": acc,
                "validation_macro_f1": _float(row.get("validation_macro_f1")),
                "validation_accuracy": _float(row.get("validation_accuracy")),
                "best_baseline_method_at_or_above_compression": "" if baseline is None else str(baseline.get("method")),
                "best_baseline_actual_support_ratio": "" if baseline is None else _float(baseline.get("actual_support_ratio", baseline.get("realized_support_ratio", baseline.get("requested_support_ratio"))), 0.0),
                "baseline_macro_f1": base_macro,
                "baseline_accuracy": base_acc,
                "delta_macro_vs_frontier": float(macro - base_macro),
                "delta_accuracy_vs_frontier": float(acc - base_acc),
                "pareto_dominated": bool(dominated),
                "primary_eval_mode": row.get("primary_eval_mode", "compressed_projected"),
                "no_test_leakage": _bool(row.get("no_test_leakage", True)),
            }
        )
    return out


def _validation_selected(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not _status_ok(row):
            continue
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("method")))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        best = max(group, key=lambda item: (_float(item.get("validation_accuracy")), _float(item.get("validation_macro_f1")), _float(item.get("accuracy"))))
        out.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "method": key[2],
                "selected_requested_support_ratio": best.get("requested_support_ratio", ""),
                "validation_macro_f1": best.get("validation_macro_f1", ""),
                "validation_accuracy": best.get("validation_accuracy", ""),
                "macro_f1": best.get("macro_f1", ""),
                "accuracy": best.get("accuracy", ""),
            }
        )
    return out


def _by_dataset_selected(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}}):
        eligible = [row for row in rows if str(row.get("dataset")) == dataset and _is_eligible(row)]
        if not eligible:
            continue
        best = max(eligible, key=lambda item: (_float(item.get("accuracy")), _float(item.get("macro_f1")), _float(item.get("validation_accuracy"))))
        out.append(
            {
                "dataset": dataset,
                "method": best.get("method"),
                "requested_support_ratio": best.get("requested_support_ratio"),
                "macro_f1": best.get("macro_f1"),
                "accuracy": best.get("accuracy"),
                "validation_macro_f1": best.get("validation_macro_f1"),
                "validation_accuracy": best.get("validation_accuracy"),
            }
        )
    return out


def _dataset_metric(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for dataset in sorted({str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}}):
        values = [_float(row.get(metric)) for row in rows if str(row.get("dataset")) == dataset and str(row.get("method")) == method and _status_ok(row)]
        if values:
            out[dataset] = float(np.mean(values))
    return out


def _best_candidate_by_method(rows: Sequence[Mapping[str, Any]], *, method_prefix: str | None = None) -> str:
    eligible = [row for row in rows if _is_eligible(row)]
    if method_prefix is not None:
        eligible = [row for row in eligible if str(row.get("method", "")).startswith(str(method_prefix))]
    if not eligible:
        return ""
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible:
        by_method.setdefault(str(row.get("method")), []).append(row)
    scored: list[tuple[tuple[float, float, float], str]] = []
    for method, group in by_method.items():
        dblp = [row for row in group if str(row.get("dataset")) == "DBLP"]
        primary = dblp or group
        scored.append(((float(np.mean([_float(row.get("accuracy")) for row in primary])), float(np.mean([_float(row.get("macro_f1")) for row in primary])), float(np.mean([_float(row.get("validation_accuracy")) for row in primary]))), method))
    return max(scored)[1]


def _decision(rows: Sequence[Mapping[str, Any]], pareto: Sequence[Mapping[str, Any]]) -> str:
    if not rows or not pareto:
        return "PIVOT_TO_OFFICIAL_EVALUATOR_FIRST"
    if not all(str(row.get("primary_eval_mode", "compressed_projected")) == "compressed_projected" for row in rows if _status_ok(row)):
        return "PIVOT_TO_OFFICIAL_EVALUATOR_FIRST"
    dblp_required = [row for row in pareto if str(row.get("dataset")) == "DBLP" and round(_float(row.get("requested_support_ratio")), 2) in {0.30, 0.70} and not _bool(row.get("pareto_dominated", False))]
    dblp_ok = bool(dblp_required) and all(_float(row.get("delta_macro_vs_frontier")) >= -1.0e-12 and _float(row.get("delta_accuracy_vs_frontier")) >= -0.005 for row in dblp_required)
    imdb_rows = [row for row in pareto if str(row.get("dataset")) == "IMDB" and not _bool(row.get("pareto_dominated", False))]
    imdb_ok = not imdb_rows or max(_float(row.get("delta_accuracy_vs_frontier")) for row in imdb_rows) >= -0.01
    if dblp_ok and imdb_ok:
        return "ENTER_GATE18_MULTI_SEED"
    stc_best = any(str(row.get("method", "")).startswith("HeSF-STC") for row in pareto)
    return "DROP_RAW_SUPPORT_SELECTION_KEEP_FEATURE_CONDENSATION" if stc_best else "CONTINUE_GATE18R_CALIBRATION_OR_CLUSTER_GATING"


def _write_decision_md(path: Path, result: Mapping[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Gate18R Decision",
                "",
                f"- decision: {result.get('decision')}",
                f"- best_method: {result.get('best_method')}",
                f"- gate18_allowed: {result.get('gate18_allowed')}",
                f"- primary_eval_mode: {result.get('primary_eval_mode')}",
                f"- no_test_leakage: {result.get('no_test_leakage')}",
                "",
                "Evaluator ceiling audit: accuracy recovery near 0.95 cannot be interpreted until the official or faithful evaluator reaches that accuracy range on the full graph lite reference.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def summarize(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = input_dir / "gate18r_raw_rows.csv"
    rows = _read_csv(raw_path)
    if input_dir != output_dir and raw_path.exists():
        write_csv(output_dir / "gate18r_raw_rows.csv", rows)
    pareto = _pareto_rows(rows)
    validation = _validation_selected(rows)
    by_dataset = _by_dataset_selected(rows)
    write_csv(output_dir / "gate18r_validation_selected_by_method.csv", validation)
    write_csv(output_dir / "gate18r_pareto_frontier.csv", pareto)
    write_csv(output_dir / "gate18r_by_dataset_selected.csv", by_dataset)
    diagnostics = [
        "gate18r_calibration.csv",
        "gate18r_per_class_metrics.csv",
        "gate18r_confusion_matrix_by_method.csv",
        "gate18r_unit_inventory.csv",
        "gate18r_unit_scores.csv",
        "gate18r_selected_units.csv",
        "gate18r_unit_overlap.csv",
        "gate18r_feature_condensation.csv",
        "gate18r_evaluator_ceiling_audit.csv",
    ]
    for name in diagnostics:
        path = output_dir / name
        if not path.exists():
            write_csv(path, [])
    decision = _decision(rows, pareto)
    best_support_candidate_method = _best_candidate_by_method(rows)
    best_feature_condensation_method = _best_candidate_by_method(rows, method_prefix="HeSF-STC")
    best_method = (
        best_feature_condensation_method
        if decision == "DROP_RAW_SUPPORT_SELECTION_KEEP_FEATURE_CONDENSATION" and best_feature_condensation_method
        else best_support_candidate_method
    )
    primary_modes = {str(row.get("primary_eval_mode", "compressed_projected")) for row in rows if _status_ok(row)}
    typedhash_included = any(str(row.get("method")) == "TypedHash-ChebHeat-support-only" for row in rows)
    no_test_leakage = all(_bool(row.get("no_test_leakage", True)) for row in rows if _status_ok(row))
    result: dict[str, Any] = {
        "stage": "Gate18R",
        "decision": decision,
        "gate18_allowed": bool(decision == "ENTER_GATE18_MULTI_SEED"),
        "best_method": best_method,
        "best_support_candidate_method": best_support_candidate_method,
        "best_feature_condensation_method": best_feature_condensation_method,
        "typedhash_included": bool(typedhash_included),
        "primary_eval_mode": "compressed_projected" if primary_modes == {"compressed_projected"} or not primary_modes else "mixed",
        "no_test_leakage": bool(no_test_leakage),
        "raw_row_count": int(len(rows)),
        "pareto_row_count": int(len(pareto)),
        "full_graph_lite_accuracy_by_dataset": _dataset_metric(rows, "full-graph-hettree-lite-tuned", "accuracy"),
        "full_graph_lite_macro_by_dataset": _dataset_metric(rows, "full-graph-hettree-lite-tuned", "macro_f1"),
        "best_strong_baseline_accuracy_by_dataset": {
            dataset: max(
                [_float(row.get("accuracy")) for row in rows if str(row.get("dataset")) == dataset and str(row.get("method")) in STRONG_BASELINES and _status_ok(row)]
                or [0.0]
            )
            for dataset in sorted({str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}})
        },
        "best_candidate_accuracy_by_dataset": {
            dataset: max([_float(row.get("accuracy")) for row in rows if str(row.get("dataset")) == dataset and _is_eligible(row)] or [0.0])
            for dataset in sorted({str(row.get("dataset")) for row in rows if row.get("dataset") not in {"", None}})
        },
        "evaluator_ceiling_audit": "accuracy target cannot be interpreted as 0.95 recovery until official/faithful evaluator reaches that range if full-graph lite is low",
    }
    with (output_dir / "gate18r_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    _write_decision_md(output_dir / "gate18r_decision.md", result)
    write_csv(
        output_dir / "gate18r_evaluator_ceiling_audit.csv",
        [
            {
                "audit_item": "accuracy_recovery_target",
                "status": "diagnostic_lite_only",
                "message": result["evaluator_ceiling_audit"],
            }
        ],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Gate18R accuracy-first reset outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/gate18r"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summarize(args.input_dir, args.output_dir or args.input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
