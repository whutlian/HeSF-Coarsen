from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import zipfile
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.gate17_4_h6 import selected_support_representatives_from_assignment
from experiments.scripts.run_gate17_6_accuracy_calibrated_h6_fill import _requested_support_count
from experiments.scripts.run_gate17_support_selection import _split_values
from experiments.scripts.run_gate19_cost_normalized_stc import _edge_count, _eval_task, leakage_audit_row, parse_dataset_seeds
from experiments.scripts.run_gate19_1_calibrated_baseline_audit import (
    _class_counts_from_gate19,
    _copy_gate19_rows,
    _full_contexts,
    _labels_from_nodes,
    _support_cost_row,
    _task_pred_payload,
)
from experiments.scripts.summarize_gate19 import _bool, _float, read_csv
from experiments.scripts.summarize_gate19_2 import summarize
from hesf_coarsen.eval.calibration import apply_logit_calibration, calibration_param_bytes, nested_calibration_split
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.logit_ensemble import (
    calibration_metrics,
    scores_from_logits,
    search_confidence_gated_ensemble,
    search_global_convex_ensemble,
    search_per_class_ensemble,
)
from hesf_coarsen.eval.per_class import confusion_matrix_rows, per_class_lookup, per_class_metrics
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.costs.accounting import CompressionCost, compute_feature_cache_bytes, compute_total_storage_ratio
from hesf_coarsen.task_first.feature_condensation.baselines import (
    cache_with_path_indices,
    evaluate_cache_logits,
    flatten_cache,
    labels_for_cache,
    local_indices,
    quantized_cache,
    select_paths_by_energy,
)
from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache, build_semantic_tree_cache, cache_metadata
from hesf_coarsen.task_first.feature_condensation.support_teacher_distill import (
    teacher_student_diagnostics,
    teacher_unavailable_result,
    train_support_teacher_student,
)


SUPPORT_BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)
HESF_CAL_NAMES = {
    "H6-no-spec-support-only": "HeSF-CAL-H6",
    "flatten-sum-support-only": "HeSF-CAL-flatten",
    "TypedHash-ChebHeat-support-only": "HeSF-CAL-TypedHash",
    "random-support-only": "HeSF-CAL-random-negative-control",
}
LEGACY_CAL_NAMES = {source: f"{source}-logit-calibrated" for source in SUPPORT_BASELINES}
TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
CLASS_BIAS_GRID = (-1.50, -1.00, -0.50, 0.00, 0.50, 1.00, 1.50)
BIAS_L2_GRID = (0.0, 0.01, 0.05)
MACRO_GUARD_EPSILON = 0.005
SPLIT_SEEDS = (11, 22, 33, 44, 55)


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _git_output(args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=False)
    return (completed.stdout if completed.returncode == 0 else completed.stderr).strip()


def _candidate_biases(num_classes: int) -> list[np.ndarray]:
    if int(num_classes) <= 4:
        return [np.asarray(values, dtype=np.float32) for values in product(CLASS_BIAS_GRID, repeat=int(num_classes))]
    biases = [np.zeros(int(num_classes), dtype=np.float32)]
    for cls in range(int(num_classes)):
        for value in CLASS_BIAS_GRID:
            if abs(float(value)) <= 1.0e-12:
                continue
            bias = np.zeros(int(num_classes), dtype=np.float32)
            bias[cls] = float(value)
            biases.append(bias)
    return biases


def _json_vector(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def _fit_repeated_calibration(
    task: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    method: str,
    support_ratio: float,
    repeat_count: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    val_logits = np.asarray(task.get("projected_val_logits", []), dtype=np.float32)
    val_labels = np.asarray(task.get("projected_val_labels", []), dtype=np.int64).reshape(-1)
    test_logits = np.asarray(task.get("projected_test_logits", []), dtype=np.float32)
    test_labels = np.asarray(task.get("projected_test_labels", []), dtype=np.int64).reshape(-1)
    if val_logits.size == 0 or test_logits.size == 0:
        out = dict(task)
        diag = {"calibration_status": "missing_logits", "calibration_uses_test_labels": False, "nested_calibration_pass": False}
        return out, diag, []
    repeats = list(SPLIT_SEEDS[: max(1, int(repeat_count))])
    nested_rows: list[dict[str, Any]] = []
    selected_payloads: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    biases = _candidate_biases(val_logits.shape[1])
    uncal_test = scores_from_logits(test_logits, test_labels)
    for repeat_index, split_seed in enumerate(repeats):
        split = nested_calibration_split(np.arange(len(val_labels), dtype=np.int64), val_labels, seed=int(split_seed))
        calib_idx = split["val_calib"]
        select_idx = split["val_select"]
        uncal_calib = scores_from_logits(val_logits[calib_idx], val_labels[calib_idx])
        uncal_select = scores_from_logits(val_logits[select_idx], val_labels[select_idx])
        min_select_macro = float(uncal_select["macro_f1"]) - MACRO_GUARD_EPSILON
        best: dict[str, Any] | None = None
        best_val_logits: np.ndarray | None = None
        best_test_logits: np.ndarray | None = None
        for temperature in TEMPERATURE_GRID:
            for bias in biases:
                adjusted_val_base = apply_logit_calibration(val_logits, float(temperature), bias)
                adjusted_test_base = apply_logit_calibration(test_logits, float(temperature), bias)
                calib_scores = scores_from_logits(adjusted_val_base[calib_idx], val_labels[calib_idx])
                select_scores = scores_from_logits(adjusted_val_base[select_idx], val_labels[select_idx])
                for l2_penalty in BIAS_L2_GRID:
                    satisfied = float(select_scores["macro_f1"]) >= min_select_macro
                    key = (
                        bool(satisfied),
                        float(select_scores["accuracy"]),
                        float(select_scores["macro_f1"]) - float(l2_penalty) * float(np.linalg.norm(bias)),
                        float(calib_scores["accuracy"]),
                        -abs(float(temperature) - 1.0),
                    )
                    if best is None or key > best["_key"]:
                        best = {
                            "_key": key,
                            "selected_temperature": float(temperature),
                            "selected_class_bias_vector": {str(i): float(bias[i]) for i in range(len(bias))},
                            "bias_l2_penalty": float(l2_penalty),
                            "constraint_satisfied": bool(satisfied),
                            "val_calib_macro": float(calib_scores["macro_f1"]),
                            "val_calib_accuracy": float(calib_scores["accuracy"]),
                            "val_select_macro": float(select_scores["macro_f1"]),
                            "val_select_accuracy": float(select_scores["accuracy"]),
                        }
                        best_val_logits = adjusted_val_base
                        best_test_logits = adjusted_test_base
        assert best is not None and best_val_logits is not None and best_test_logits is not None
        test_scores = scores_from_logits(best_test_logits, test_labels)
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "method": method,
            "support_ratio": float(support_ratio),
            "nested_repeat": int(repeat_index),
            "split_seed": int(split_seed),
            "val_calib_size": int(len(calib_idx)),
            "val_select_size": int(len(select_idx)),
            "selected_temperature": float(best["selected_temperature"]),
            "selected_class_bias_vector": best["selected_class_bias_vector"],
            "bias_l2_penalty": float(best["bias_l2_penalty"]),
            "val_calib_macro": float(best["val_calib_macro"]),
            "val_calib_accuracy": float(best["val_calib_accuracy"]),
            "val_select_macro": float(best["val_select_macro"]),
            "val_select_accuracy": float(best["val_select_accuracy"]),
            "test_macro": float(test_scores["macro_f1"]),
            "test_macro_f1": float(test_scores["macro_f1"]),
            "test_accuracy": float(test_scores["accuracy"]),
            "generalization_gap_accuracy": float(best["val_select_accuracy"] - test_scores["accuracy"]),
            "generalization_gap_macro": float(best["val_select_macro"] - test_scores["macro_f1"]),
            "constraint_satisfied": bool(best["constraint_satisfied"]),
        }
        nested_rows.append(row)
        selected_payloads.append((best, best_val_logits, best_test_logits))
    acc_values = [_float(row.get("test_accuracy")) for row in nested_rows]
    macro_values = [_float(row.get("test_macro")) for row in nested_rows]
    temp_values = [_float(row.get("selected_temperature")) for row in nested_rows]
    bias_l2_values = [float(np.linalg.norm(np.asarray(list(row["selected_class_bias_vector"].values()), dtype=np.float32))) for row in nested_rows]
    constraint_rate = float(np.mean([_bool(row.get("constraint_satisfied")) for row in nested_rows])) if nested_rows else 0.0
    acc_std = _safe_std(acc_values)
    macro_std = _safe_std(macro_values)
    summary = {
        "nested_accuracy_mean": float(np.mean(acc_values)) if acc_values else 0.0,
        "nested_accuracy_std": acc_std,
        "nested_macro_mean": float(np.mean(macro_values)) if macro_values else 0.0,
        "nested_macro_std": macro_std,
        "temperature_std": _safe_std(temp_values),
        "class_bias_l2_mean": float(np.mean(bias_l2_values)) if bias_l2_values else 0.0,
        "class_bias_l2_std": _safe_std(bias_l2_values),
        "constraint_satisfied_rate": constraint_rate,
        "nested_calibration_pass": bool(acc_std <= 0.01 and macro_std <= 0.01 and constraint_rate >= 0.8),
    }
    for row in nested_rows:
        row.update(summary)
    first_best, first_val_logits, first_test_logits = selected_payloads[0]
    val_scores = scores_from_logits(first_val_logits, val_labels)
    test_scores = scores_from_logits(first_test_logits, test_labels)
    out = dict(task)
    out.update(
        {
            "macro_f1": float(test_scores["macro_f1"]),
            "micro_f1": float(test_scores["micro_f1"]),
            "accuracy": float(test_scores["accuracy"]),
            "validation_macro_f1": float(first_best["val_select_macro"]),
            "validation_micro_f1": float(scores_from_logits(first_val_logits, val_labels)["micro_f1"]),
            "validation_accuracy": float(first_best["val_select_accuracy"]),
            "projected_val_logits": np.asarray(first_val_logits, dtype=np.float32).tolist(),
            "projected_test_logits": np.asarray(first_test_logits, dtype=np.float32).tolist(),
            "projected_val_pred": scores_from_logits(first_val_logits, val_labels)["pred"],
            "projected_test_pred": scores_from_logits(first_test_logits, test_labels)["pred"],
        }
    )
    metrics = calibration_metrics(first_test_logits, test_labels)
    diag = {
        "calibration_status": "success",
        "calibration_method": "temperature_scaling;class_bias_grid_search;macro_guarded_accuracy_search",
        "calibration_modes": "temperature_scaling;class_bias_grid_search;macro_guarded_accuracy_search",
        "calibration_split": "val_calib",
        "calibration_selected_on": "val_select",
        "calibration_uses_test_labels": False,
        "calibrator_uses_test_labels": False,
        "temperature": float(first_best["selected_temperature"]),
        "class_bias_vector": first_best["selected_class_bias_vector"],
        "class_bias": first_best["selected_class_bias_vector"],
        "bias_l2_penalty": float(first_best["bias_l2_penalty"]),
        "calibration_constraint_satisfied": bool(first_best["constraint_satisfied"]),
        "constraint_satisfied": bool(first_best["constraint_satisfied"]),
        "calibration_param_bytes": calibration_param_bytes(val_logits.shape[1]),
        "uncalibrated_macro_f1": float(uncal_test["macro_f1"]),
        "uncalibrated_accuracy": float(uncal_test["accuracy"]),
        "calibrated_macro_f1": float(test_scores["macro_f1"]),
        "calibrated_accuracy": float(test_scores["accuracy"]),
        "delta_macro_from_calibration": float(test_scores["macro_f1"] - uncal_test["macro_f1"]),
        "delta_accuracy_from_calibration": float(test_scores["accuracy"] - uncal_test["accuracy"]),
        **metrics,
        **summary,
    }
    return out, diag, nested_rows


def _row_from_task_metrics(
    *,
    dataset: str,
    seed: int,
    method: str,
    method_family: str,
    requested_budget: float,
    source_method: str,
    task: Mapping[str, Any],
    cost: Mapping[str, Any],
    calibrated: bool,
    eligible: bool,
    diagnostic_only: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "stage": "Gate19.2",
        "dataset": str(dataset),
        "seed": int(seed),
        "method": str(method),
        "source_method": str(source_method),
        "method_family": str(method_family),
        "requested_budget": float(requested_budget),
        "requested_support_ratio": float(requested_budget),
        "status": "success",
        "diagnostic_only": bool(diagnostic_only),
        "eligible_for_main_decision": bool(eligible),
        "primary_eval_mode": "compressed_projected",
        "no_test_leakage": True,
        "method_invalid": False,
        "typedhash_included": True,
        "calibrated": bool(calibrated),
        "macro_f1": task.get("macro_f1", ""),
        "micro_f1": task.get("micro_f1", ""),
        "accuracy": task.get("accuracy", ""),
        "validation_macro_f1": task.get("validation_macro_f1", ""),
        "validation_micro_f1": task.get("validation_micro_f1", ""),
        "validation_accuracy": task.get("validation_accuracy", ""),
        "test_macro_f1": task.get("macro_f1", ""),
        "test_accuracy": task.get("accuracy", ""),
    }
    for key in (
        "support_node_ratio",
        "support_edge_ratio",
        "unit_count_ratio",
        "feature_cache_size_ratio",
        "path_channel_count_ratio",
        "feature_cache_bytes",
        "logit_cache_bytes",
        "model_param_bytes",
        "calibration_param_bytes",
        "ensemble_param_bytes",
        "total_storage_bytes",
        "total_inference_storage_bytes",
        "total_storage_ratio_vs_full_stc",
        "total_storage_ratio_vs_full_graph",
        "cost_axis_used",
    ):
        row[key] = cost.get(key, 0)
    if extra:
        row.update(dict(extra))
    return row


def _cost_row_from_compression(cost: CompressionCost, *, calibration_bytes: int = 0, ensemble_bytes: int = 0) -> dict[str, Any]:
    computed = asdict(compute_total_storage_ratio(cost))
    computed["calibration_param_bytes"] = int(calibration_bytes)
    computed["ensemble_param_bytes"] = int(ensemble_bytes)
    computed["total_inference_storage_bytes"] = int(computed["total_storage_bytes"]) + int(calibration_bytes) + int(ensemble_bytes)
    full_stc = max(1, int(cost.full_feature_cache_bytes + cost.full_model_param_bytes + cost.full_logit_cache_bytes))
    full_graph = max(1, int(full_stc + cost.full_support_node_count * 8 + cost.full_support_edge_count * 16 + cost.full_unit_count * 8))
    computed["total_storage_bytes"] = int(computed["total_inference_storage_bytes"])
    computed["total_storage_ratio_vs_full_stc"] = float(computed["total_inference_storage_bytes"] / full_stc)
    computed["total_storage_ratio_vs_full_graph"] = float(computed["total_inference_storage_bytes"] / full_graph)
    computed["cost_axis_used"] = "total_storage_ratio_vs_full_stc"
    return computed


def _ensemble_cost(
    *,
    method: str,
    dataset: str,
    seed: int,
    support_ratio: float,
    sources: Sequence[Mapping[str, Any]],
    full_context: Mapping[str, int],
    class_count: int,
    mode: str,
) -> dict[str, Any]:
    ensemble_bytes = (len(sources) * int(class_count) if mode == "per_class" else len(sources)) * np.dtype(np.float32).itemsize
    total_source_bytes = sum(int(_float(row.get("total_inference_storage_bytes", row.get("total_storage_bytes", 0)))) for row in sources)
    full_stc = max(1, int(full_context["full_feature_cache_bytes"]) + int(full_context["full_model_param_bytes"]))
    full_graph = max(1, full_stc + int(full_context["full_support_node_count"]) * 8 + int(full_context["full_support_edge_count"]) * 16)
    return {
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "requested_budget": float(support_ratio),
        "support_node_ratio": float(sum(_float(row.get("support_node_ratio")) for row in sources)),
        "support_edge_ratio": float(sum(_float(row.get("support_edge_ratio")) for row in sources)),
        "unit_count_ratio": 0.0,
        "feature_cache_size_ratio": 0.0,
        "path_channel_count_ratio": 0.0,
        "feature_cache_bytes": 0,
        "logit_cache_bytes": 0,
        "model_param_bytes": 0,
        "calibration_param_bytes": int(sum(int(_float(row.get("calibration_param_bytes"))) for row in sources)),
        "ensemble_param_bytes": int(ensemble_bytes),
        "total_storage_bytes": int(total_source_bytes + ensemble_bytes),
        "total_inference_storage_bytes": int(total_source_bytes + ensemble_bytes),
        "total_storage_ratio_vs_full_stc": float((total_source_bytes + ensemble_bytes) / full_stc),
        "total_storage_ratio_vs_full_graph": float((total_source_bytes + ensemble_bytes) / full_graph),
        "cost_axis_used": "total_storage_ratio_vs_full_stc",
    }


def _copy_required_gate19_rows(
    *,
    gate19_dir: Path,
    rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    gate19_rows = read_csv(gate19_dir / "gate19_raw_rows.csv")
    gate19_cost = read_csv(gate19_dir / "gate19_cost_breakdown.csv")
    gate19_per_class = read_csv(gate19_dir / "gate19_per_class_metrics.csv")
    gate19_confusion = read_csv(gate19_dir / "gate19_confusion_matrix_by_method.csv")
    full_context, full_denominator = _full_contexts(gate19_cost)
    dataset_class_count = _class_counts_from_gate19(gate19_per_class)
    true_distill_rows: list[dict[str, Any]] = []
    _copy_gate19_rows(
        gate19_rows=gate19_rows,
        gate19_per_class=gate19_per_class,
        gate19_confusion=gate19_confusion,
        rows=rows,
        cost_rows=cost_rows,
        feature_rows=feature_rows,
        true_distill_rows=true_distill_rows,
        teacher_rows=teacher_rows,
        per_class_rows_out=per_class_rows,
        confusion_rows_out=confusion_rows,
        dataset_class_count=dataset_class_count,
        full_denominator=full_denominator,
    )
    for row in rows:
        row["stage"] = "Gate19.2"
        if str(row.get("method_family")) == "full_stc_reference":
            row["method_family"] = "full_stc_reference"
        row["primary_eval_mode"] = "compressed_projected"
    return full_context, full_denominator


def _write_preflight(output_dir: Path) -> None:
    preflight = output_dir / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    status = _git_output(["status", "--short"])
    method_rows = [
        {"method": "HeSF-CAL-H6", "method_family": "hesf_cal_support", "code_path": "experiments/scripts/run_gate19_2_hesf_cal_support_teacher.py", "function": "_fit_repeated_calibration"},
        {"method": "HeSF-CAL-flatten", "method_family": "hesf_cal_support", "code_path": "experiments/scripts/run_gate19_2_hesf_cal_support_teacher.py", "function": "_fit_repeated_calibration"},
        {"method": "HeSF-CAL-TypedHash", "method_family": "hesf_cal_support", "code_path": "experiments/scripts/run_gate19_2_hesf_cal_support_teacher.py", "function": "_fit_repeated_calibration"},
        {"method": "HeSF-CAL-LogitEnsemble", "method_family": "hesf_cal_ensemble", "code_path": "hesf_coarsen/eval/logit_ensemble.py", "function": "search_global_convex_ensemble"},
        {"method": "STC-support-teacher-distill-int8", "method_family": "stc_support_teacher_distill", "code_path": "hesf_coarsen/task_first/feature_condensation/support_teacher_distill.py", "function": "train_support_teacher_student"},
    ]
    write_csv(output_dir / "method_to_code_path.csv", method_rows)
    lines = [
        "# Gate19.2 Code Sync Report",
        "",
        f"- git_commit_sha: `{git_commit_hash()}`",
        f"- branch: `{_git_output(['branch', '--show-current'])}`",
        "- primary_eval_mode_check: compressed_projected is required and enforced by runner.",
        "- calibrated_baseline_eligibility_check: calibrated H6/flatten/TypedHash/best-support are eligible formal baselines.",
        "- header_normalization_check: Gate19 shared `read_csv` strips BOM and quotes.",
        "- dataset_null_check: runner writes dataset on every generated row.",
        "",
        "## Modified / New Files At Preflight",
        "```",
        status,
        "```",
        "",
        "## method_to_code_path",
        "- `outputs/gate19_2/method_to_code_path.csv`",
    ]
    (preflight / "code_sync_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_calibration_shift_report(output_dir: Path, rows: Sequence[Mapping[str, Any]], per_class_rows: Sequence[Mapping[str, Any]]) -> None:
    db_rows = [row for row in rows if str(row.get("dataset")) == "DBLP" and str(row.get("status", "success")) == "success"]
    methods = [
        "H6-no-spec-support-only",
        "H6-no-spec-support-only-logit-calibrated",
        "HeSF-CAL-H6",
        "HeSF-CAL-flatten",
        "HeSF-CAL-TypedHash",
        "HeSF-CAL-LogitEnsemble",
        "Full-STC-MLP-logit-calibrated",
        "STC-feature-cache-quantized-int8",
        "STC-support-teacher-distill-int8",
    ]
    lines = ["# Gate19.2 Calibration Shift Report", "", "## DBLP method comparison", "", "| method | budget | val_acc | test_acc | macro | cost |", "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        candidates = [row for row in db_rows if str(row.get("method")) == method]
        row = max(candidates, key=lambda item: (_float(item.get("validation_accuracy")), _float(item.get("validation_macro_f1")))) if candidates else {}
        lines.append(f"| {method} | {row.get('requested_budget', '')} | {row.get('validation_accuracy', '')} | {row.get('accuracy', '')} | {row.get('macro_f1', '')} | {row.get('total_storage_ratio_vs_full_stc', '')} |")
    lines.extend(["", "## DBLP per-class calibration focus", ""])
    for method in ("HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash"):
        pcs = [row for row in per_class_rows if str(row.get("dataset")) == "DBLP" and str(row.get("method")) == method]
        improved_recall = [str(row.get("class_id")) for row in pcs if _float(row.get("delta_recall_vs_uncalibrated")) > 1.0e-12 or _float(row.get("delta_recall_vs_uncalibrated_source")) > 1.0e-12]
        harmed_recall = [str(row.get("class_id")) for row in pcs if _float(row.get("delta_recall_vs_uncalibrated")) < -1.0e-12 or _float(row.get("delta_recall_vs_uncalibrated_source")) < -1.0e-12]
        lines.append(f"- {method}: recall improved classes = {', '.join(improved_recall) if improved_recall else 'none'}; recall harmed classes = {', '.join(harmed_recall) if harmed_recall else 'none'}.")
    lines.extend(
        [
            "",
            "## Required answers",
            "- Class 1 recall: inspect rows above and `gate19_2_per_class_metrics.csv`.",
            "- Class 0/2/3 recall harm: listed above when present.",
            "- Predicted class prior changes: inspect `gate19_2_confusion_matrix_by_method.csv` predicted totals.",
            "- Stable class-prior correction: checked through repeated nested calibration std and val_select/test gaps.",
            "- Validation overfit: checked in `gate19_2_nested_calibration.csv` via generalization gaps and constraint rate.",
        ]
    )
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (diag_dir / "gate19_2_calibration_shift_report.md").write_text(text, encoding="utf-8")
    (output_dir / "gate19_2_calibration_shift_report.md").write_text(text, encoding="utf-8")


def _write_zip(output_dir: Path, name: str, files: Sequence[str]) -> None:
    with zipfile.ZipFile(output_dir / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in files:
            path = output_dir / file_name
            if path.exists():
                archive.write(path, arcname=file_name)


def _write_requirement_checklist(output_dir: Path, result: Mapping[str, Any]) -> None:
    raw_rows = read_csv(output_dir / "gate19_2_raw_rows.csv")
    nested_rows = read_csv(output_dir / "gate19_2_nested_calibration.csv")
    ensemble_rows = read_csv(output_dir / "gate19_2_logit_ensemble.csv")
    distill_rows = read_csv(output_dir / "gate19_2_support_teacher_distill.csv")
    method_names = {str(row.get("method")) for row in raw_rows}
    datasets = {str(row.get("dataset")) for row in raw_rows if row.get("dataset") not in {"", None}}
    support_ratios = {round(_float(row.get("requested_budget", row.get("support_ratio"))), 2) for row in raw_rows if str(row.get("method")).startswith("HeSF-CAL")}
    feature_ratios = {round(_float(row.get("requested_budget")), 2) for row in raw_rows if str(row.get("method")).startswith("STC-")}
    nested_repeats = {int(_float(row.get("nested_repeat"), -1)) for row in nested_rows}
    distill_methods = {str(row.get("student_method")) for row in distill_rows}
    required = [
        "gate19_2_raw_rows.csv",
        "gate19_2_validation_selected_by_method.csv",
        "gate19_2_by_dataset_selected.csv",
        "gate19_2_calibrated_support_baselines.csv",
        "gate19_2_logit_ensemble.csv",
        "gate19_2_ensemble_weights.csv",
        "gate19_2_support_teacher_distill.csv",
        "gate19_2_cost_breakdown.csv",
        "gate19_2_pareto_frontier.csv",
        "gate19_2_nested_calibration.csv",
        "gate19_2_per_class_metrics.csv",
        "gate19_2_confusion_matrix_by_method.csv",
        "gate19_2_result.json",
        "gate19_2_decision.md",
        "diagnostics/gate19_2_calibration.csv",
        "diagnostics/gate19_2_calibration_shift_report.md",
        "preflight/code_sync_report.md",
    ]
    lines = [
        "# Gate19.2 Requirement Checklist",
        "",
        f"- [{'x' if result.get('primary_eval_mode') == 'compressed_projected' else ' '}] primary_eval_mode = compressed_projected.",
        f"- [{'x' if result.get('no_test_leakage') else ' '}] no test leakage.",
        f"- [{'x' if not result.get('test_oracle_used_for_decision') else ' '}] decision does not use test-oracle.",
        f"- [{'x' if result.get('typedhash_included') else ' '}] TypedHash included.",
        f"- [{'x' if result.get('calibrated_support_baselines_included') else ' '}] calibrated support baselines included.",
        f"- [{'x' if result.get('nested_calibration_audit_pass') else ' '}] repeated nested calibration audit pass.",
        f"- [{'x' if result.get('per_class_confusion_present') else ' '}] per-class and confusion diagnostics present.",
        f"- [x] decision written: `{result.get('decision')}`.",
        "",
        "## Attachment Section Checks",
        f"- [{'x' if datasets >= {'ACM', 'DBLP', 'IMDB'} else ' '}] Section 11 datasets ACM/DBLP/IMDB present.",
        f"- [{'x' if {0.30, 0.50, 0.70, 1.00} <= support_ratios else ' '}] Section 11 support ratios 0.30/0.50/0.70/1.00 present for HeSF-CAL.",
        f"- [{'x' if {0.30, 0.50, 0.70, 1.00} <= feature_ratios else ' '}] Section 11 feature-cache/path ratios 0.30/0.50/0.70/1.00 present for STC rows.",
        f"- [{'x' if len(nested_repeats - {-1}) >= 5 else ' '}] Section 6 repeated nested calibration has 5 repeats.",
        f"- [{'x' if {'HeSF-CAL-H6', 'HeSF-CAL-flatten', 'HeSF-CAL-TypedHash', 'HeSF-CAL-best-support'} <= method_names else ' '}] Section 5 HeSF-CAL single methods present.",
        f"- [{'x' if {'H6-no-spec-support-only-logit-calibrated', 'flatten-sum-support-only-logit-calibrated', 'TypedHash-ChebHeat-support-only-logit-calibrated', 'best-support-baseline-logit-calibrated'} <= method_names else ' '}] Section 1.4 calibrated support baselines are formal rows.",
        f"- [{'x' if {'Full-STC-MLP', 'Full-STC-MLP-logit-calibrated', 'Full-STC-linear', 'Full-STC-centroid'} <= method_names else ' '}] Section 11 STC full-reference methods present.",
        f"- [{'x' if {'HeSF-CAL-LogitEnsemble', 'HeSF-CAL-PerClassEnsemble', 'HeSF-CAL-ConfidenceGatedEnsemble'} <= method_names and len(ensemble_rows) > 0 else ' '}] Section 7 support-logit ensemble rows present.",
        f"- [{'x' if {'STC-support-teacher-distill-int8', 'STC-support-teacher-distill-fp16', 'STC-support-ensemble-teacher-distill-int8', 'STC-support-ensemble-teacher-distill-fp16'} <= distill_methods else ' '}] Section 8 support-teacher STC distillation rows present.",
        f"- [{'x' if (output_dir / 'gate19_2_cost_breakdown.csv').exists() and (output_dir / 'gate19_2_pareto_frontier.csv').exists() else ' '}] Section 9 cost and Pareto outputs present.",
        f"- [{'x' if (output_dir / 'gate19_2_per_class_metrics.csv').exists() and (output_dir / 'gate19_2_confusion_matrix_by_method.csv').exists() else ' '}] Section 10 per-class and confusion outputs present.",
        f"- [{'x' if (output_dir / 'preflight' / 'code_sync_report.md').exists() else ' '}] Section 3 preflight code sync report present.",
        "",
        "## Required Output Files",
    ]
    lines.extend(f"- [{'x' if (output_dir / name).exists() else ' '}] `{name}`" for name in required)
    output_dir.joinpath("gate19_2_requirement_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _teacher_logits_for_cache(cache: SemanticTreeCache, task: Mapping[str, Any]) -> np.ndarray | None:
    train_logits = np.asarray(task.get("projected_train_logits", []), dtype=np.float32)
    val_logits = np.asarray(task.get("projected_val_logits", []), dtype=np.float32)
    test_logits = np.asarray(task.get("projected_test_logits", []), dtype=np.float32)
    if train_logits.size == 0 or val_logits.size == 0 or test_logits.size == 0:
        return None
    class_count = int(train_logits.shape[1])
    out = np.zeros((len(cache.target_nodes), class_count), dtype=np.float32)
    filled = np.zeros(len(cache.target_nodes), dtype=bool)
    lookup = {int(node): idx for idx, node in enumerate(np.asarray(cache.target_nodes, dtype=np.int64).tolist())}
    for split_name, logits in (("train", train_logits), ("val", val_logits), ("test", test_logits)):
        nodes = np.asarray(task.get(f"projected_{split_name}_nodes", []), dtype=np.int64).reshape(-1)
        for local_row, node in enumerate(nodes.tolist()):
            idx = lookup.get(int(node))
            if idx is not None and local_row < len(logits):
                out[idx] = logits[local_row]
                filled[idx] = True
    if not np.any(filled):
        return None
    return out


def _distill_student(
    *,
    dataset: str,
    seed: int,
    student_method: str,
    teacher_method: str,
    teacher_task: Mapping[str, Any],
    teacher_row: Mapping[str, Any],
    full_cache: SemanticTreeCache,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    feature_ratio: float,
    quantization_mode: str,
    full_context: Mapping[str, int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if float(feature_ratio) > 0.70 + 1.0e-12:
        failed = teacher_unavailable_result(dataset=dataset, student_method=student_method, teacher_method=teacher_method, quantization_mode=quantization_mode, feature_cache_size_ratio=feature_ratio, path_channel_count_ratio=feature_ratio)
        failed["teacher_status"] = "compression_constraint_violation"
        failed["method_failed"] = True
        row = {
            "stage": "Gate19.2",
            "dataset": dataset,
            "seed": int(seed),
            "method": student_method,
            "student_method": student_method,
            "teacher_method": teacher_method,
            "method_family": "stc_support_teacher_distill",
            "requested_budget": float(feature_ratio),
            "status": "failed",
            "diagnostic_only": False,
            "eligible_for_main_decision": True,
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": True,
            "method_failed": True,
            **failed,
        }
        return row, failed, [], []
    keep = select_paths_by_energy(full_cache, split, float(feature_ratio))
    pruned = cache_with_path_indices(full_cache, keep)
    bits = 8 if str(quantization_mode) == "int8" else 16
    cache, qdiag = quantized_cache(pruned, bits=bits, budget=1.0, split=split)
    teacher_logits = _teacher_logits_for_cache(cache, teacher_task)
    if teacher_logits is None:
        failed = teacher_unavailable_result(dataset=dataset, student_method=student_method, teacher_method=teacher_method, quantization_mode=quantization_mode, feature_cache_size_ratio=feature_ratio, path_channel_count_ratio=len(cache.paths) / max(1, len(full_cache.paths)))
        row = {
            "stage": "Gate19.2",
            "dataset": dataset,
            "seed": int(seed),
            "method": student_method,
            "student_method": student_method,
            "teacher_method": teacher_method,
            "method_family": "stc_support_teacher_distill",
            "requested_budget": float(feature_ratio),
            "status": "failed",
            "diagnostic_only": False,
            "eligible_for_main_decision": True,
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": True,
            "method_failed": True,
            **failed,
        }
        return row, failed, [], []
    local_labels = labels_for_cache(cache, labels)
    train_idx = local_indices(cache, np.asarray(split["train"], dtype=np.int64))
    features = flatten_cache(cache)
    best_row: dict[str, Any] | None = None
    best_diag: dict[str, Any] | None = None
    for lambda_kl in (0.25, 0.5, 1.0):
        for temperature in (1.0, 2.0, 4.0):
            for lambda_margin in (0.0, 0.1):
                fit = train_support_teacher_student(
                    features,
                    local_labels,
                    train_idx,
                    teacher_logits=teacher_logits,
                    teacher_method=teacher_method,
                    student_method=student_method,
                    seed=int(seed),
                    epochs=max(1, int(args.distill_epochs)),
                    hidden_dim=min(int(args.task_hidden_dim), int(args.distill_hidden_dim)),
                    lambda_kl=float(lambda_kl),
                    teacher_temperature=float(temperature),
                    lambda_margin=float(lambda_margin),
                    device=str(args.device),
                )
                result = evaluate_cache_logits(method=student_method, cache=cache, labels=labels, split=split, logits=np.asarray(fit["logits"], dtype=np.float32), model_param_bytes=int(fit.get("model_param_bytes", 0)))
                val_diag = teacher_student_diagnostics(teacher_logits[result["val_indices"]], result["all_logits"][result["val_indices"]], temperature=float(temperature))
                test_diag = teacher_student_diagnostics(teacher_logits[result["test_indices"]], result["all_logits"][result["test_indices"]], temperature=float(temperature))
                feature_bytes = int(np.asarray(cache.tensor).size * int(qdiag.get("quantized_bytes_per_value", 4)))
                cost = _cost_row_from_compression(
                    CompressionCost(
                        method=student_method,
                        dataset=dataset,
                        seed=int(seed),
                        requested_budget=float(feature_ratio),
                        support_node_count=0,
                        support_edge_count=0,
                        path_channel_count=int(len(cache.paths)),
                        feature_cache_elements=int(np.asarray(cache.tensor).size),
                        feature_cache_bytes=feature_bytes,
                        model_param_bytes=int(result.get("model_param_bytes", 0) or 0),
                        full_support_node_count=int(full_context["full_support_node_count"]),
                        full_support_edge_count=int(full_context["full_support_edge_count"]),
                        full_path_channel_count=int(full_context["full_path_channel_count"]),
                        full_feature_cache_elements=int(full_context["full_feature_cache_elements"]),
                        full_feature_cache_bytes=int(full_context["full_feature_cache_bytes"]),
                        full_model_param_bytes=int(full_context["full_model_param_bytes"]),
                    )
                )
                row = _row_from_task_metrics(
                    dataset=dataset,
                    seed=int(seed),
                    method=student_method,
                    method_family="stc_support_teacher_distill",
                    requested_budget=float(feature_ratio),
                    source_method=teacher_method,
                    task=result,
                    cost=cost,
                    calibrated=False,
                    eligible=True,
                    extra={
                        "student_method": student_method,
                        "teacher_method": teacher_method,
                        "teacher_macro": _float(teacher_row.get("macro_f1")),
                        "teacher_accuracy": _float(teacher_row.get("accuracy")),
                        "student_macro": _float(result.get("macro_f1")),
                        "student_accuracy": _float(result.get("accuracy")),
                        "student_teacher_agreement": test_diag.get("student_teacher_agreement", ""),
                        "student_teacher_KL": test_diag.get("student_teacher_KL", ""),
                        "student_cost_ratio": cost.get("total_storage_ratio_vs_full_stc", ""),
                        "lambda_KL": float(lambda_kl),
                        "teacher_temperature": float(temperature),
                        "lambda_margin": float(lambda_margin),
                        "feature_cache_size_ratio": cost.get("feature_cache_size_ratio", ""),
                        "path_channel_count_ratio": cost.get("path_channel_count_ratio", ""),
                        "quantization_mode": quantization_mode,
                        "teacher_status": test_diag.get("teacher_status", fit.get("teacher_status", "")),
                        "method_failed": bool(fit.get("method_failed", False)),
                        **qdiag,
                    },
                )
                diag = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "student_method": student_method,
                    "teacher_method": teacher_method,
                    "teacher_macro": _float(teacher_row.get("macro_f1")),
                    "teacher_accuracy": _float(teacher_row.get("accuracy")),
                    "student_macro": _float(result.get("macro_f1")),
                    "student_accuracy": _float(result.get("accuracy")),
                    "student_teacher_agreement": test_diag.get("student_teacher_agreement", ""),
                    "student_teacher_KL": test_diag.get("student_teacher_KL", ""),
                    "student_cost_ratio": cost.get("total_storage_ratio_vs_full_stc", ""),
                    "lambda_KL": float(lambda_kl),
                    "teacher_temperature": float(temperature),
                    "lambda_margin": float(lambda_margin),
                    "feature_cache_size_ratio": cost.get("feature_cache_size_ratio", ""),
                    "path_channel_count_ratio": cost.get("path_channel_count_ratio", ""),
                    "quantization_mode": quantization_mode,
                    "teacher_status": test_diag.get("teacher_status", fit.get("teacher_status", "")),
                    "method_failed": bool(fit.get("method_failed", False)),
                }
                key = (_float(row.get("validation_accuracy")), _float(row.get("validation_macro_f1")), -_float(diag.get("student_teacher_KL"), 1.0e9))
                if best_row is None or key > (_float(best_row.get("validation_accuracy")), _float(best_row.get("validation_macro_f1")), -_float(best_diag.get("student_teacher_KL"), 1.0e9) if best_diag else -1.0e9):
                    best_row = row
                    best_diag = diag
    assert best_row is not None and best_diag is not None
    y_true, y_pred = _task_pred_payload(best_row, split_name="test")
    pc = per_class_metrics(
        dataset=dataset,
        seed=int(seed),
        method=student_method,
        method_family="stc_support_teacher_distill",
        requested_budget=float(feature_ratio),
        cost_ratio=_float(best_row.get("total_storage_ratio_vs_full_stc")),
        total_storage_ratio_vs_full_stc=_float(best_row.get("total_storage_ratio_vs_full_stc")),
        calibrated=False,
        source_method=teacher_method,
        y_true=y_true,
        y_pred=y_pred,
    )
    cm = confusion_matrix_rows(dataset=dataset, seed=int(seed), method=student_method, requested_budget=float(feature_ratio), calibrated=False, source_method=teacher_method, y_true=y_true, y_pred=y_pred)
    return best_row, best_diag, pc, cm


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate19.2 requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    _write_preflight(output_dir)

    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    calibrated_support_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    ensemble_weight_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    cache_size_rows: list[dict[str, Any]] = []

    gate19_dir = Path(args.gate19_input_dir)
    full_context_by_dataset, _full_denominator = _copy_required_gate19_rows(
        gate19_dir=gate19_dir,
        rows=rows,
        cost_rows=cost_rows,
        feature_rows=feature_rows,
        teacher_rows=teacher_rows,
        per_class_rows=per_class_rows,
        confusion_rows=confusion_rows,
    )

    for dataset, seed in args.dataset_seed_pairs:
        graph = load_hgb_graph(Path(args.data_root), str(dataset))
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1), dtype=np.int64)
        target_type = infer_target_node_type(graph)
        train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
        split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
        if str(dataset) in full_context_by_dataset:
            full_context = full_context_by_dataset[str(dataset)]
        else:
            full_cache_tmp = build_semantic_tree_cache(graph, target_type=int(target_type), max_hops=2, max_paths=int(args.max_paths))
            full_context = {
                "full_support_node_count": int(np.sum(np.asarray(graph.node_type) != int(target_type))),
                "full_support_edge_count": int(_edge_count(graph)),
                "full_path_channel_count": int(len(full_cache_tmp.paths)),
                "full_feature_cache_elements": int(np.asarray(full_cache_tmp.tensor).size),
                "full_feature_cache_bytes": int(compute_feature_cache_bytes(full_cache_tmp.tensor, np.float32)),
                "full_model_param_bytes": 1,
            }
        full_cache = build_semantic_tree_cache(graph, target_type=int(target_type), max_hops=2, max_paths=int(args.max_paths))
        cache_size_rows.append({"dataset": dataset, "seed": int(seed), **cache_metadata(full_cache), "feature_cache_bytes": int(compute_feature_cache_bytes(full_cache.tensor, np.float32)), "cache_role": "gate19_2_full_cache"})
        dataset_teachers: list[dict[str, Any]] = []
        dataset_ensemble_teachers: list[dict[str, Any]] = []

        for support_ratio in args.support_ratios_parsed:
            support_payloads: dict[str, dict[str, Any]] = {}
            best_uncal_lookup: dict[int, dict[str, float]] = {}
            best_uncal_acc = -1.0
            for source_method in SUPPORT_BASELINES:
                start = perf_counter()
                coarse, assignment, diag = run_support_baseline(graph, baseline=source_method, ratio=float(support_ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                assignment = np.asarray(assignment, dtype=np.int64)
                task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                selected_nodes = selected_support_representatives_from_assignment(graph, assignment, int(target_type))
                support_node_count = int(diag.get("final_support_nodes", len(selected_nodes)))
                support_edge_count = _edge_count(coarse)
                cost = _support_cost_row(method=source_method, dataset=dataset, seed=int(seed), requested_budget=float(support_ratio), support_node_count=support_node_count, support_edge_count=support_edge_count, full_context=full_context)
                extra = {
                    **split_protocol,
                    "selected_support_count": support_node_count,
                    "requested_support_count": _requested_support_count(int(full_context["full_support_node_count"]), float(support_ratio)),
                    "actual_support_ratio": float(support_node_count / max(1, int(full_context["full_support_node_count"]))),
                    "wall_clock_sec": float(perf_counter() - start),
                }
                row = _row_from_task_metrics(dataset=dataset, seed=int(seed), method=source_method, method_family="support_baseline", requested_budget=float(support_ratio), source_method="", task=task, cost=cost, calibrated=False, eligible=True, extra=extra)
                rows.append(row)
                cost_rows.append({**cost, "method": source_method, "dataset": dataset, "seed": int(seed), "requested_budget": float(support_ratio)})
                y_true, y_pred = _task_pred_payload(task, split_name="test")
                pc_rows = per_class_metrics(dataset=dataset, seed=int(seed), method=source_method, method_family="support_baseline", requested_budget=float(support_ratio), cost_ratio=_float(cost.get("total_storage_ratio_vs_full_stc")), total_storage_ratio_vs_full_stc=_float(cost.get("total_storage_ratio_vs_full_stc")), calibrated=False, source_method="", y_true=y_true, y_pred=y_pred, train_labels=_labels_from_nodes(labels, train_nodes), val_labels=_labels_from_nodes(labels, val_nodes))
                per_class_rows.extend(pc_rows)
                confusion_rows.extend(confusion_matrix_rows(dataset=dataset, seed=int(seed), method=source_method, requested_budget=float(support_ratio), calibrated=False, source_method="", y_true=y_true, y_pred=y_pred))
                lookup = per_class_lookup(pc_rows)
                if _float(row.get("accuracy")) > best_uncal_acc:
                    best_uncal_acc = _float(row.get("accuracy"))
                    best_uncal_lookup = lookup
                support_payloads[source_method] = {"task": task, "cost": cost, "row": row, "per_class_lookup": lookup, "support_node_count": support_node_count, "support_edge_count": support_edge_count}
                leak = leakage_audit_row(method=source_method, uses_train_labels=True, uses_val_labels=False, uses_test_labels_before_final_eval=False, calibration_split="none", path_selection_split="train", teacher_training_split="none", student_training_split="train")
                leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_budget": float(support_ratio), **leak})

            hesf_payloads: dict[str, dict[str, Any]] = {}
            legacy_payloads: dict[str, dict[str, Any]] = {}
            for source_method, payload in support_payloads.items():
                legacy_method = LEGACY_CAL_NAMES[source_method]
                hesf_method = HESF_CAL_NAMES[source_method]
                cal_task, cal_diag, nested = _fit_repeated_calibration(payload["task"], dataset=dataset, seed=int(seed), method=hesf_method, support_ratio=float(support_ratio), repeat_count=int(args.nested_calibration_repeats))
                cal_cost = _support_cost_row(method=hesf_method, dataset=dataset, seed=int(seed), requested_budget=float(support_ratio), support_node_count=int(payload["support_node_count"]), support_edge_count=int(payload["support_edge_count"]), full_context=full_context, calibration_bytes=int(cal_diag.get("calibration_param_bytes", 0)))
                for method, family in ((legacy_method, "calibrated_support_baseline"), (hesf_method, "hesf_cal_support")):
                    task_copy = dict(cal_task)
                    row = _row_from_task_metrics(dataset=dataset, seed=int(seed), method=method, method_family=family, requested_budget=float(support_ratio), source_method=source_method, task=task_copy, cost=cal_cost, calibrated=True, eligible=not method.endswith("random-negative-control"), diagnostic_only=source_method == "random-support-only", extra=cal_diag)
                    rows.append(row)
                    if family == "calibrated_support_baseline":
                        calibrated_support_rows.append(row)
                        legacy_payloads[method] = {"task": task_copy, "row": row, "source_method": source_method}
                    else:
                        hesf_payloads[method] = {"task": task_copy, "row": row, "source_method": source_method}
                        if source_method != "random-support-only":
                            dataset_teachers.append({"method": method, "task": task_copy, "row": row})
                    y_true, y_pred = _task_pred_payload(task_copy, split_name="test")
                    per_class = per_class_metrics(
                        dataset=dataset,
                        seed=int(seed),
                        method=method,
                        method_family=family,
                        requested_budget=float(support_ratio),
                        cost_ratio=_float(cal_cost.get("total_storage_ratio_vs_full_stc")),
                        total_storage_ratio_vs_full_stc=_float(cal_cost.get("total_storage_ratio_vs_full_stc")),
                        calibrated=True,
                        source_method=source_method,
                        y_true=y_true,
                        y_pred=y_pred,
                        train_labels=_labels_from_nodes(labels, train_nodes),
                        val_labels=_labels_from_nodes(labels, val_nodes),
                        baseline_per_class=payload["per_class_lookup"],
                        best_uncalibrated_support_per_class=best_uncal_lookup,
                    )
                    for pc in per_class:
                        pc["delta_precision_vs_uncalibrated"] = pc.get("delta_precision_vs_uncalibrated_source", "")
                        pc["delta_recall_vs_uncalibrated"] = pc.get("delta_recall_vs_uncalibrated_source", "")
                        pc["delta_f1_vs_uncalibrated"] = pc.get("delta_f1_vs_uncalibrated_source", "")
                        pc["delta_precision_vs_best_calibrated_support"] = ""
                        pc["delta_recall_vs_best_calibrated_support"] = ""
                        pc["delta_f1_vs_best_calibrated_support"] = ""
                    per_class_rows.extend(per_class)
                    confusion_rows.extend(confusion_matrix_rows(dataset=dataset, seed=int(seed), method=method, requested_budget=float(support_ratio), calibrated=True, source_method=source_method, y_true=y_true, y_pred=y_pred))
                for nested_row in nested:
                    nested_rows.append({**nested_row, "method": hesf_method})
                cal_out = {"dataset": dataset, "seed": int(seed), "method": hesf_method, "source_method": source_method, "requested_budget": float(support_ratio), **cal_diag}
                calibration_rows.append(cal_out)
                cost_rows.append({**cal_cost, "method": hesf_method, "dataset": dataset, "seed": int(seed), "requested_budget": float(support_ratio)})
                leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_budget": float(support_ratio), **leakage_audit_row(method=hesf_method, uses_train_labels=True, uses_val_labels=True, uses_test_labels_before_final_eval=False, calibration_split="val_calib/val_select", path_selection_split="train", teacher_training_split="none", student_training_split="train")})

            eligible_hesf = [payload for name, payload in hesf_payloads.items() if name in {"HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash"}]
            if eligible_hesf:
                best = max(eligible_hesf, key=lambda item: (_float(item["row"].get("validation_accuracy")), _float(item["row"].get("validation_macro_f1")), -_float(item["row"].get("total_storage_ratio_vs_full_stc"))))
                for method, family in (("best-support-baseline-logit-calibrated", "calibrated_support_baseline"), ("HeSF-CAL-best-support", "hesf_cal_support")):
                    best_row = dict(best["row"])
                    best_row["method"] = method
                    best_row["method_family"] = family
                    best_row["source_method"] = best["row"].get("method", "")
                    rows.append(best_row)
                    if family == "calibrated_support_baseline":
                        calibrated_support_rows.append(best_row)
                    else:
                        dataset_teachers.append({"method": method, "task": best["task"], "row": best_row})
                calibration_rows.append({"dataset": dataset, "seed": int(seed), "method": "HeSF-CAL-best-support", "source_method": best["row"].get("method", ""), "requested_budget": float(support_ratio), "calibration_uses_test_labels": False, "selection_rule": "validation_accuracy_then_validation_macro"})

            ensemble_sources = {name: payload for name, payload in hesf_payloads.items() if name in {"HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash"}}
            if len(ensemble_sources) >= 2:
                val_logits = {name: payload["task"]["projected_val_logits"] for name, payload in ensemble_sources.items()}
                test_logits = {name: payload["task"]["projected_test_logits"] for name, payload in ensemble_sources.items()}
                val_labels_arr = next(iter(ensemble_sources.values()))["task"]["projected_val_labels"]
                test_labels_arr = next(iter(ensemble_sources.values()))["task"]["projected_test_labels"]
                macro_floor = max(_float(payload["row"].get("validation_macro_f1")) for payload in ensemble_sources.values()) - MACRO_GUARD_EPSILON
                ensemble_results = {
                    "HeSF-CAL-LogitEnsemble": search_global_convex_ensemble(val_logits, val_labels_arr, test_logits, test_labels_arr, macro_floor=macro_floor, grid_step=float(args.ensemble_grid_step)),
                    "HeSF-CAL-PerClassEnsemble": search_per_class_ensemble(val_logits, val_labels_arr, test_logits, test_labels_arr, macro_floor=macro_floor, l2_penalty=0.01),
                    "HeSF-CAL-ConfidenceGatedEnsemble": search_confidence_gated_ensemble(val_logits, val_labels_arr, test_logits, test_labels_arr, macro_floor=macro_floor, thresholds=(0.50, 0.60, 0.70, 0.80, 0.90)),
                }
                source_rows = [payload["row"] for payload in ensemble_sources.values()]
                for method, ens in ensemble_results.items():
                    ens_cost = _ensemble_cost(method=method, dataset=dataset, seed=int(seed), support_ratio=float(support_ratio), sources=source_rows, full_context=full_context, class_count=len(set(np.asarray(test_labels_arr, dtype=np.int64).tolist())), mode=str(ens["ensemble_mode"]))
                    task = {
                        "macro_f1": ens["test_macro"],
                        "micro_f1": ens["test_accuracy"],
                        "accuracy": ens["test_accuracy"],
                        "validation_macro_f1": ens["val_macro"],
                        "validation_micro_f1": ens["val_accuracy"],
                        "validation_accuracy": ens["val_accuracy"],
                        "projected_val_logits": np.asarray(ens["val_logits"], dtype=np.float32).tolist(),
                        "projected_test_logits": np.asarray(ens["test_logits"], dtype=np.float32).tolist(),
                        "projected_val_labels": val_labels_arr,
                        "projected_test_labels": test_labels_arr,
                        "projected_val_pred": scores_from_logits(ens["val_logits"], val_labels_arr)["pred"],
                        "projected_test_pred": scores_from_logits(ens["test_logits"], test_labels_arr)["pred"],
                    }
                    row = _row_from_task_metrics(dataset=dataset, seed=int(seed), method=method, method_family="hesf_cal_ensemble", requested_budget=float(support_ratio), source_method=";".join(ens["source_methods"]), task=task, cost=ens_cost, calibrated=True, eligible=True, extra={"ensemble_mode": ens["ensemble_mode"], "source_methods": ens["source_methods"], "weights": ens["weights"], "per_class_weights": ens["per_class_weights"], "confidence_threshold": ens["confidence_threshold"], "calibration_uses_test_labels": False, "ECE": ens["ECE"], "NLL": ens["NLL"], "Brier": ens["Brier"]})
                    rows.append(row)
                    ensemble_rows.append(row)
                    ensemble_weight_rows.append({"dataset": dataset, "seed": int(seed), "method": method, "support_ratio": float(support_ratio), "ensemble_mode": ens["ensemble_mode"], "source_methods": ens["source_methods"], "weights": ens["weights"], "per_class_weights": ens["per_class_weights"], "confidence_threshold": ens["confidence_threshold"], "val_macro": ens["val_macro"], "val_accuracy": ens["val_accuracy"], "test_macro": ens["test_macro"], "test_accuracy": ens["test_accuracy"], "ECE": ens["ECE"], "NLL": ens["NLL"], "Brier": ens["Brier"]})
                    y_true = np.asarray(test_labels_arr, dtype=np.int64)
                    y_pred = np.asarray(task["projected_test_pred"], dtype=np.int64)
                    per_class_rows.extend(per_class_metrics(dataset=dataset, seed=int(seed), method=method, method_family="hesf_cal_ensemble", requested_budget=float(support_ratio), cost_ratio=_float(ens_cost.get("total_storage_ratio_vs_full_stc")), total_storage_ratio_vs_full_stc=_float(ens_cost.get("total_storage_ratio_vs_full_stc")), calibrated=True, source_method=row["source_method"], y_true=y_true, y_pred=y_pred))
                    confusion_rows.extend(confusion_matrix_rows(dataset=dataset, seed=int(seed), method=method, requested_budget=float(support_ratio), calibrated=True, source_method=row["source_method"], y_true=y_true, y_pred=y_pred))
                    if method == "HeSF-CAL-LogitEnsemble":
                        # Teacher logits are available on val/test; train logits remain unavailable for ensembles.
                        dataset_ensemble_teachers.append({"method": method, "task": task, "row": row})

        best_single_teacher = max(dataset_teachers, key=lambda item: (_float(item["row"].get("validation_accuracy")), _float(item["row"].get("validation_macro_f1")))) if dataset_teachers else None
        best_ensemble_teacher = max(dataset_ensemble_teachers, key=lambda item: (_float(item["row"].get("validation_accuracy")), _float(item["row"].get("validation_macro_f1")))) if dataset_ensemble_teachers else None
        distill_teacher_specs: list[tuple[str, dict[str, Any] | None]] = [
            ("support", best_single_teacher),
            ("ensemble", best_ensemble_teacher),
        ]
        if _bool_arg(args.run_support_teacher_distill):
            for teacher_kind, teacher in distill_teacher_specs:
                for feature_ratio in args.feature_cache_ratios_parsed:
                    for quantization in ("int8", "fp16"):
                        if teacher_kind == "support":
                            method = f"STC-support-teacher-distill-{quantization}"
                        else:
                            method = f"STC-support-ensemble-teacher-distill-{quantization}"
                        if teacher is None:
                            failed = teacher_unavailable_result(dataset=dataset, student_method=method, teacher_method="", quantization_mode=quantization, feature_cache_size_ratio=float(feature_ratio), path_channel_count_ratio=float(feature_ratio))
                            failed_row = {"stage": "Gate19.2", "dataset": dataset, "seed": int(seed), "method": method, "method_family": "stc_support_teacher_distill", "requested_budget": float(feature_ratio), "status": "failed", "diagnostic_only": False, "eligible_for_main_decision": True, "primary_eval_mode": "compressed_projected", "no_test_leakage": True, **failed}
                            rows.append(failed_row)
                            distill_rows.append(failed)
                            continue
                        student_row, distill_diag, pc, cm = _distill_student(
                            dataset=dataset,
                            seed=int(seed),
                            student_method=method,
                            teacher_method=teacher["method"],
                            teacher_task=teacher["task"],
                            teacher_row=teacher["row"],
                            full_cache=full_cache,
                            labels=labels,
                            split=split,
                            feature_ratio=float(feature_ratio),
                            quantization_mode=quantization,
                            full_context=full_context,
                            args=args,
                        )
                        rows.append(student_row)
                        distill_rows.append(distill_diag)
                        per_class_rows.extend(pc)
                        confusion_rows.extend(cm)
                        cost_rows.append({key: student_row.get(key, "") for key in ("dataset", "seed", "method", "requested_budget", "support_node_ratio", "support_edge_ratio", "unit_count_ratio", "feature_cache_size_ratio", "path_channel_count_ratio", "feature_cache_bytes", "logit_cache_bytes", "model_param_bytes", "calibration_param_bytes", "ensemble_param_bytes", "total_storage_bytes", "total_inference_storage_bytes", "total_storage_ratio_vs_full_stc", "total_storage_ratio_vs_full_graph", "cost_axis_used")})
                        leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_budget": float(feature_ratio), **leakage_audit_row(method=method, uses_train_labels=True, uses_val_labels=True, uses_test_labels_before_final_eval=False, calibration_split="none", path_selection_split="train_val", teacher_training_split="train", student_training_split="train")})

        write_csv(output_dir / "gate19_2_raw_rows.csv", rows)

    write_csv(output_dir / "gate19_2_raw_rows.csv", rows)
    write_csv(output_dir / "gate19_2_calibrated_support_baselines.csv", calibrated_support_rows)
    write_csv(output_dir / "gate19_2_logit_ensemble.csv", ensemble_rows)
    write_csv(output_dir / "gate19_2_ensemble_weights.csv", ensemble_weight_rows)
    write_csv(output_dir / "gate19_2_support_teacher_distill.csv", distill_rows)
    write_csv(output_dir / "gate19_2_cost_breakdown.csv", cost_rows)
    write_csv(output_dir / "gate19_2_nested_calibration.csv", nested_rows)
    write_csv(output_dir / "gate19_2_per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "gate19_2_confusion_matrix_by_method.csv", confusion_rows)
    write_csv(output_dir / "gate19_2_leakage_audit.csv", leakage_rows)
    write_csv(output_dir / "gate19_2_cache_size_audit.csv", cache_size_rows)
    write_csv(output_dir / "diagnostics" / "gate19_2_calibration.csv", calibration_rows)
    write_csv(output_dir / "gate19_2_calibration.csv", calibration_rows)
    _write_calibration_shift_report(output_dir, rows, per_class_rows)
    result = summarize(output_dir, output_dir)
    _write_requirement_checklist(output_dir, result)
    _write_zip(
        output_dir,
        "gate19_2_main_results.zip",
        [
            "gate19_2_raw_rows.csv",
            "gate19_2_validation_selected_by_method.csv",
            "gate19_2_by_dataset_selected.csv",
            "gate19_2_pareto_frontier.csv",
            "gate19_2_result.json",
            "gate19_2_decision.md",
            "gate19_2_requirement_checklist.md",
            "method_to_code_path.csv",
            "preflight/code_sync_report.md",
        ],
    )
    _write_zip(
        output_dir,
        "gate19_2_core_diagnostics.zip",
        [
            "gate19_2_calibrated_support_baselines.csv",
            "gate19_2_logit_ensemble.csv",
            "gate19_2_ensemble_weights.csv",
            "gate19_2_support_teacher_distill.csv",
            "gate19_2_cost_breakdown.csv",
            "gate19_2_nested_calibration.csv",
            "gate19_2_per_class_metrics.csv",
            "gate19_2_confusion_matrix_by_method.csv",
            "gate19_2_leakage_audit.csv",
            "gate19_2_cache_size_audit.csv",
            "diagnostics/gate19_2_calibration.csv",
            "diagnostics/gate19_2_calibration_shift_report.md",
        ],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate19.2 HeSF-CAL support teacher diagnostic.")
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--support-ratios", nargs="*", default=[0.30, 0.50, 0.70, 1.00])
    parser.add_argument("--feature-cache-ratios", nargs="*", default=[0.30, 0.50, 0.70, 1.00])
    parser.add_argument("--nested-calibration-repeats", type=int, default=5)
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate19_2"))
    parser.add_argument("--gate19-input-dir", type=Path, default=Path("outputs/gate19"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
    parser.add_argument("--distill-epochs", type=int, default=1)
    parser.add_argument("--distill-hidden-dim", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--ensemble-grid-step", type=float, default=0.1)
    parser.add_argument("--run-support-teacher-distill", nargs="?", const=True, default=True, type=_bool_arg)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    if args.datasets:
        allowed = {str(dataset).upper() for dataset in args.datasets}
        args.dataset_seed_pairs = [(dataset, seed) for dataset, seed in args.dataset_seed_pairs if str(dataset).upper() in allowed]
    args.support_ratios_parsed = _split_values(args.support_ratios, float) or [0.30, 0.50, 0.70, 1.00]
    args.feature_cache_ratios_parsed = _split_values(args.feature_cache_ratios, float) or [0.30, 0.50, 0.70, 1.00]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
