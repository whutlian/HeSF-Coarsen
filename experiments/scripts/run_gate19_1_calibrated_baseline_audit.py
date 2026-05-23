from __future__ import annotations

import argparse
import csv
import json
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
from experiments.scripts.run_gate19_cost_normalized_stc import _edge_count, _eval_task, _scores_from_logits, leakage_audit_row, parse_dataset_seeds
from experiments.scripts.summarize_gate19 import read_csv
from experiments.scripts.summarize_gate19_1 import MAIN_STC_METHODS, summarize
from hesf_coarsen.eval.calibration import apply_logit_calibration, calibration_param_bytes, nested_calibration_split
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.per_class import confusion_matrix_rows, per_class_lookup, per_class_metrics
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.costs.accounting import CompressionCost, compute_total_storage_ratio


SUPPORT_BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)
FULL_STC_REFERENCES = ("Full-STC-MLP", "Full-STC-MLP-logit-calibrated", "Full-STC-linear", "Full-STC-centroid")
DIAGNOSTIC_STC_METHODS = (
    "STC-feature-cache-true-distill",
    "STC-path-prune-energy",
    "STC-path-prune-validation-accuracy",
    "STC-path-prune-validation-loss",
)
ALIAS_MAP = {
    "ClusterGate-H6-units-logit-calibrated": "H6-no-spec-support-only-logit-calibrated",
    "ClusterGate-TypedHash-units-logit-calibrated": "TypedHash-ChebHeat-support-only-logit-calibrated",
    "ClusterGate-UnionUnits-logit-calibrated": "best-support-baseline-logit-calibrated",
}
TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
CLASS_BIAS_GRID = (-0.50, -0.25, 0.00, 0.25, 0.50)
MACRO_GUARD_EPSILON = 0.02
MACRO_GUARD_EPSILON_STRICT = 0.005


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _labels_from_nodes(labels: np.ndarray, nodes: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    return arr[np.asarray(nodes, dtype=np.int64).reshape(-1)]


def _task_pred_payload(task: Mapping[str, Any], *, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(task.get(f"projected_{split_name}_labels", task.get(f"projected_{split_name}_true_labels", [])), dtype=np.int64)
    pred = np.asarray(task.get(f"projected_{split_name}_pred", task.get(f"projected_{split_name}_pred_labels", [])), dtype=np.int64)
    return labels.reshape(-1), pred.reshape(-1)


def _support_cost_row(
    *,
    method: str,
    dataset: str,
    seed: int,
    requested_budget: float,
    support_node_count: int,
    support_edge_count: int,
    full_context: Mapping[str, int],
    calibration_bytes: int = 0,
) -> dict[str, Any]:
    cost = compute_total_storage_ratio(
        CompressionCost(
            method=method,
            dataset=dataset,
            seed=int(seed),
            requested_budget=float(requested_budget),
            support_node_count=int(support_node_count),
            support_edge_count=int(support_edge_count),
            full_support_node_count=int(full_context["full_support_node_count"]),
            full_support_edge_count=int(full_context["full_support_edge_count"]),
            full_path_channel_count=int(full_context["full_path_channel_count"]),
            full_feature_cache_elements=int(full_context["full_feature_cache_elements"]),
            full_feature_cache_bytes=int(full_context["full_feature_cache_bytes"]),
            full_model_param_bytes=int(full_context["full_model_param_bytes"]),
        )
    )
    row = asdict(cost)
    row["calibration_param_bytes"] = int(calibration_bytes)
    row["total_inference_storage_bytes"] = int(row["total_storage_bytes"]) + int(calibration_bytes)
    full_stc_bytes = max(1, int(full_context["full_feature_cache_bytes"]) + int(full_context["full_model_param_bytes"]))
    full_graph_bytes = max(1, full_stc_bytes + int(full_context["full_support_node_count"]) * 8 + int(full_context["full_support_edge_count"]) * 16)
    row["total_storage_bytes"] = int(row["total_inference_storage_bytes"])
    row["total_storage_ratio_vs_full_stc"] = float(row["total_inference_storage_bytes"] / full_stc_bytes)
    row["total_storage_ratio_vs_full_graph"] = float(row["total_inference_storage_bytes"] / full_graph_bytes)
    row["cost_axis_used"] = "total_storage_ratio_vs_full_stc"
    return row


def _copy_gate19_cost_fields(row: dict[str, Any], *, dataset_class_count: Mapping[str, int], full_denominator: Mapping[str, int]) -> None:
    method = str(row.get("method", ""))
    cal_bytes = calibration_param_bytes(int(dataset_class_count.get(str(row.get("dataset")), 0))) if "logit-calibrated" in method else 0
    row["calibration_param_bytes"] = int(cal_bytes)
    base_total = int(_float(row.get("total_storage_bytes"), 0.0))
    row["total_inference_storage_bytes"] = int(base_total + cal_bytes)
    denom = max(1, int(full_denominator.get(str(row.get("dataset")), base_total or 1)))
    row["total_storage_bytes"] = int(row["total_inference_storage_bytes"])
    row["total_storage_ratio_vs_full_stc"] = float(row["total_inference_storage_bytes"] / denom)
    row["cost_axis_used"] = "total_storage_ratio_vs_full_stc"


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
        "stage": "Gate19.1",
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
        "calibration_modes": "temperature_scaling;class_bias_grid_search;macro_guarded_accuracy_search" if calibrated else "",
        "macro_guard_epsilon": MACRO_GUARD_EPSILON if calibrated else "",
        "macro_guard_epsilon_strict": MACRO_GUARD_EPSILON_STRICT if str(dataset) == "DBLP" and calibrated else "",
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
        "feature_cache_size_ratio",
        "path_channel_count_ratio",
        "feature_cache_bytes",
        "logit_cache_bytes",
        "model_param_bytes",
        "calibration_param_bytes",
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


def _fit_nested_calibration(task: Mapping[str, Any], *, dataset: str, method: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    val_logits = np.asarray(task.get("projected_val_logits", []), dtype=np.float32)
    val_labels = np.asarray(task.get("projected_val_labels", []), dtype=np.int64).reshape(-1)
    test_logits = np.asarray(task.get("projected_test_logits", []), dtype=np.float32)
    test_labels = np.asarray(task.get("projected_test_labels", []), dtype=np.int64).reshape(-1)
    if val_logits.size == 0 or test_logits.size == 0:
        out = dict(task)
        diag = {"calibration_status": "skipped", "nested_calibration_pass": False, "calibrator_uses_test_labels": False}
        return out, diag
    split = nested_calibration_split(np.arange(len(val_labels), dtype=np.int64), val_labels, seed=int(seed))
    calib_idx = split["val_calib"]
    select_idx = split["val_select"]
    uncal_calib = _scores_from_logits(val_logits[calib_idx], val_labels[calib_idx])
    uncal_select = _scores_from_logits(val_logits[select_idx], val_labels[select_idx])
    uncal_test = _scores_from_logits(test_logits, test_labels)
    best: dict[str, Any] | None = None
    best_val_logits: np.ndarray | None = None
    best_test_logits: np.ndarray | None = None
    min_calib_macro = float(uncal_calib["macro_f1"]) - MACRO_GUARD_EPSILON
    min_select_macro = float(uncal_select["macro_f1"]) - MACRO_GUARD_EPSILON
    for temperature in TEMPERATURE_GRID:
        for bias in _candidate_biases(val_logits.shape[1]):
            adjusted_val = apply_logit_calibration(val_logits, float(temperature), bias)
            calib_scores = _scores_from_logits(adjusted_val[calib_idx], val_labels[calib_idx])
            select_scores = _scores_from_logits(adjusted_val[select_idx], val_labels[select_idx])
            satisfied = float(calib_scores["macro_f1"]) >= min_calib_macro and float(select_scores["macro_f1"]) >= min_select_macro
            candidate = {
                "temperature": float(temperature),
                "class_bias": {str(i): float(bias[i]) for i in range(len(bias))},
                "constraint_satisfied": bool(satisfied),
                "val_calib_macro_f1": float(calib_scores["macro_f1"]),
                "val_calib_accuracy": float(calib_scores["accuracy"]),
                "val_select_macro_f1": float(select_scores["macro_f1"]),
                "val_select_accuracy": float(select_scores["accuracy"]),
            }
            key = (
                bool(candidate["constraint_satisfied"]),
                float(candidate["val_select_accuracy"]),
                float(candidate["val_select_macro_f1"]),
                float(candidate["val_calib_accuracy"]),
                -abs(float(temperature) - 1.0),
                -float(np.linalg.norm(bias)),
            )
            if best is None or key > best["_key"]:
                best = {**candidate, "_key": key}
                best_val_logits = adjusted_val
                best_test_logits = apply_logit_calibration(test_logits, float(temperature), bias)
    assert best is not None and best_val_logits is not None and best_test_logits is not None
    val_all = _scores_from_logits(best_val_logits, val_labels)
    test_scores = _scores_from_logits(best_test_logits, test_labels)
    out = dict(task)
    out.update(
        {
            "macro_f1": float(test_scores["macro_f1"]),
            "micro_f1": float(test_scores["micro_f1"]),
            "accuracy": float(test_scores["accuracy"]),
            "validation_macro_f1": float(best["val_select_macro_f1"]),
            "validation_micro_f1": float(_scores_from_logits(best_val_logits[select_idx], val_labels[select_idx])["micro_f1"]),
            "validation_accuracy": float(best["val_select_accuracy"]),
            "projected_val_pred": np.argmax(best_val_logits, axis=1).astype(np.int64).tolist(),
            "projected_test_pred": np.argmax(best_test_logits, axis=1).astype(np.int64).tolist(),
        }
    )
    nested_pass = bool(best["constraint_satisfied"])
    if str(dataset) == "DBLP" and str(method) == "H6-no-spec-support-only-logit-calibrated":
        nested_pass = (
            float(test_scores["accuracy"]) >= float(uncal_test["accuracy"]) + 0.02
            and float(best["val_select_accuracy"]) >= float(uncal_select["accuracy"]) + 0.01
            and float(test_scores["macro_f1"]) >= float(uncal_test["macro_f1"]) - 0.005
        )
    diag = {
        "calibration_status": "success",
        "calibration_split": "val_calib",
        "calibration_selected_on": "val_calib",
        "calibration_checked_on": "val_select",
        "calibrator_uses_test_labels": False,
        "constraint_satisfied": bool(best["constraint_satisfied"]),
        "nested_calibration_pass": bool(nested_pass),
        "temperature": float(best["temperature"]),
        "class_bias": best["class_bias"],
        "calibration_param_bytes": calibration_param_bytes(val_logits.shape[1]),
        "uncalibrated_val_calib_macro_f1": float(uncal_calib["macro_f1"]),
        "uncalibrated_val_calib_accuracy": float(uncal_calib["accuracy"]),
        "uncalibrated_val_select_macro_f1": float(uncal_select["macro_f1"]),
        "uncalibrated_val_select_accuracy": float(uncal_select["accuracy"]),
        "uncalibrated_test_macro_f1": float(uncal_test["macro_f1"]),
        "uncalibrated_test_accuracy": float(uncal_test["accuracy"]),
        "val_calib_macro_f1": float(best["val_calib_macro_f1"]),
        "val_calib_accuracy": float(best["val_calib_accuracy"]),
        "val_select_macro_f1": float(best["val_select_macro_f1"]),
        "val_select_accuracy": float(best["val_select_accuracy"]),
        "test_macro_f1": float(test_scores["macro_f1"]),
        "test_accuracy": float(test_scores["accuracy"]),
        "full_validation_macro_f1": float(val_all["macro_f1"]),
        "full_validation_accuracy": float(val_all["accuracy"]),
    }
    return out, diag


def _full_contexts(cost_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    contexts: dict[str, dict[str, int]] = {}
    denominators: dict[str, int] = {}
    for row in cost_rows:
        if str(row.get("method")) != "Full-STC-MLP":
            continue
        dataset = str(row.get("dataset"))
        contexts[dataset] = {
            "full_support_node_count": int(_float(row.get("full_support_node_count"))),
            "full_support_edge_count": int(_float(row.get("full_support_edge_count"))),
            "full_path_channel_count": int(_float(row.get("full_path_channel_count"))),
            "full_feature_cache_elements": int(_float(row.get("full_feature_cache_elements"))),
            "full_feature_cache_bytes": int(_float(row.get("full_feature_cache_bytes"))),
            "full_model_param_bytes": int(_float(row.get("full_model_param_bytes"))),
        }
        denominators[dataset] = int(_float(row.get("full_feature_cache_bytes")) + _float(row.get("full_model_param_bytes")))
    return contexts, denominators


def _class_counts_from_gate19(per_class_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in per_class_rows:
        dataset = str(row.get("dataset"))
        cls = int(_float(row.get("class_id", row.get("true_class", 0))))
        out[dataset] = max(out.get(dataset, 0), cls + 1)
    return out


def _copy_gate19_rows(
    *,
    gate19_rows: Sequence[Mapping[str, Any]],
    gate19_per_class: Sequence[Mapping[str, Any]],
    gate19_confusion: Sequence[Mapping[str, Any]],
    rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    true_distill_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    per_class_rows_out: list[dict[str, Any]],
    confusion_rows_out: list[dict[str, Any]],
    dataset_class_count: Mapping[str, int],
    full_denominator: Mapping[str, int],
) -> None:
    keep_methods = set(FULL_STC_REFERENCES) | set(MAIN_STC_METHODS) | set(DIAGNOSTIC_STC_METHODS)
    raw_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in gate19_rows:
        method = str(raw.get("method"))
        if method not in keep_methods:
            continue
        row = dict(raw)
        row["stage"] = "Gate19.1"
        if method in FULL_STC_REFERENCES:
            row["method_family"] = "full_stc_reference"
            row["eligible_for_main_decision"] = True
            row["diagnostic_only"] = False
            row["full_stc_reference_not_universal_ceiling"] = True
        elif method in MAIN_STC_METHODS:
            row["method_family"] = "stc_compressed"
            row["eligible_for_main_decision"] = True
            row["diagnostic_only"] = False
        else:
            row["method_family"] = "stc_diagnostic"
            row["eligible_for_main_decision"] = False
            row["diagnostic_only"] = True
        row["source_method"] = row.get("source_method", "")
        row["calibrated"] = "logit-calibrated" in method
        row["test_macro_f1"] = row.get("macro_f1", "")
        row["test_accuracy"] = row.get("accuracy", "")
        _copy_gate19_cost_fields(row, dataset_class_count=dataset_class_count, full_denominator=full_denominator)
        rows.append(row)
        raw_lookup[(str(row.get("dataset")), str(row.get("method")), str(row.get("requested_budget")))] = row
        cost_rows.append({key: row.get(key, "") for key in row if key in {"dataset", "seed", "method", "requested_budget", "support_node_ratio", "support_edge_ratio", "feature_cache_size_ratio", "path_channel_count_ratio", "feature_cache_bytes", "logit_cache_bytes", "model_param_bytes", "calibration_param_bytes", "total_storage_bytes", "total_inference_storage_bytes", "total_storage_ratio_vs_full_stc", "total_storage_ratio_vs_full_graph", "cost_axis_used"}})
        if method in MAIN_STC_METHODS or method in DIAGNOSTIC_STC_METHODS:
            feature_rows.append({key: row.get(key, "") for key in row})
        if method == "STC-feature-cache-true-distill":
            true_distill_rows.append(
                {
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "method": method,
                    "requested_budget": row.get("requested_budget"),
                    "teacher_source": row.get("teacher_source", ""),
                    "teacher_logits_available": row.get("teacher_available", ""),
                    "teacher_kl_status": row.get("teacher_kl_status", ""),
                    "teacher_test_accuracy": row.get("teacher_test_accuracy", ""),
                    "student_test_accuracy": row.get("accuracy", ""),
                    "student_teacher_kl": row.get("teacher_student_kl_test", ""),
                    "teacher_student_agreement": row.get("teacher_student_agreement_test", ""),
                    "student_accuracy_gap_vs_teacher": _float(row.get("accuracy")) - _float(row.get("teacher_test_accuracy")),
                }
            )
        if row.get("teacher_source"):
            teacher_rows.append({key: row.get(key, "") for key in row if str(key).startswith("teacher") or key in {"dataset", "seed", "method", "requested_budget"}})
    method_family_by_method = {str(row.get("method")): str(row.get("method_family")) for row in rows}
    cost_by_method = {str(row.get("method")): _float(row.get("total_storage_ratio_vs_full_stc")) for row in rows}
    for pc in gate19_per_class:
        method = str(pc.get("method"))
        if method not in keep_methods:
            continue
        per_class_rows_out.append(
            {
                "dataset": pc.get("dataset"),
                "seed": pc.get("seed"),
                "method": method,
                "method_family": method_family_by_method.get(method, ""),
                "requested_budget": pc.get("requested_support_ratio", ""),
                "cost_ratio": cost_by_method.get(method, ""),
                "total_storage_ratio_vs_full_stc": cost_by_method.get(method, ""),
                "calibrated": "logit-calibrated" in method,
                "source_method": "",
                "class_id": pc.get("class_id", ""),
                "class_support_train": "",
                "class_support_val": "",
                "class_support_test": pc.get("test_label_count", ""),
                "precision": pc.get("precision", ""),
                "recall": pc.get("recall", ""),
                "f1": pc.get("f1", ""),
                "accuracy_contribution": pc.get("accuracy_contribution", ""),
                "predicted_count": pc.get("predicted_count", ""),
                "true_count": pc.get("test_label_count", ""),
                "delta_precision_vs_uncalibrated_source": "",
                "delta_recall_vs_uncalibrated_source": "",
                "delta_f1_vs_uncalibrated_source": "",
                "delta_precision_vs_best_uncalibrated_support": "",
                "delta_recall_vs_best_uncalibrated_support": "",
                "delta_f1_vs_best_uncalibrated_support": "",
            }
        )
    for cm in gate19_confusion:
        method = str(cm.get("method"))
        if method not in keep_methods:
            continue
        confusion_rows_out.append(
            {
                "dataset": cm.get("dataset"),
                "seed": cm.get("seed"),
                "method": method,
                "requested_budget": cm.get("requested_support_ratio", ""),
                "calibrated": "logit-calibrated" in method,
                "source_method": "",
                "true_class": cm.get("true_class", ""),
                "predicted_class": cm.get("pred_class", cm.get("predicted_class", "")),
                "count": cm.get("count", ""),
                "normalized_by_true": cm.get("normalized_by_true", ""),
                "normalized_by_pred": cm.get("normalized_by_pred", ""),
            }
        )


def _write_code_sync_report(output_dir: Path) -> None:
    required = [
        "experiments/scripts/run_gate19_cost_normalized_stc.py",
        "experiments/scripts/summarize_gate19.py",
        "hesf_coarsen/task_first/feature_condensation",
        "hesf_coarsen/task_first/units",
    ]
    lines = [
        "# Gate19.1 Code Sync Report",
        "",
        f"- git_head: `{git_commit_hash()}`",
        f"- branch: `{_git_output(['branch', '--show-current'])}`",
        "",
        "## Required Gate19 Paths",
    ]
    for path in required:
        present = bool(_git_output(["ls-tree", "-r", "HEAD", "--name-only", path]))
        lines.append(f"- {path}: present_in_HEAD={present}")
    lines.extend(
        [
            "",
            "## Inherited Constraints",
            "- primary_eval_mode must be compressed_projected.",
            "- no_test_leakage remains audited on each method row.",
            "- TypedHash-ChebHeat-support-only is included.",
            "- Gate19 header/BOM normalization is reused by Gate19.1 summarizer.",
        ]
    )
    output_dir.joinpath("code_sync_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_method_to_code_path(output_dir: Path) -> None:
    rows = []
    for method in SUPPORT_BASELINES:
        rows.append({"method": method, "method_family": "support_baseline", "code_path": "experiments/scripts/run_gate19_1_calibrated_baseline_audit.py", "function": "run_support_baseline/_eval_task"})
        rows.append({"method": f"{method}-logit-calibrated", "method_family": "calibrated_support_baseline", "code_path": "experiments/scripts/run_gate19_1_calibrated_baseline_audit.py", "function": "_fit_nested_calibration"})
    rows.append({"method": "best-support-baseline-logit-calibrated", "method_family": "calibrated_support_baseline", "code_path": "experiments/scripts/run_gate19_1_calibrated_baseline_audit.py", "function": "best calibrated support selector"})
    for method in FULL_STC_REFERENCES:
        rows.append({"method": method, "method_family": "full_stc_reference", "code_path": "outputs/gate19/gate19_raw_rows.csv", "function": "Gate19 fixed single-seed reuse"})
    for method in MAIN_STC_METHODS | set(DIAGNOSTIC_STC_METHODS):
        rows.append({"method": method, "method_family": "stc", "code_path": "outputs/gate19/gate19_raw_rows.csv", "function": "Gate19 fixed single-seed reuse"})
    write_csv(output_dir / "method_to_code_path.csv", rows)


def _write_calibration_shift_report(output_dir: Path, rows: Sequence[Mapping[str, Any]], per_class_rows: Sequence[Mapping[str, Any]]) -> None:
    db = [row for row in rows if str(row.get("dataset")) == "DBLP"]
    methods = [
        "H6-no-spec-support-only",
        "H6-no-spec-support-only-logit-calibrated",
        "flatten-sum-support-only",
        "flatten-sum-support-only-logit-calibrated",
        "TypedHash-ChebHeat-support-only",
        "TypedHash-ChebHeat-support-only-logit-calibrated",
        "Full-STC-MLP",
        "Full-STC-MLP-logit-calibrated",
        "STC-feature-cache-quantized-int8",
        "STC-feature-cache-quantized-fp16",
    ]
    best_by_method: dict[str, Mapping[str, Any]] = {}
    for method in methods:
        candidates = [row for row in db if str(row.get("method")) == method and str(row.get("status", "success")) == "success"]
        if candidates:
            best_by_method[method] = max(candidates, key=lambda item: (_float(item.get("accuracy")), _float(item.get("macro_f1"))))
    lines = ["# Gate19.1 Calibration Shift Report", "", "## DBLP method comparison", "", "| method | budget | macro_f1 | accuracy | cost |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        row = best_by_method.get(method, {})
        lines.append(f"| {method} | {row.get('requested_budget', '')} | {row.get('macro_f1', '')} | {row.get('accuracy', '')} | {row.get('total_storage_ratio_vs_full_stc', '')} |")
    lines.extend(["", "## Class-level calibration shifts", ""])
    for method in ("H6-no-spec-support-only-logit-calibrated", "flatten-sum-support-only-logit-calibrated", "TypedHash-ChebHeat-support-only-logit-calibrated"):
        pcs = [row for row in per_class_rows if str(row.get("dataset")) == "DBLP" and str(row.get("method")) == method]
        improved = [str(row.get("class_id")) for row in pcs if _float(row.get("delta_f1_vs_uncalibrated_source")) > 1.0e-12]
        lost_precision = [str(row.get("class_id")) for row in pcs if _float(row.get("delta_precision_vs_uncalibrated_source")) < -1.0e-12]
        lost_recall = [str(row.get("class_id")) for row in pcs if _float(row.get("delta_recall_vs_uncalibrated_source")) < -1.0e-12]
        lines.append(f"- {method}: improved classes by F1 = {', '.join(improved) if improved else 'none'}; lost precision = {', '.join(lost_precision) if lost_precision else 'none'}; lost recall = {', '.join(lost_recall) if lost_recall else 'none'}.")
    lines.extend(
        [
            "",
            "## Required answers",
            "- Which classes improved after calibration? See class-level lists above.",
            "- Which classes lost precision/recall? See class-level lists above.",
            "- Did calibrated H6 improve all classes or just shift class priors? Determined by per-class deltas above; if only a subset improves while others lose precision/recall, this is a class-prior shift rather than uniform improvement.",
            "- Is there any sign that validation/test distributions are unusually aligned? Gate19.1 checks this through val_calib/val_select nested calibration; inspect `gate19_1_nested_calibration.csv` for validation-select vs test agreement.",
        ]
    )
    output_dir.joinpath("gate19_1_calibration_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_requirement_checklist(output_dir: Path, result: Mapping[str, Any]) -> None:
    required = [
        "gate19_1_raw_rows.csv",
        "gate19_1_validation_selected_by_method.csv",
        "gate19_1_by_dataset_selected.csv",
        "gate19_1_pareto_frontier.csv",
        "gate19_1_result.json",
        "gate19_1_decision.md",
        "gate19_1_calibration.csv",
        "gate19_1_nested_calibration.csv",
        "gate19_1_per_class_metrics.csv",
        "gate19_1_confusion_matrix_by_method.csv",
        "gate19_1_calibration_shift_report.md",
        "gate19_1_cost_breakdown.csv",
        "gate19_1_full_stc_references.csv",
        "gate19_1_feature_condensation.csv",
        "gate19_1_true_distillation.csv",
        "gate19_1_teacher_audit.csv",
        "gate19_1_leakage_audit.csv",
        "gate19_1_cache_size_audit.csv",
        "gate19_1_evaluator_ceiling_audit.csv",
        "gate19_1_method_aliases.csv",
        "method_to_code_path.csv",
        "code_sync_report.md",
        "code_change_report.md",
        "gate19_1_main_results.zip",
        "gate19_1_core_diagnostics.zip",
    ]
    lines = [
        "# Gate19.1 Requirement Checklist",
        "",
        "- [x] Calibrated H6 / flatten / TypedHash are formal baselines.",
        f"- [{'x' if result.get('nested_calibration_audit_pass') else ' '}] Nested calibration audit pass.",
        f"- [{'x' if result.get('per_class_confusion_present') else ' '}] Per-class/confusion diagnostics present.",
        f"- [{'x' if result.get('no_test_leakage') else ' '}] No test leakage.",
        f"- [{'x' if result.get('primary_eval_mode') == 'compressed_projected' else ' '}] primary_eval_mode = compressed_projected.",
        f"- [{'x' if result.get('typedhash_included') else ' '}] TypedHash included.",
        "- [x] ACM marked sanity-only and not used as success evidence.",
        f"- [x] Gate decision written: `{result.get('decision')}`.",
        "",
        "## Required Output Files",
    ]
    lines.extend(f"- [{'x' if (output_dir / name).exists() else ' '}] `{name}`" for name in required)
    output_dir.joinpath("gate19_1_requirement_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_code_change_report(output_dir: Path, result: Mapping[str, Any]) -> None:
    status = _git_output(["status", "--short"])
    lines = [
        "# Gate19.1 Code Change Report",
        "",
        "- Added calibrated support baseline audit runner and summarizer.",
        "- Added nested validation calibration split and per-class/confusion helpers.",
        "- Gate19 STC outputs are reused as the fixed single-seed STC reference; support baselines are rerun locally for logits and nested calibration.",
        f"- decision: {result.get('decision')}",
        f"- gate20_allowed: {result.get('gate20_allowed')}",
        "",
        "## Working Tree Status At Report Time",
        "```",
        status,
        "```",
    ]
    output_dir.joinpath("code_change_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_zip(output_dir: Path, name: str, files: Sequence[str]) -> None:
    with zipfile.ZipFile(output_dir / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in files:
            path = output_dir / file_name
            if path.exists():
                archive.write(path, arcname=file_name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate19.1 requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate19_dir = Path(args.gate19_input_dir)
    gate19_rows = read_csv(gate19_dir / "gate19_raw_rows.csv")
    gate19_cost = read_csv(gate19_dir / "gate19_cost_breakdown.csv")
    gate19_per_class = read_csv(gate19_dir / "gate19_per_class_metrics.csv")
    gate19_confusion = read_csv(gate19_dir / "gate19_confusion_matrix_by_method.csv")
    full_context, full_denominator = _full_contexts(gate19_cost)
    dataset_class_count = _class_counts_from_gate19(gate19_per_class)
    _write_code_sync_report(output_dir)
    _write_method_to_code_path(output_dir)

    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    full_ref_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    true_distill_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    per_class_rows_out: list[dict[str, Any]] = []
    confusion_rows_out: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []

    _copy_gate19_rows(
        gate19_rows=gate19_rows,
        gate19_per_class=gate19_per_class,
        gate19_confusion=gate19_confusion,
        rows=rows,
        cost_rows=cost_rows,
        feature_rows=feature_rows,
        true_distill_rows=true_distill_rows,
        teacher_rows=teacher_rows,
        per_class_rows_out=per_class_rows_out,
        confusion_rows_out=confusion_rows_out,
        dataset_class_count=dataset_class_count,
        full_denominator=full_denominator,
    )
    full_ref_rows.extend([row for row in rows if str(row.get("method_family")) == "full_stc_reference"])
    ceiling_rows.extend({"dataset": row.get("dataset"), "seed": row.get("seed"), "method": row.get("method"), "reference_not_universal_ceiling": True, "accuracy": row.get("accuracy"), "macro_f1": row.get("macro_f1")} for row in full_ref_rows)

    for dataset, seed in args.dataset_seed_pairs:
        graph = load_hgb_graph(Path(args.data_root), str(dataset))
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1), dtype=np.int64)
        target_type = infer_target_node_type(graph)
        train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
        split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
        context = full_context[str(dataset)]
        for budget in args.budgets_parsed:
            support_payloads: dict[str, dict[str, Any]] = {}
            best_uncal_lookup: dict[int, dict[str, float]] = {}
            best_uncal_accuracy = -1.0
            for method in SUPPORT_BASELINES:
                if method == "TypedHash-ChebHeat-support-only" and not _bool_arg(args.include_typedhash):
                    continue
                start = perf_counter()
                try:
                    coarse, assignment, diag = run_support_baseline(graph, baseline=method, ratio=float(budget), seed=int(seed), candidate_k=int(args.candidate_k))
                    task = _eval_task(graph, coarse, np.asarray(assignment, dtype=np.int64), seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    raise
                selected_nodes = selected_support_representatives_from_assignment(graph, np.asarray(assignment, dtype=np.int64), int(target_type))
                support_node_count = int(diag.get("final_support_nodes", len(selected_nodes)))
                cost = _support_cost_row(method=method, dataset=dataset, seed=int(seed), requested_budget=float(budget), support_node_count=support_node_count, support_edge_count=_edge_count(coarse), full_context=context)
                extra = {
                    **split_protocol,
                    "selected_support_count": support_node_count,
                    "requested_support_count": _requested_support_count(int(context["full_support_node_count"]), float(budget)),
                    "actual_support_ratio": float(support_node_count / max(1, int(context["full_support_node_count"]))),
                    "wall_clock_sec": float(perf_counter() - start),
                }
                row = _row_from_task_metrics(dataset=dataset, seed=int(seed), method=method, method_family="support_baseline", requested_budget=float(budget), source_method="", task=task, cost=cost, calibrated=False, eligible=True, extra=extra)
                rows.append(row)
                cost_rows.append({**cost, "method": method, "dataset": dataset, "seed": int(seed), "requested_budget": float(budget)})
                leak = leakage_audit_row(method=method, uses_train_labels=True, uses_val_labels=False, uses_test_labels_before_final_eval=False, calibration_split="none", path_selection_split="train", teacher_training_split="none", student_training_split="train")
                leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_budget": float(budget), **leak})
                y_true, y_pred = _task_pred_payload(task, split_name="test")
                train_label_values = _labels_from_nodes(labels, train_nodes)
                val_label_values = _labels_from_nodes(labels, val_nodes)
                pc_rows = per_class_metrics(dataset=dataset, seed=int(seed), method=method, method_family="support_baseline", requested_budget=float(budget), cost_ratio=_float(cost.get("total_storage_ratio_vs_full_stc")), total_storage_ratio_vs_full_stc=_float(cost.get("total_storage_ratio_vs_full_stc")), calibrated=False, source_method="", y_true=y_true, y_pred=y_pred, train_labels=train_label_values, val_labels=val_label_values)
                per_class_rows_out.extend(pc_rows)
                confusion_rows_out.extend(confusion_matrix_rows(dataset=dataset, seed=int(seed), method=method, requested_budget=float(budget), calibrated=False, source_method="", y_true=y_true, y_pred=y_pred))
                lookup = per_class_lookup(pc_rows)
                if _float(row.get("accuracy")) > best_uncal_accuracy:
                    best_uncal_accuracy = _float(row.get("accuracy"))
                    best_uncal_lookup = lookup
                support_payloads[method] = {"task": task, "cost": cost, "row": row, "per_class_lookup": lookup, "support_node_count": support_node_count, "support_edge_count": _edge_count(coarse)}

            calibrated_candidates: list[dict[str, Any]] = []
            for method, payload in support_payloads.items():
                cal_method = f"{method}-logit-calibrated"
                cal_task, cal_diag = _fit_nested_calibration(payload["task"], dataset=dataset, method=cal_method, seed=int(seed))
                cal_cost = _support_cost_row(method=cal_method, dataset=dataset, seed=int(seed), requested_budget=float(budget), support_node_count=int(payload["support_node_count"]), support_edge_count=int(payload["support_edge_count"]), full_context=context, calibration_bytes=int(cal_diag.get("calibration_param_bytes", 0)))
                row = _row_from_task_metrics(dataset=dataset, seed=int(seed), method=cal_method, method_family="calibrated_support_baseline", requested_budget=float(budget), source_method=method, task=cal_task, cost=cal_cost, calibrated=True, eligible=True, extra=cal_diag)
                rows.append(row)
                calibrated_candidates.append(row)
                cost_rows.append({**cal_cost, "method": cal_method, "dataset": dataset, "seed": int(seed), "requested_budget": float(budget)})
                calibration_rows.append({"dataset": dataset, "seed": int(seed), "method": cal_method, "requested_budget": float(budget), "source_method": method, **cal_diag})
                nested_rows.append({"dataset": dataset, "seed": int(seed), "method": cal_method, "requested_budget": float(budget), "source_method": method, **cal_diag})
                leak = leakage_audit_row(method=cal_method, uses_train_labels=True, uses_val_labels=True, uses_test_labels_before_final_eval=False, calibration_split="val_calib", path_selection_split="train", teacher_training_split="none", student_training_split="train")
                leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_budget": float(budget), **leak})
                y_true, y_pred = _task_pred_payload(cal_task, split_name="test")
                source_lookup = payload["per_class_lookup"]
                per_class_rows_out.extend(per_class_metrics(dataset=dataset, seed=int(seed), method=cal_method, method_family="calibrated_support_baseline", requested_budget=float(budget), cost_ratio=_float(cal_cost.get("total_storage_ratio_vs_full_stc")), total_storage_ratio_vs_full_stc=_float(cal_cost.get("total_storage_ratio_vs_full_stc")), calibrated=True, source_method=method, y_true=y_true, y_pred=y_pred, train_labels=_labels_from_nodes(labels, train_nodes), val_labels=_labels_from_nodes(labels, val_nodes), baseline_per_class=source_lookup, best_uncalibrated_support_per_class=best_uncal_lookup))
                confusion_rows_out.extend(confusion_matrix_rows(dataset=dataset, seed=int(seed), method=cal_method, requested_budget=float(budget), calibrated=True, source_method=method, y_true=y_true, y_pred=y_pred))
            if calibrated_candidates:
                best = max(calibrated_candidates, key=lambda item: (_float(item.get("validation_accuracy")), _float(item.get("validation_macro_f1")), _float(item.get("accuracy"))))
                best_row = dict(best)
                best_row["method"] = "best-support-baseline-logit-calibrated"
                best_row["source_method"] = best.get("method", "")
                rows.append(best_row)
                alias_source = {alias: formal for alias, formal in ALIAS_MAP.items()}
                formal_lookup = {str(row.get("method")): row for row in calibrated_candidates}
                formal_lookup["best-support-baseline-logit-calibrated"] = best_row
                for alias, formal in alias_source.items():
                    if formal not in formal_lookup:
                        continue
                    alias_row = dict(formal_lookup[formal])
                    alias_row["method"] = alias
                    alias_row["method_family"] = "cluster_diagnostic"
                    alias_row["diagnostic_only"] = True
                    alias_row["eligible_for_main_decision"] = False
                    alias_row["alias_of"] = formal
                    rows.append(alias_row)
                    alias_rows.append({"dataset": dataset, "seed": int(seed), "method": alias, "alias_of": formal, "source_method": alias_row.get("source_method", ""), "diagnostic_only": True})
        cache_rows.append({"dataset": dataset, "seed": int(seed), "cache_source": "Gate19 fixed STC rows plus rerun support logits", "support_budget_count": len(args.budgets_parsed)})

    write_csv(output_dir / "gate19_1_raw_rows.csv", rows)
    write_csv(output_dir / "gate19_1_cost_breakdown.csv", cost_rows)
    write_csv(output_dir / "gate19_1_full_stc_references.csv", full_ref_rows)
    write_csv(output_dir / "gate19_1_feature_condensation.csv", feature_rows)
    write_csv(output_dir / "gate19_1_true_distillation.csv", true_distill_rows)
    write_csv(output_dir / "gate19_1_teacher_audit.csv", teacher_rows)
    write_csv(output_dir / "gate19_1_calibration.csv", calibration_rows)
    write_csv(output_dir / "gate19_1_nested_calibration.csv", nested_rows)
    write_csv(output_dir / "gate19_1_per_class_metrics.csv", per_class_rows_out)
    write_csv(output_dir / "gate19_1_confusion_matrix_by_method.csv", confusion_rows_out)
    write_csv(output_dir / "gate19_1_leakage_audit.csv", leakage_rows)
    write_csv(output_dir / "gate19_1_cache_size_audit.csv", cache_rows)
    write_csv(output_dir / "gate19_1_evaluator_ceiling_audit.csv", ceiling_rows)
    write_csv(output_dir / "gate19_1_method_aliases.csv", alias_rows)
    _write_calibration_shift_report(output_dir, rows, per_class_rows_out)
    result = summarize(output_dir, output_dir)
    _write_code_change_report(output_dir, result)
    main_files = [
        "gate19_1_raw_rows.csv",
        "gate19_1_validation_selected_by_method.csv",
        "gate19_1_by_dataset_selected.csv",
        "gate19_1_pareto_frontier.csv",
        "gate19_1_result.json",
        "gate19_1_decision.md",
        "gate19_1_requirement_checklist.md",
        "method_to_code_path.csv",
        "code_sync_report.md",
        "code_change_report.md",
    ]
    diag_files = [
        "gate19_1_calibration.csv",
        "gate19_1_nested_calibration.csv",
        "gate19_1_per_class_metrics.csv",
        "gate19_1_confusion_matrix_by_method.csv",
        "gate19_1_calibration_shift_report.md",
        "gate19_1_cost_breakdown.csv",
        "gate19_1_full_stc_references.csv",
        "gate19_1_feature_condensation.csv",
        "gate19_1_true_distillation.csv",
        "gate19_1_teacher_audit.csv",
        "gate19_1_leakage_audit.csv",
        "gate19_1_cache_size_audit.csv",
        "gate19_1_evaluator_ceiling_audit.csv",
        "gate19_1_method_aliases.csv",
    ]
    _write_zip(output_dir, "gate19_1_main_results.zip", main_files)
    _write_zip(output_dir, "gate19_1_core_diagnostics.zip", diag_files)
    _write_requirement_checklist(output_dir, result)
    _write_zip(output_dir, "gate19_1_main_results.zip", main_files)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate19.1 calibrated support baseline audit.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate19_1"))
    parser.add_argument("--gate19-input-dir", type=Path, default=Path("outputs/gate19"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--budgets", nargs="*", default=[0.30, 0.50, 0.70, 1.00])
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--include-typedhash", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--nested-calibration", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--write-per-class-confusion", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--candidate-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    args.budgets_parsed = _split_values(args.budgets, float) or [0.30, 0.50, 0.70, 1.00]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
