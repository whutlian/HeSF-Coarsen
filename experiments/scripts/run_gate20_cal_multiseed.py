from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.run_gate17_1_support_sensitivity import _full_graph_row, _target_only_row
from experiments.scripts.gate17_4_h6 import selected_support_representatives_from_assignment
from experiments.scripts.run_gate17_6_accuracy_calibrated_h6_fill import _requested_support_count
from experiments.scripts.run_gate17_support_selection import _row_from_task
from experiments.scripts.run_gate19_cost_normalized_stc import _edge_count, _eval_task, leakage_audit_row
from experiments.scripts.run_gate19_1_calibrated_baseline_audit import _full_contexts, _labels_from_nodes, _support_cost_row, _task_pred_payload
from experiments.scripts.summarize_gate19 import _bool, _float, read_csv
from experiments.scripts.summarize_gate20_cal import summarize
from hesf_coarsen.eval.calibration import apply_logit_calibration, calibration_param_bytes, nested_calibration_split
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.logit_ensemble import calibration_metrics, scores_from_logits
from hesf_coarsen.eval.per_class import confusion_matrix_rows, per_class_lookup, per_class_metrics
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.task_first.costs.accounting import CompressionCost, compute_feature_cache_bytes, compute_total_storage_ratio
from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import build_semantic_tree_cache


SUPPORT_BASELINES = (
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)
MAIN_SOURCES = ("H6-no-spec-support-only", "flatten-sum-support-only", "TypedHash-ChebHeat-support-only")
HESF_NAMES = {
    "H6-no-spec-support-only": "HeSF-CAL-H6",
    "flatten-sum-support-only": "HeSF-CAL-flatten",
    "TypedHash-ChebHeat-support-only": "HeSF-CAL-TypedHash",
    "random-support-only": "HeSF-CAL-random-negative-control",
}
LEGACY_NAMES = {source: f"{source}-logit-calibrated" for source in SUPPORT_BASELINES}
LEGACY_NAMES["best-support"] = "best-support-baseline-logit-calibrated"
TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
CLASS_BIAS_GRID = (-1.00, -0.50, -0.25, 0.00, 0.25, 0.50, 1.00)
BIAS_L2_PENALTY = (0.0, 0.001, 0.01)
MACRO_GUARD_EPSILON = 0.005


def make_repro_metadata(*, script_name: str, config: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    canonical = json.dumps(dict(config), sort_keys=True, default=str)
    return {
        "git_commit": git_commit_hash() or "",
        "script_name": str(script_name),
        "config_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "run_id": str(run_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def with_repro_metadata(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("git_commit", "script_name", "config_hash", "run_id", "timestamp"):
        out[key] = metadata.get(key, "")
    return out


def _git_output(args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=False)
    return (completed.stdout if completed.returncode == 0 else completed.stderr).strip()


def _candidate_biases(num_classes: int) -> list[np.ndarray]:
    # Modest class-bias grid: zero vector plus one-class shifts using the requested values.
    biases = [np.zeros(int(num_classes), dtype=np.float32)]
    for cls in range(int(num_classes)):
        for value in CLASS_BIAS_GRID:
            if abs(float(value)) <= 1.0e-12:
                continue
            bias = np.zeros(int(num_classes), dtype=np.float32)
            bias[cls] = float(value)
            biases.append(bias)
    return biases


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _bias_dict(bias: np.ndarray) -> dict[str, float]:
    return {str(i): float(value) for i, value in enumerate(np.asarray(bias, dtype=np.float32).reshape(-1).tolist())}


def _bias_from_any(value: Any, class_count: int) -> np.ndarray:
    if isinstance(value, Mapping):
        out = np.zeros(int(class_count), dtype=np.float32)
        for key, raw in value.items():
            idx = int(key)
            if 0 <= idx < class_count:
                out[idx] = float(raw)
        return out
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < class_count:
        padded = np.zeros(int(class_count), dtype=np.float32)
        padded[: arr.size] = arr
        return padded
    return arr[:class_count]


def fit_gate20_calibration_from_logits(
    val_logits: Any,
    val_labels: Any,
    test_logits: Any,
    test_labels: Any,
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    split_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    val_arr = np.asarray(val_logits, dtype=np.float32)
    val_y = np.asarray(val_labels, dtype=np.int64).reshape(-1)
    test_arr = np.asarray(test_logits, dtype=np.float32)
    test_y = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    if val_arr.ndim != 2 or test_arr.ndim != 2 or val_arr.shape[0] != val_y.shape[0] or test_arr.shape[0] != test_y.shape[0]:
        raise ValueError("invalid Gate20 calibration logits/labels")
    class_count = int(val_arr.shape[1])
    biases = _candidate_biases(class_count)
    nested_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_payloads: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    uncal_test_scores = scores_from_logits(test_arr, test_y)
    for repeat_idx, split_seed in enumerate(split_seeds):
        split = nested_calibration_split(np.arange(len(val_y), dtype=np.int64), val_y, seed=int(split_seed))
        calib_idx = split["val_calib"]
        select_idx = split["val_select"]
        uncal_select = scores_from_logits(val_arr[select_idx], val_y[select_idx])
        uncal_select_quality = calibration_metrics(val_arr[select_idx], val_y[select_idx])
        uncal_test_quality = calibration_metrics(test_arr, test_y)
        min_macro = float(uncal_select["macro_f1"]) - MACRO_GUARD_EPSILON
        best: dict[str, Any] | None = None
        best_val_logits: np.ndarray | None = None
        best_test_logits: np.ndarray | None = None
        for temperature, bias, l2_penalty in product(TEMPERATURE_GRID, biases, BIAS_L2_PENALTY):
            adjusted_val = apply_logit_calibration(val_arr, float(temperature), bias)
            adjusted_test = apply_logit_calibration(test_arr, float(temperature), bias)
            select_scores = scores_from_logits(adjusted_val[select_idx], val_y[select_idx])
            select_quality = calibration_metrics(adjusted_val[select_idx], val_y[select_idx])
            satisfied = float(select_scores["macro_f1"]) >= min_macro
            bias_l2 = float(np.linalg.norm(bias))
            candidate = {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "ratio": float(ratio),
                "nested_repeat": int(repeat_idx),
                "split_seed": int(split_seed),
                "temperature": float(temperature),
                "class_bias_vector": _bias_dict(bias),
                "bias_l2_penalty": float(l2_penalty),
                "class_bias_l2": bias_l2,
                "val_select_accuracy": float(select_scores["accuracy"]),
                "val_select_macro_f1": float(select_scores["macro_f1"]),
                "val_select_nll": float(select_quality["NLL"]),
                "val_select_ece": float(select_quality["ECE"]),
                "constraint_satisfied": bool(satisfied),
                "uses_test_labels": False,
            }
            candidate_rows.append(candidate)
            key = (
                bool(satisfied),
                float(select_scores["accuracy"]),
                float(select_scores["macro_f1"]),
                -float(select_quality["NLL"]),
                -float(select_quality["ECE"]),
                -bias_l2,
                -abs(float(temperature) - 1.0),
            )
            if best is None or key > best["_key"]:
                best = {**candidate, "_key": key}
                best_val_logits = adjusted_val
                best_test_logits = adjusted_test
        assert best is not None and best_val_logits is not None and best_test_logits is not None
        test_scores = scores_from_logits(best_test_logits, test_y)
        test_quality = calibration_metrics(best_test_logits, test_y)
        val_select_scores = scores_from_logits(best_val_logits[select_idx], val_y[select_idx])
        nested_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "ratio": float(ratio),
                "nested_repeat": int(repeat_idx),
                "split_seed": int(split_seed),
                "val_calib_size": int(len(calib_idx)),
                "val_select_size": int(len(select_idx)),
                "selected_temperature": float(best["temperature"]),
                "selected_class_bias_vector": best["class_bias_vector"],
                "val_select_macro_f1": float(val_select_scores["macro_f1"]),
                "val_select_accuracy": float(val_select_scores["accuracy"]),
                "test_macro_f1": float(test_scores["macro_f1"]),
                "test_macro": float(test_scores["macro_f1"]),
                "test_accuracy": float(test_scores["accuracy"]),
                "generalization_gap_accuracy": float(val_select_scores["accuracy"] - test_scores["accuracy"]),
                "generalization_gap_macro": float(val_select_scores["macro_f1"] - test_scores["macro_f1"]),
                "constraint_satisfied": bool(best["constraint_satisfied"]),
            }
        )
        quality_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "method": method,
                "ratio": float(ratio),
                "split_seed": int(split_seed),
                "ece_bins": 10,
                "uncalibrated_ece": float(uncal_test_quality["ECE"]),
                "calibrated_ece": float(test_quality["ECE"]),
                "uncalibrated_nll": float(uncal_test_quality["NLL"]),
                "calibrated_nll": float(test_quality["NLL"]),
                "uncalibrated_brier": float(uncal_test_quality["Brier"]),
                "calibrated_brier": float(test_quality["Brier"]),
                "delta_ece": float(test_quality["ECE"] - uncal_test_quality["ECE"]),
                "delta_nll": float(test_quality["NLL"] - uncal_test_quality["NLL"]),
                "delta_brier": float(test_quality["Brier"] - uncal_test_quality["Brier"]),
                "uncalibrated_accuracy": float(uncal_test_scores["accuracy"]),
                "calibrated_accuracy": float(test_scores["accuracy"]),
                "uncalibrated_macro_f1": float(uncal_test_scores["macro_f1"]),
                "calibrated_macro_f1": float(test_scores["macro_f1"]),
                "constraint_satisfied": bool(best["constraint_satisfied"]),
                "temperature": float(best["temperature"]),
                "class_bias_l2": float(best["class_bias_l2"]),
                "uncalibrated_val_select_ece": float(uncal_select_quality["ECE"]),
                "uncalibrated_val_select_nll": float(uncal_select_quality["NLL"]),
            }
        )
        selected_payloads.append((best, best_val_logits, best_test_logits))
    acc_mean, acc_std = _mean_std([_float(row["test_accuracy"]) for row in nested_rows])
    macro_mean, macro_std = _mean_std([_float(row["test_macro_f1"]) for row in nested_rows])
    val_acc_mean, _ = _mean_std([_float(row["val_select_accuracy"]) for row in nested_rows])
    val_macro_mean, _ = _mean_std([_float(row["val_select_macro_f1"]) for row in nested_rows])
    temps = [_float(row["selected_temperature"]) for row in nested_rows]
    temp_mean, temp_std = _mean_std(temps)
    bias_vectors = [_bias_from_any(row["selected_class_bias_vector"], class_count) for row in nested_rows]
    bias_stack = np.stack(bias_vectors, axis=0).astype(np.float32)
    bias_mean = np.mean(bias_stack, axis=0)
    bias_std = np.std(bias_stack, axis=0)
    bias_l2 = [float(np.linalg.norm(vec)) for vec in bias_vectors]
    bias_l2_mean, bias_l2_std = _mean_std(bias_l2)
    constraint_rate = float(np.mean([_bool(row["constraint_satisfied"]) for row in nested_rows])) if nested_rows else 0.0
    summary = {
        "nested_accuracy_mean": acc_mean,
        "nested_accuracy_std": acc_std,
        "nested_macro_mean": macro_mean,
        "nested_macro_std": macro_std,
        "nested_val_select_accuracy_mean": val_acc_mean,
        "nested_val_select_macro_mean": val_macro_mean,
        "calibration_constraint_satisfied_rate": constraint_rate,
        "temperature_mean": temp_mean,
        "temperature_std": temp_std,
        "class_bias_mean_vector": _bias_dict(bias_mean),
        "class_bias_std_vector": _bias_dict(bias_std),
        "class_bias_l2_mean": bias_l2_mean,
        "class_bias_l2_std": bias_l2_std,
    }
    for row in nested_rows:
        row.update(summary)
    first, first_val_logits, first_test_logits = selected_payloads[0]
    first_test_scores = scores_from_logits(first_test_logits, test_y)
    first_val_scores = scores_from_logits(first_val_logits, val_y)
    calibrated = {
        "macro_f1": acc_mean * 0.0 + macro_mean,
        "micro_f1": acc_mean,
        "accuracy": acc_mean,
        "validation_macro_f1": val_macro_mean,
        "validation_micro_f1": val_acc_mean,
        "validation_accuracy": val_acc_mean,
        "projected_macro_f1": macro_mean,
        "projected_accuracy": acc_mean,
        "projected_val_logits": np.asarray(first_val_logits, dtype=np.float32).tolist(),
        "projected_test_logits": np.asarray(first_test_logits, dtype=np.float32).tolist(),
        "projected_val_labels": val_y.tolist(),
        "projected_test_labels": test_y.tolist(),
        "projected_val_pred": first_val_scores["pred"],
        "projected_test_pred": first_test_scores["pred"],
    }
    diag = {
        "calibration_method": "temperature_scaling;class_bias_grid_search;macro_guarded_accuracy_search",
        "calibration_uses_test_labels": False,
        "calibrator_uses_test_labels": False,
        "selected_temperature": float(first["temperature"]),
        "selected_class_bias_vector": first["class_bias_vector"],
        "temperature": float(first["temperature"]),
        "class_bias_vector": first["class_bias_vector"],
        "class_bias_l2": float(first["class_bias_l2"]),
        "uncalibrated_accuracy": float(uncal_test_scores["accuracy"]),
        "uncalibrated_macro_f1": float(uncal_test_scores["macro_f1"]),
        "calibrated_accuracy": acc_mean,
        "calibrated_macro_f1": macro_mean,
        "delta_accuracy_from_calibration": float(acc_mean - uncal_test_scores["accuracy"]),
        "delta_macro_from_calibration": float(macro_mean - uncal_test_scores["macro_f1"]),
        **summary,
    }
    return calibrated, diag, nested_rows, quality_rows, candidate_rows


def select_best_support_candidate(candidates: Sequence[Mapping[str, Any]], *, best_uncalibrated_support_val_macro: float) -> dict[str, Any]:
    valid = [
        row
        for row in candidates
        if str(row.get("method")) in {"HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash"}
        and _float(row.get("validation_macro_f1")) >= float(best_uncalibrated_support_val_macro) - MACRO_GUARD_EPSILON
    ]
    if not valid:
        valid = [row for row in candidates if str(row.get("method")) in {"HeSF-CAL-H6", "HeSF-CAL-flatten", "HeSF-CAL-TypedHash"}]
    selected = dict(max(valid, key=lambda row: (_float(row.get("validation_accuracy")), _float(row.get("validation_macro_f1")), -_float(row.get("total_storage_ratio_vs_full_stc"), 1.0e9))))
    selected["selection_uses_test_labels"] = False
    return selected


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "datasets": [str(x).upper() for x in args.datasets],
        "seeds": [int(x) for x in args.seeds],
        "support_ratios": [float(x) for x in args.support_ratios],
        "nested_split_seeds": [int(x) for x in args.nested_split_seeds],
        "primary_eval_mode": str(args.primary_eval_mode),
        "task_epochs": int(args.task_epochs),
        "task_hidden_dim": int(args.task_hidden_dim),
        "max_paths": int(args.max_paths),
        "candidate_k": int(args.candidate_k),
    }


def _cost_row_from_compression(cost: CompressionCost, *, calibration_bytes: int = 0) -> dict[str, Any]:
    row = asdict(compute_total_storage_ratio(cost))
    row["calibration_param_bytes"] = int(calibration_bytes)
    row["total_inference_storage_bytes"] = int(row["total_storage_bytes"]) + int(calibration_bytes)
    full_stc = max(1, int(cost.full_feature_cache_bytes + cost.full_model_param_bytes + cost.full_logit_cache_bytes))
    full_graph = max(1, int(full_stc + cost.full_support_node_count * 8 + cost.full_support_edge_count * 16 + cost.full_unit_count * 8))
    row["model_param_bytes"] = int(row.get("model_param_bytes", 0)) + int(calibration_bytes)
    row["total_storage_bytes"] = int(row["total_inference_storage_bytes"])
    row["total_storage_ratio_vs_full_stc"] = float(row["total_inference_storage_bytes"] / full_stc)
    row["total_storage_ratio_vs_full_graph"] = float(row["total_inference_storage_bytes"] / full_graph)
    row["cost_axis_used"] = "total_storage_ratio_vs_full_stc"
    return row


def _full_contexts_from_gate19(gate19_dir: Path) -> dict[str, dict[str, int]]:
    contexts, _denoms = _full_contexts(read_csv(gate19_dir / "gate19_cost_breakdown.csv"))
    return contexts


def _fallback_full_context(graph: Any, target_type: int, *, max_paths: int) -> dict[str, int]:
    cache = build_semantic_tree_cache(graph, target_type=int(target_type), max_hops=2, max_paths=int(max_paths))
    return {
        "full_support_node_count": int(np.sum(np.asarray(graph.node_type) != int(target_type))),
        "full_support_edge_count": int(_edge_count(graph)),
        "full_path_channel_count": int(len(cache.paths)),
        "full_feature_cache_elements": int(np.asarray(cache.tensor).size),
        "full_feature_cache_bytes": int(compute_feature_cache_bytes(cache.tensor, np.float32)),
        "full_model_param_bytes": 1,
    }


def _task_payload_for_calibrated(base_task: Mapping[str, Any], calibrated: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base_task)
    out.update(dict(calibrated))
    out["projected_original_macro_f1"] = calibrated.get("projected_macro_f1", calibrated.get("macro_f1", ""))
    out["projected_original_accuracy"] = calibrated.get("projected_accuracy", calibrated.get("accuracy", ""))
    out["primary_task_metric_name"] = "projected_original_macro_f1"
    out["primary_eval_mode"] = "compressed_projected"
    return out


def _base_raw_row(*, dataset: str, seed: int, method: str, method_family: str, ratio: float, diagnostic_only: bool, eligible: bool, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return with_repro_metadata(
        {
            "stage": "Gate20-CAL",
            "dataset": dataset,
            "seed": int(seed),
            "method": method,
            "method_family": method_family,
            "ratio": float(ratio),
            "requested_budget": float(ratio),
            "requested_support_ratio": float(ratio),
            "status": "success",
            "diagnostic_only": bool(diagnostic_only),
            "eligible_for_main_decision": bool(eligible),
            "exclude_from_nested_gate": bool(diagnostic_only or not eligible),
            "primary_eval_mode": "compressed_projected",
            "no_test_leakage": True,
            "calibration_uses_test_labels": False,
            "selector_uses_test_labels": False,
            "failed": False,
            "failure_reason": "",
        },
        metadata,
    )


def _copy_cost_fields(row: dict[str, Any], cost: Mapping[str, Any]) -> None:
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
        "total_storage_bytes",
        "total_inference_storage_bytes",
        "total_storage_ratio_vs_full_stc",
        "total_storage_ratio_vs_full_graph",
        "cost_axis_used",
    ):
        row[key] = cost.get(key, 0)


def _gate20_per_class_rows(
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    uncalibrated_lookup: Mapping[int, Mapping[str, float]] | None,
    h6_lookup: Mapping[int, Mapping[str, float]] | None,
    selected_support_labels: Sequence[int] = (),
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = per_class_metrics(
        dataset=dataset,
        seed=int(seed),
        method=method,
        method_family="gate20",
        requested_budget=float(ratio),
        cost_ratio=0.0,
        total_storage_ratio_vs_full_stc=0.0,
        calibrated=True,
        source_method="",
        y_true=y_true,
        y_pred=y_pred,
        baseline_per_class=uncalibrated_lookup,
        best_uncalibrated_support_per_class=h6_lookup,
    )
    selected_counts: dict[int, int] = {}
    for label in np.asarray(selected_support_labels, dtype=np.int64).reshape(-1).tolist():
        if int(label) >= 0:
            selected_counts[int(label)] = selected_counts.get(int(label), 0) + 1
    out: list[dict[str, Any]] = []
    for row in rows:
        cls = int(row.get("class_id", 0))
        out.append(
            with_repro_metadata(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "method": method,
                    "ratio": float(ratio),
                    "class_id": cls,
                    "support_count_selected": int(selected_counts.get(cls, 0)),
                    "true_test_count": row.get("true_count", row.get("class_support_test", 0)),
                    "predicted_test_count": row.get("predicted_count", 0),
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1": row.get("f1", 0.0),
                    "accuracy_contribution": row.get("accuracy_contribution", 0.0),
                    "uncalibrated_precision": (uncalibrated_lookup or {}).get(cls, {}).get("precision", row.get("precision", 0.0)),
                    "uncalibrated_recall": (uncalibrated_lookup or {}).get(cls, {}).get("recall", row.get("recall", 0.0)),
                    "uncalibrated_f1": (uncalibrated_lookup or {}).get(cls, {}).get("f1", row.get("f1", 0.0)),
                    "delta_precision_from_calibration": row.get("delta_precision_vs_uncalibrated_source", 0.0),
                    "delta_recall_from_calibration": row.get("delta_recall_vs_uncalibrated_source", 0.0),
                    "delta_f1_from_calibration": row.get("delta_f1_vs_uncalibrated_source", 0.0),
                    "delta_recall_vs_uncalibrated_H6": float(row.get("recall", 0.0)) - float((h6_lookup or {}).get(cls, {}).get("recall", row.get("recall", 0.0))),
                    "delta_precision_vs_uncalibrated_H6": float(row.get("precision", 0.0)) - float((h6_lookup or {}).get(cls, {}).get("precision", row.get("precision", 0.0))),
                },
                metadata,
            )
        )
    return out


def _gate20_confusion_rows(
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    uncalibrated_counts: Mapping[tuple[int, int], int] | None,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = confusion_matrix_rows(dataset=dataset, seed=int(seed), method=method, requested_budget=float(ratio), calibrated=True, source_method="", y_true=y_true, y_pred=y_pred)
    out: list[dict[str, Any]] = []
    for row in rows:
        true_cls = int(row.get("true_class", 0))
        pred_cls = int(row.get("predicted_class", 0))
        uncal_count = int((uncalibrated_counts or {}).get((true_cls, pred_cls), 0))
        count = int(row.get("count", 0))
        out.append(
            with_repro_metadata(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "method": method,
                    "ratio": float(ratio),
                    "true_class": true_cls,
                    "pred_class": pred_cls,
                    "count": count,
                    "normalized_by_true": row.get("normalized_by_true", 0.0),
                    "normalized_by_pred": row.get("normalized_by_pred", 0.0),
                    "uncalibrated_count": uncal_count,
                    "count_delta_from_calibration": int(count - uncal_count),
                },
                metadata,
            )
        )
    return out


def _confusion_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for row in rows:
        out[(int(row.get("true_class", 0)), int(row.get("predicted_class", row.get("pred_class", 0))))] = int(row.get("count", 0))
    return out


def _write_class_shift_report(output_dir: Path, per_class_rows: Sequence[Mapping[str, Any]], confusion_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Gate20-CAL Class Shift Report", "", "## DBLP calibration focus"]
    db = [row for row in per_class_rows if str(row.get("dataset")) == "DBLP" and str(row.get("method")) == "HeSF-CAL-best-support" and abs(_float(row.get("ratio")) - 0.30) < 1.0e-12]
    class1 = [row for row in db if int(_float(row.get("class_id"), -1)) == 1]
    if class1:
        delta_recall = np.mean([_float(row.get("delta_recall_from_calibration")) for row in class1])
        delta_f1 = np.mean([_float(row.get("delta_f1_from_calibration")) for row in class1])
        lines.append(f"- class 1 mean recall delta from calibration: {delta_recall:.6f}; F1 delta: {delta_f1:.6f}.")
    for cls in (0, 2, 3):
        cls_rows = [row for row in db if int(_float(row.get("class_id"), -1)) == cls]
        if cls_rows:
            delta = np.mean([_float(row.get("delta_recall_from_calibration")) for row in cls_rows])
            lines.append(f"- class {cls} mean recall delta from calibration: {delta:.6f}.")
    pred_counts: dict[int, int] = {}
    for row in confusion_rows:
        if str(row.get("dataset")) == "DBLP" and str(row.get("method")) == "HeSF-CAL-best-support" and abs(_float(row.get("ratio")) - 0.30) < 1.0e-12:
            pred = int(_float(row.get("pred_class"), 0))
            pred_counts[pred] = pred_counts.get(pred, 0) + int(_float(row.get("count"), 0))
    lines.append(f"- predicted class distribution after calibration: {json.dumps(pred_counts, sort_keys=True)}")
    lines.append("- Calibration is interpreted as class-prior correction only when accuracy improves without large negative per-class recall deltas.")
    lines.append("- Inspect `gate20_cal_per_class_metrics.csv` for any class collapse hidden by aggregate accuracy.")
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "gate20_cal_class_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "gate20_class_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_method_to_code_path(output_dir: Path, metadata: Mapping[str, Any]) -> None:
    rows = []
    for method in [
        "full-graph-hettree-lite-tuned",
        "target-only-empty-support",
        *SUPPORT_BASELINES,
        "HeSF-CAL-H6",
        "HeSF-CAL-flatten",
        "HeSF-CAL-TypedHash",
        "HeSF-CAL-best-support",
    ]:
        rows.append(
            with_repro_metadata(
                {
                    "method": method,
                    "method_family": "hesf_cal" if method.startswith("HeSF-CAL") else "support_baseline",
                    "runner_function": "run",
                    "builder_function": "run_support_baseline / full_graph / target_only",
                    "evaluator_function": "evaluate_hettree_task",
                    "calibration_function": "fit_gate20_calibration_from_logits" if method.startswith("HeSF-CAL") else "none",
                    "summary_inclusion": "main" if method.startswith("HeSF-CAL") and "random" not in method else "diagnostic_or_baseline",
                    "eligible_for_main_decision": method.startswith("HeSF-CAL") and "random" not in method,
                    "diagnostic_only": "random" in method or method in {"full-graph-hettree-lite-tuned", "target-only-empty-support"},
                },
                metadata,
            )
        )
    write_csv(output_dir / "diagnostics" / "gate20_cal_method_to_code_path.csv", rows)


def _write_requirement_checklist(output_dir: Path, result: Mapping[str, Any]) -> None:
    raw = read_csv(output_dir / "gate20_cal_raw_rows.csv")
    methods = {str(row.get("method")) for row in raw}
    datasets = {str(row.get("dataset")) for row in raw if row.get("dataset") not in {"", None}}
    seeds = {int(_float(row.get("seed"))) for row in raw if row.get("seed") not in {"", None}}
    ratios = {round(_float(row.get("ratio")), 2) for row in raw if str(row.get("method")).startswith("HeSF-CAL")}
    required_files = [
        "gate20_cal_raw_rows.csv",
        "gate20_cal_validation_selected_by_method.csv",
        "gate20_cal_by_dataset_selected.csv",
        "gate20_cal_pareto_frontier.csv",
        "gate20_cal_exact_ratio_comparison.csv",
        "gate20_cal_nested_calibration.csv",
        "gate20_cal_calibration_quality.csv",
        "gate20_cal_per_class_metrics.csv",
        "gate20_cal_confusion_matrix_by_method.csv",
        "gate20_cal_result.json",
        "gate20_cal_decision.md",
        "diagnostics/gate20_cal_method_to_code_path.csv",
        "diagnostics/gate20_cal_leakage_audit.csv",
        "diagnostics/gate20_cal_cost_breakdown.csv",
        "diagnostics/gate20_cal_class_shift_report.md",
        "diagnostics/gate20_cal_nested_stability_by_method.csv",
        "diagnostics/gate20_cal_seedwise_dblp_summary.csv",
    ]
    lines = [
        "# Gate20-CAL Requirement Checklist",
        "",
        f"- [{'x' if datasets >= {'ACM', 'DBLP', 'IMDB'} else ' '}] datasets ACM/DBLP/IMDB present.",
        f"- [{'x' if {12345, 23456, 34567, 45678, 56789} <= seeds else ' '}] five required seeds present.",
        f"- [{'x' if {0.30, 0.50, 0.70} <= ratios else ' '}] support ratios 0.30/0.50/0.70 present.",
        f"- [{'x' if {'HeSF-CAL-H6', 'HeSF-CAL-flatten', 'HeSF-CAL-TypedHash', 'HeSF-CAL-best-support'} <= methods else ' '}] canonical HeSF-CAL methods present.",
        f"- [{'x' if {'full-graph-hettree-lite-tuned', 'target-only-empty-support', *SUPPORT_BASELINES} <= methods else ' '}] required uncalibrated/reference baselines present.",
        f"- [{'x' if result.get('primary_eval_mode') == 'compressed_projected' else ' '}] primary_eval_mode compressed_projected.",
        f"- [{'x' if result.get('no_test_leakage') else ' '}] no test leakage.",
        f"- [{'x' if not result.get('test_oracle_used_for_decision') else ' '}] decision uses validation selection only.",
        f"- [{'x' if result.get('typedhash_included') else ' '}] TypedHash included.",
        f"- [{'x' if result.get('per_class_confusion_present') else ' '}] per-class/confusion diagnostics present.",
        f"- [x] STC/support-teacher distillation frozen as diagnostic-only; not rerun for mainline.",
        f"- [x] decision label: `{result.get('decision')}`.",
        "",
        "## Required Output Files",
    ]
    lines.extend(f"- [{'x' if (output_dir / name).exists() else ' '}] `{name}`" for name in required_files)
    (output_dir / "gate20_cal_requirement_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_zip(output_dir: Path, name: str, files: Sequence[str]) -> None:
    with zipfile.ZipFile(output_dir / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in files:
            path = output_dir / file_name
            if path.exists():
                archive.write(path, arcname=file_name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate20-CAL requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.output_dir)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    config = _config_from_args(args)
    run_id = "gate20_cal_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = make_repro_metadata(script_name="experiments/scripts/run_gate20_cal_multiseed.py", config=config, run_id=run_id)
    full_contexts = _full_contexts_from_gate19(Path(args.gate19_input_dir))

    raw_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows_out: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    stc_frozen_rows: list[dict[str, Any]] = []

    _write_method_to_code_path(output_dir, metadata)

    for dataset in [str(item).upper() for item in args.datasets]:
        for seed in [int(item) for item in args.seeds]:
            graph = load_hgb_graph(Path(args.data_root), dataset)
            labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1), dtype=np.int64)
            target_type = infer_target_node_type(graph)
            train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
            split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
            support_count = int(np.sum(np.asarray(graph.node_type) != int(target_type)))
            full_context = full_contexts.get(dataset) or _fallback_full_context(graph, int(target_type), max_paths=int(args.max_paths))

            for ref_method, ref_ratio, family in (
                ("full-graph-hettree-lite-tuned", 1.0, "full_graph_reference"),
                ("target-only-empty-support", 0.0, "target_only_reference"),
            ):
                try:
                    row = _base_raw_row(dataset=dataset, seed=int(seed), method=ref_method, method_family=family, ratio=float(ref_ratio), diagnostic_only=True, eligible=False, metadata=metadata)
                    if ref_method == "full-graph-hettree-lite-tuned":
                        task_row = _full_graph_row(graph, dataset, int(seed), float(ref_ratio), args, split)
                        support_nodes = support_count
                        support_edges = _edge_count(graph)
                    else:
                        task_row, _coarse, _assignment = _target_only_row(graph, dataset, int(seed), float(ref_ratio), args, split)
                        support_nodes = 0
                        support_edges = 0
                    row.update(task_row)
                    row["method_family"] = family
                    row["diagnostic_only"] = True
                    row["eligible_for_main_decision"] = False
                    row["ratio"] = float(ref_ratio)
                    cost = _cost_row_from_compression(
                        CompressionCost(
                            method=ref_method,
                            dataset=dataset,
                            seed=int(seed),
                            requested_budget=float(ref_ratio),
                            support_node_count=int(support_nodes),
                            support_edge_count=int(support_edges),
                            full_support_node_count=int(full_context["full_support_node_count"]),
                            full_support_edge_count=int(full_context["full_support_edge_count"]),
                            full_path_channel_count=int(full_context["full_path_channel_count"]),
                            full_feature_cache_elements=int(full_context["full_feature_cache_elements"]),
                            full_feature_cache_bytes=int(full_context["full_feature_cache_bytes"]),
                            full_model_param_bytes=int(full_context["full_model_param_bytes"]),
                        )
                    )
                    _copy_cost_fields(row, cost)
                    raw_rows.append(with_repro_metadata(row, metadata))
                    cost_rows.append(with_repro_metadata({**cost, "method": ref_method, "dataset": dataset, "seed": int(seed), "ratio": float(ref_ratio)}, metadata))
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    raw_rows.append(with_repro_metadata({"stage": "Gate20-CAL", "dataset": dataset, "seed": int(seed), "method": ref_method, "ratio": ref_ratio, "status": "failed", "failed": True, "failure_reason": repr(exc), "eligible_for_main_decision": False, "diagnostic_only": True, "primary_eval_mode": "compressed_projected"}, metadata))

            for ratio in [float(item) for item in args.support_ratios]:
                support_payloads: dict[str, dict[str, Any]] = {}
                h6_lookup: dict[int, dict[str, float]] | None = None
                best_uncal_val_macro = -1.0
                for baseline in SUPPORT_BASELINES:
                    start = perf_counter()
                    row = _base_raw_row(dataset=dataset, seed=int(seed), method=baseline, method_family="support_baseline", ratio=float(ratio), diagnostic_only=False, eligible=False, metadata=metadata)
                    row.update(split_protocol)
                    coarse, assignment, diag = run_support_baseline(graph, baseline=baseline, ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                    assignment = np.asarray(assignment, dtype=np.int64)
                    task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                    selected_nodes = selected_support_representatives_from_assignment(graph, assignment, int(target_type))
                    selected_count = int(diag.get("final_support_nodes", len(selected_nodes)))
                    row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                    _row_from_task(row, dict(task))
                    row["ratio"] = float(ratio)
                    row["selected_support_count"] = int(selected_count)
                    row["requested_support_count"] = _requested_support_count(support_count, float(ratio))
                    row["actual_support_ratio"] = float(selected_count / max(1, support_count))
                    row["wall_clock_sec"] = float(perf_counter() - start)
                    row["eligible_for_main_decision"] = False
                    row["diagnostic_only"] = False
                    row["calibration_uses_test_labels"] = False
                    row["selector_uses_test_labels"] = False
                    cost = _support_cost_row(method=baseline, dataset=dataset, seed=int(seed), requested_budget=float(ratio), support_node_count=selected_count, support_edge_count=_edge_count(coarse), full_context=full_context)
                    _copy_cost_fields(row, cost)
                    raw_rows.append(with_repro_metadata(row, metadata))
                    cost_rows.append(with_repro_metadata({**cost, "method": baseline, "dataset": dataset, "seed": int(seed), "ratio": float(ratio)}, metadata))
                    leak = leakage_audit_row(method=baseline, uses_train_labels=True, uses_val_labels=False, uses_test_labels_before_final_eval=False, calibration_split="none", path_selection_split="train", teacher_training_split="none", student_training_split="train")
                    leakage_rows.append(with_repro_metadata({"dataset": dataset, "seed": int(seed), "method": baseline, "ratio": float(ratio), "uses_test_labels": False, "calibration_uses_test_labels": False, "selector_uses_test_labels": False, "leakage_notes": "support baseline uses train labels only", **leak}, metadata))
                    y_true, y_pred = _task_pred_payload(task, split_name="test")
                    pc_source = per_class_metrics(dataset=dataset, seed=int(seed), method=baseline, method_family="support_baseline", requested_budget=float(ratio), cost_ratio=_float(cost.get("total_storage_ratio_vs_full_stc")), total_storage_ratio_vs_full_stc=_float(cost.get("total_storage_ratio_vs_full_stc")), calibrated=False, source_method="", y_true=y_true, y_pred=y_pred, train_labels=_labels_from_nodes(labels, train_nodes), val_labels=_labels_from_nodes(labels, val_nodes))
                    source_lookup = per_class_lookup(pc_source)
                    if baseline == "H6-no-spec-support-only":
                        h6_lookup = source_lookup
                    uncal_confusion = confusion_matrix_rows(dataset=dataset, seed=int(seed), method=baseline, requested_budget=float(ratio), calibrated=False, source_method="", y_true=y_true, y_pred=y_pred)
                    support_payloads[baseline] = {
                        "task": dict(task),
                        "row": row,
                        "cost": cost,
                        "source_lookup": source_lookup,
                        "confusion_lookup": _confusion_lookup(uncal_confusion),
                        "selected_labels": _labels_from_nodes(labels, selected_nodes),
                        "support_node_count": selected_count,
                        "support_edge_count": _edge_count(coarse),
                    }
                    best_uncal_val_macro = max(best_uncal_val_macro, _float(row.get("validation_macro_f1")))
                hesf_rows_for_best: list[dict[str, Any]] = []
                hesf_tasks: dict[str, dict[str, Any]] = {}
                for source in SUPPORT_BASELINES:
                    payload = support_payloads[source]
                    method = HESF_NAMES[source]
                    calibrated, diag, nested, quality, candidates = fit_gate20_calibration_from_logits(
                        payload["task"].get("projected_val_logits", []),
                        payload["task"].get("projected_val_labels", []),
                        payload["task"].get("projected_test_logits", []),
                        payload["task"].get("projected_test_labels", []),
                        dataset=dataset,
                        seed=int(seed),
                        method=method,
                        ratio=float(ratio),
                        split_seeds=[int(x) for x in args.nested_split_seeds],
                    )
                    cal_task = _task_payload_for_calibrated(payload["task"], calibrated)
                    cal_cost = _support_cost_row(method=method, dataset=dataset, seed=int(seed), requested_budget=float(ratio), support_node_count=int(payload["support_node_count"]), support_edge_count=int(payload["support_edge_count"]), full_context=full_context, calibration_bytes=calibration_param_bytes(int(np.asarray(payload["task"].get("projected_test_logits", [[0]])).shape[1])))
                    for out_method, family, diagnostic, eligible in (
                        (method, "hesf_cal", source == "random-support-only", source != "random-support-only"),
                        (LEGACY_NAMES[source], "calibrated_support_baseline", True, False),
                    ):
                        row = _base_raw_row(dataset=dataset, seed=int(seed), method=out_method, method_family=family, ratio=float(ratio), diagnostic_only=diagnostic, eligible=eligible, metadata=metadata)
                        _row_from_task(row, cal_task)
                        row.update(diag)
                        row["source_support_method"] = source
                        row["source_uncalibrated_macro_f1"] = _float(payload["row"].get("macro_f1"))
                        row["source_uncalibrated_accuracy"] = _float(payload["row"].get("accuracy"))
                        row["uncalibrated_macro_f1"] = _float(payload["row"].get("macro_f1"))
                        row["uncalibrated_accuracy"] = _float(payload["row"].get("accuracy"))
                        row["selected_calibrated_macro_f1"] = _float(row.get("macro_f1"))
                        row["selected_calibrated_accuracy"] = _float(row.get("accuracy"))
                        row["selection_objective_value"] = _float(row.get("validation_accuracy"))
                        row["exclude_from_nested_gate"] = bool(diagnostic or not eligible)
                        _copy_cost_fields(row, cal_cost)
                        raw_rows.append(with_repro_metadata(row, metadata))
                        if out_method == method:
                            hesf_rows_for_best.append(row)
                            hesf_tasks[method] = dict(cal_task)
                    for item in nested:
                        nested_rows.append(with_repro_metadata(item, metadata))
                    for item in quality:
                        quality_rows.append(with_repro_metadata(item, metadata))
                    for item in candidates:
                        candidate_rows.append(with_repro_metadata(item, metadata))
                    cost_rows.append(with_repro_metadata({**cal_cost, "method": method, "dataset": dataset, "seed": int(seed), "ratio": float(ratio)}, metadata))
                    leakage_rows.append(with_repro_metadata({"dataset": dataset, "seed": int(seed), "method": method, "ratio": float(ratio), "uses_train_labels": True, "uses_val_labels": True, "uses_test_labels": False, "calibration_uses_test_labels": False, "selector_uses_test_labels": False, "no_test_leakage": True, "leakage_notes": "calibration val_calib/val_select only"}, metadata))
                    y_true, y_pred = _task_pred_payload(cal_task, split_name="test")
                    per_class_rows.extend(_gate20_per_class_rows(dataset=dataset, seed=int(seed), method=method, ratio=float(ratio), y_true=y_true, y_pred=y_pred, uncalibrated_lookup=payload["source_lookup"], h6_lookup=h6_lookup, selected_support_labels=payload["selected_labels"], metadata=metadata))
                    confusion_rows_out.extend(_gate20_confusion_rows(dataset=dataset, seed=int(seed), method=method, ratio=float(ratio), y_true=y_true, y_pred=y_pred, uncalibrated_counts=payload["confusion_lookup"], metadata=metadata))
                selected_best = select_best_support_candidate(hesf_rows_for_best, best_uncalibrated_support_val_macro=best_uncal_val_macro)
                source_method = str(selected_best.get("source_support_method", selected_best.get("method", "")))
                source_hesf = str(selected_best.get("method"))
                best_task = hesf_tasks.get(source_hesf, {})
                best_row = dict(selected_best)
                best_row["method"] = "HeSF-CAL-best-support"
                best_row["method_family"] = "hesf_cal"
                best_row["source_support_method"] = source_method
                best_row["selection_uses_test_labels"] = False
                best_row["selection_objective_value"] = selected_best.get("validation_accuracy", "")
                raw_rows.append(with_repro_metadata(best_row, metadata))
                alias = dict(best_row)
                alias["method"] = "best-support-baseline-logit-calibrated"
                alias["method_family"] = "calibrated_support_baseline"
                alias["diagnostic_only"] = True
                alias["eligible_for_main_decision"] = False
                raw_rows.append(with_repro_metadata(alias, metadata))
                for row in [item for item in nested_rows if str(item.get("dataset")) == dataset and int(_float(item.get("seed"))) == int(seed) and str(item.get("method")) == source_hesf and abs(_float(item.get("ratio")) - float(ratio)) < 1.0e-12]:
                    copied = dict(row)
                    copied["method"] = "HeSF-CAL-best-support"
                    nested_rows.append(with_repro_metadata(copied, metadata))
                for row in [item for item in quality_rows if str(item.get("dataset")) == dataset and int(_float(item.get("seed"))) == int(seed) and str(item.get("method")) == source_hesf and abs(_float(item.get("ratio")) - float(ratio)) < 1.0e-12]:
                    copied = dict(row)
                    copied["method"] = "HeSF-CAL-best-support"
                    quality_rows.append(with_repro_metadata(copied, metadata))
                if best_task:
                    payload = support_payloads.get(source_method, {})
                    y_true, y_pred = _task_pred_payload(best_task, split_name="test")
                    per_class_rows.extend(_gate20_per_class_rows(dataset=dataset, seed=int(seed), method="HeSF-CAL-best-support", ratio=float(ratio), y_true=y_true, y_pred=y_pred, uncalibrated_lookup=payload.get("source_lookup"), h6_lookup=h6_lookup, selected_support_labels=payload.get("selected_labels", []), metadata=metadata))
                    confusion_rows_out.extend(_gate20_confusion_rows(dataset=dataset, seed=int(seed), method="HeSF-CAL-best-support", ratio=float(ratio), y_true=y_true, y_pred=y_pred, uncalibrated_counts=payload.get("confusion_lookup"), metadata=metadata))
                write_csv(output_dir / "gate20_cal_raw_rows.csv", raw_rows)

    gate19_2 = Path(args.gate19_2_input_dir)
    if gate19_2.exists():
        for row in read_csv(gate19_2 / "gate19_2_validation_selected_by_method.csv"):
            if str(row.get("method", "")).startswith("STC-") or str(row.get("method", "")).startswith("Full-STC"):
                stc_frozen_rows.append(with_repro_metadata({**row, "gate20_role": "frozen_diagnostic_comparator", "eligible_for_main_decision": False, "diagnostic_only": True}, metadata))

    write_csv(output_dir / "gate20_cal_raw_rows.csv", raw_rows)
    write_csv(output_dir / "gate20_cal_nested_calibration.csv", nested_rows)
    write_csv(output_dir / "gate20_cal_calibration_quality.csv", quality_rows)
    write_csv(output_dir / "gate20_cal_per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "gate20_cal_confusion_matrix_by_method.csv", confusion_rows_out)
    write_csv(diag_dir / "gate20_cal_leakage_audit.csv", leakage_rows)
    write_csv(diag_dir / "gate20_cal_cost_breakdown.csv", cost_rows)
    write_csv(diag_dir / "gate20_cal_calibration_candidates.csv", candidate_rows)
    write_csv(diag_dir / "gate20_cal_stc_frozen_comparator.csv", stc_frozen_rows)
    _write_class_shift_report(output_dir, per_class_rows, confusion_rows_out)
    result = summarize(output_dir, output_dir)
    _write_requirement_checklist(output_dir, result)
    _write_zip(output_dir, "gate20_cal_main_results.zip", ["gate20_cal_raw_rows.csv", "gate20_cal_validation_selected_by_method.csv", "gate20_cal_by_dataset_selected.csv", "gate20_cal_exact_ratio_comparison.csv", "gate20_cal_pareto_frontier.csv", "gate20_cal_result.json", "gate20_cal_decision.md", "gate20_cal_requirement_checklist.md"])
    _write_zip(output_dir, "gate20_cal_core_diagnostics.zip", ["gate20_cal_nested_calibration.csv", "gate20_cal_calibration_quality.csv", "gate20_cal_per_class_metrics.csv", "gate20_cal_confusion_matrix_by_method.csv", "diagnostics/gate20_cal_method_to_code_path.csv", "diagnostics/gate20_cal_leakage_audit.csv", "diagnostics/gate20_cal_cost_breakdown.csv", "diagnostics/gate20_cal_class_shift_report.md", "diagnostics/gate20_cal_nested_stability_by_method.csv", "diagnostics/gate20_cal_seedwise_dblp_summary.csv", "diagnostics/gate20_cal_calibration_candidates.csv", "diagnostics/gate20_cal_stc_frozen_comparator.csv"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate20-CAL multiseed experiment.")
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--support-ratios", nargs="*", type=float, default=[0.30, 0.50, 0.70])
    parser.add_argument("--nested-split-seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate20_cal"))
    parser.add_argument("--gate19-input-dir", type=Path, default=Path("outputs/gate19"))
    parser.add_argument("--gate19-2-input-dir", type=Path, default=Path("outputs/gate19_2"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
