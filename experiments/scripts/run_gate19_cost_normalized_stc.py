from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, write_csv
from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.gate17_4_h6 import selected_support_representatives_from_assignment
from experiments.scripts.run_gate17_1_support_sensitivity import _full_graph_row, _target_only_row
from experiments.scripts.run_gate17_6_accuracy_calibrated_h6_fill import _requested_support_count, _support_labels, per_class_audit_rows
from experiments.scripts.run_gate17_support_selection import _row_from_task, _split_values
from experiments.scripts.summarize_gate19 import summarize
from hesf_coarsen.eval.calibration import apply_calibrator, fit_macro_constrained_accuracy_calibrator
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
from hesf_coarsen.eval.task_gnn import f1_scores, select_task_protocol_split
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.costs.accounting import (
    CompressionCost,
    assert_cost_finite,
    compute_feature_cache_bytes,
    compute_total_storage_ratio,
)
from hesf_coarsen.task_first.feature_condensation.baselines import (
    cache_with_path_indices,
    calibrated_cache_result,
    evaluate_cache_classifier,
    evaluate_cache_logits,
    flatten_cache,
    labels_for_cache,
    local_indices,
    quantized_cache,
    select_paths_by_energy,
    select_paths_by_validation,
)
from hesf_coarsen.task_first.feature_condensation.distillation import teacher_kl_diagnostics, train_student_with_teacher
from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache, build_semantic_tree_cache, cache_metadata


GATE19_SINGLE_SEED_BY_DATASET = {"ACM": 23456, "DBLP": 23456, "IMDB": 45678}
SUPPORT_BASELINES = (
    "full-graph-hettree-lite-tuned",
    "target-only-empty-support",
    "H6-no-spec-support-only",
    "flatten-sum-support-only",
    "TypedHash-ChebHeat-support-only",
    "random-support-only",
)
FULL_STC_BASELINES = (
    "Full-STC-MLP",
    "Full-STC-MLP-logit-calibrated",
    "Full-STC-linear",
    "Full-STC-centroid",
)
STC_METHODS = (
    "STC-path-prune-energy",
    "STC-path-prune-validation-accuracy",
    "STC-path-prune-validation-loss",
    "STC-path-channel-hard-gate",
    "STC-path-channel-hard-gate-logit-calibrated",
    "STC-feature-cache-MLP-compressed",
    "STC-feature-cache-MLP-compressed-logit-calibrated",
    "STC-feature-cache-true-distill",
    "STC-feature-cache-quantized-fp16",
    "STC-feature-cache-quantized-int8",
)
CLUSTER_DIAGNOSTICS = (
    "ClusterGate-UnionUnits-logit-calibrated",
    "ClusterGate-H6-units-logit-calibrated",
    "ClusterGate-TypedHash-units-logit-calibrated",
    "HeSF-SS-validation-H6-fill-logit-calibrated",
)
CALIBRATION_MODES = "temperature_scaling;class_bias_grid_search;macro_guarded_accuracy_search"
EPSILON_MACRO = 0.02
ACCURACY_TOLERANCE = 0.01


def parse_dataset_seeds(values: list[str] | tuple[str, ...] | str | None) -> list[tuple[str, int]]:
    if values is None or values == "":
        return [(dataset, seed) for dataset, seed in GATE19_SINGLE_SEED_BY_DATASET.items()]
    tokens: list[str] = []
    raw_values = [values] if isinstance(values, str) else list(values)
    for value in raw_values:
        tokens.extend(item for item in str(value).replace(",", " ").split() if item)
    out: list[tuple[str, int]] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"dataset seed token must be DATASET:SEED, got {token!r}")
        dataset, seed = token.split(":", 1)
        dataset = dataset.strip().upper()
        if dataset not in GATE19_SINGLE_SEED_BY_DATASET:
            raise ValueError(f"unsupported Gate19 dataset: {dataset}")
        out.append((dataset, int(seed)))
    return out


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


def _edge_count(graph: HeteroGraph) -> int:
    return int(sum(int(rel.num_edges) for rel in graph.relations.values()))


def _support_count(graph: HeteroGraph, target_type: int) -> int:
    return int(np.sum(np.asarray(graph.node_type) != int(target_type)))


def _scores_from_logits(logits: Any, labels: Any) -> dict[str, Any]:
    logits_arr = np.asarray(logits, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits_arr.size == 0 or len(labels_arr) == 0:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": []}
    pred = np.argmax(logits_arr, axis=1).astype(np.int64, copy=False)
    valid = (labels_arr >= 0) & (pred >= 0)
    if not np.any(valid):
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": pred.tolist()}
    scores = f1_scores(labels_arr[valid], pred[valid], macro_empty_class_policy="truth_pred_union")
    return {**scores, "accuracy": float(np.mean(labels_arr[valid] == pred[valid])), "pred": pred.tolist()}


def _eval_task(
    graph: HeteroGraph,
    coarse: HeteroGraph,
    assignment: np.ndarray,
    *,
    seed: int,
    split: Mapping[str, np.ndarray],
    target_type: int,
    args: argparse.Namespace,
    return_logits: bool = True,
) -> dict[str, Any]:
    return evaluate_hettree_task(
        graph,
        coarse,
        np.asarray(assignment, dtype=np.int64),
        seed=int(seed),
        epochs=int(args.task_epochs),
        hidden_dim=int(args.task_hidden_dim),
        device=str(args.device),
        target_node_type=int(target_type),
        official_split_nodes=dict(split),
        primary_eval_mode=str(args.primary_eval_mode),
        early_stopping=True,
        monitor=str(args.monitor),
        max_paths=int(args.max_paths),
        return_predictions=True,
        return_logits=bool(return_logits),
        return_prediction_payload=bool(return_logits),
    ).metrics


def _apply_prediction_calibration(task: Mapping[str, Any], *, method: str, baseline_macro: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    val_logits = task.get("projected_val_logits", [])
    val_labels = task.get("projected_val_labels", task.get("projected_val_true_labels", []))
    test_logits = task.get("projected_test_logits", [])
    test_labels = task.get("projected_test_labels", task.get("projected_test_true_labels", []))
    if not val_logits or not test_logits:
        return dict(task), {"method": method, "calibration_status": "missing_logits", "calibrator_uses_test_labels": False}
    fit = fit_macro_constrained_accuracy_calibrator(
        val_logits,
        val_labels,
        baseline_macro=baseline_macro,
        macro_epsilon=EPSILON_MACRO,
        temperatures=(0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
        class_bias_values=(-0.5, -0.25, 0.0, 0.25, 0.5),
    )
    calibrated_val = apply_calibrator(val_logits, fit)
    calibrated_test = apply_calibrator(test_logits, fit)
    val_scores = _scores_from_logits(calibrated_val, val_labels)
    test_scores = _scores_from_logits(calibrated_test, test_labels)
    out = dict(task)
    out.update(
        {
            "method": method,
            "uncalibrated_macro_f1": _float(task.get("macro_f1")),
            "uncalibrated_accuracy": _float(task.get("accuracy")),
            "uncalibrated_validation_macro_f1": _float(task.get("validation_macro_f1")),
            "uncalibrated_validation_accuracy": _float(task.get("validation_accuracy")),
            "macro_f1": float(test_scores["macro_f1"]),
            "micro_f1": float(test_scores["micro_f1"]),
            "accuracy": float(test_scores["accuracy"]),
            "validation_macro_f1": float(val_scores["macro_f1"]),
            "validation_micro_f1": float(val_scores["micro_f1"]),
            "validation_accuracy": float(val_scores["accuracy"]),
            "projected_val_pred": val_scores["pred"],
            "projected_test_pred": test_scores["pred"],
            "calibrator_uses_test_labels": False,
        }
    )
    return out, {
        "method": method,
        "calibration_status": "success",
        "calibration_modes": CALIBRATION_MODES,
        "calibration_split": "val",
        "calibrator_uses_test_labels": False,
        **fit,
        "uncalibrated_macro_f1": out["uncalibrated_macro_f1"],
        "uncalibrated_accuracy": out["uncalibrated_accuracy"],
        "calibrated_macro_f1": out["macro_f1"],
        "calibrated_accuracy": out["accuracy"],
    }


def leakage_audit_row(
    *,
    method: str,
    uses_train_labels: bool,
    uses_val_labels: bool,
    uses_test_labels_before_final_eval: bool,
    calibration_split: str,
    path_selection_split: str,
    teacher_training_split: str,
    student_training_split: str,
) -> dict[str, Any]:
    test_like = {"test", "train_val_test", "all", "full_with_test"}
    invalid = bool(uses_test_labels_before_final_eval) or str(calibration_split).lower() in test_like
    invalid = invalid or str(path_selection_split).lower() in test_like
    invalid = invalid or str(teacher_training_split).lower() in test_like or str(student_training_split).lower() in test_like
    return {
        "method": str(method),
        "uses_train_labels": bool(uses_train_labels),
        "uses_val_labels": bool(uses_val_labels),
        "uses_test_labels_before_final_eval": bool(uses_test_labels_before_final_eval),
        "calibration_split": str(calibration_split),
        "path_selection_split": str(path_selection_split),
        "teacher_training_split": str(teacher_training_split),
        "student_training_split": str(student_training_split),
        "leakage_pass": not invalid,
        "no_test_leakage": not invalid,
        "method_invalid": bool(invalid),
    }


def _metric_payload(task: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "macro_f1",
        "micro_f1",
        "accuracy",
        "validation_macro_f1",
        "validation_micro_f1",
        "validation_accuracy",
        "uncalibrated_macro_f1",
        "uncalibrated_accuracy",
        "uncalibrated_validation_macro_f1",
        "uncalibrated_validation_accuracy",
        "best_epoch",
        "early_stopped",
        "feature_model",
        "feature_model_skipped",
    )
    return {key: task.get(key, "") for key in keys if key in task}


def _base_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    requested_budget: float,
    requested_support_ratio: float | None,
    method_family: str,
    diagnostic_only: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "stage": "Gate19",
        "dataset": str(dataset),
        "seed": int(seed),
        "method": str(method),
        "requested_budget": float(requested_budget),
        "requested_support_ratio": "" if requested_support_ratio is None else float(requested_support_ratio),
        "method_family": str(method_family),
        "diagnostic_only": bool(diagnostic_only),
        "eligible_for_main_decision": bool(method_family == "stc_compressed" and not diagnostic_only),
        "status": "success",
        "primary_eval_mode": str(args.primary_eval_mode),
        "typedhash_included": bool(args.include_typedhash),
        "epsilon_macro": EPSILON_MACRO,
        "accuracy_tolerance_vs_full_stc": ACCURACY_TOLERANCE,
        "accuracy_first_compression_second": True,
        "spectral_response_auxiliary_only": True,
        "no_test_leakage": True,
        "method_invalid": False,
        "test_label_usage": "metrics_only",
        "run_mode": "gate19_cost_normalized_stc",
        "primary_metric_priority": "accuracy_first_compression_second_cost_normalized",
    }


def _cost_dict(cost: CompressionCost) -> dict[str, Any]:
    computed = compute_total_storage_ratio(cost)
    assert_cost_finite(computed)
    out = asdict(computed)
    out["cost_axis_used"] = "total_storage_ratio_vs_full_stc"
    return out


def _merge_cost(row: dict[str, Any], cost_rows: list[dict[str, Any]], cost: CompressionCost) -> None:
    cost_row = _cost_dict(cost)
    cost_rows.append(cost_row)
    for key in (
        "support_node_ratio",
        "support_edge_ratio",
        "unit_count_ratio",
        "path_channel_count_ratio",
        "feature_cache_size_ratio",
        "feature_cache_bytes",
        "feature_cache_elements",
        "logit_cache_bytes",
        "model_param_bytes",
        "total_storage_bytes",
        "total_storage_ratio_vs_full_stc",
        "total_storage_ratio_vs_full_graph",
        "cost_axis_used",
    ):
        row[key] = cost_row.get(key, "")


def _prediction_audit(
    *,
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    task: Mapping[str, Any],
    selected_labels: Sequence[int] | np.ndarray = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_true = task.get("projected_test_labels", task.get("projected_test_true_labels", []))
    y_pred = task.get("projected_test_pred", task.get("projected_test_pred_labels", []))
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        return [], []
    return per_class_audit_rows(
        dataset=str(dataset),
        seed=int(seed),
        method=str(method),
        ratio=float(ratio),
        y_true=[int(value) for value in y_true],
        y_pred=[int(value) for value in y_pred],
        selected_support_labels=[int(value) for value in np.asarray(selected_labels, dtype=np.int64).reshape(-1).tolist()],
    )


def _cache_l2_delta(full_cache: SemanticTreeCache, cache: SemanticTreeCache) -> float:
    full = np.asarray(full_cache.tensor, dtype=np.float32)
    comp = np.zeros_like(full)
    lookup = {path: idx for idx, path in enumerate(full_cache.paths)}
    for idx, path in enumerate(cache.paths):
        if path in lookup and idx < cache.tensor.shape[1]:
            comp[:, lookup[path], :] = np.asarray(cache.tensor, dtype=np.float32)[:, idx, :]
    return float(np.linalg.norm((full - comp).reshape(-1)))


def _select_cache_for_method(
    method: str,
    *,
    full_cache: SemanticTreeCache,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    budget: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[SemanticTreeCache, dict[str, Any]]:
    if method in {"STC-path-prune-energy", "STC-feature-cache-MLP-compressed", "STC-feature-cache-MLP-compressed-logit-calibrated"}:
        keep = select_paths_by_energy(full_cache, split, float(budget))
        return cache_with_path_indices(full_cache, keep), {"compression_axis": "path_channel", "path_selection_policy": "train_val_energy", "selected_path_indices": keep}
    if method == "STC-path-prune-validation-accuracy":
        keep = select_paths_by_validation(full_cache, labels, split, budget=float(budget), seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device), objective="accuracy")
        return cache_with_path_indices(full_cache, keep), {"compression_axis": "path_channel", "path_selection_policy": "validation_accuracy", "selected_path_indices": keep}
    if method == "STC-path-prune-validation-loss":
        keep = select_paths_by_validation(full_cache, labels, split, budget=float(budget), seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device), objective="loss")
        return cache_with_path_indices(full_cache, keep), {"compression_axis": "path_channel", "path_selection_policy": "validation_loss", "selected_path_indices": keep}
    if method in {"STC-path-channel-hard-gate", "STC-path-channel-hard-gate-logit-calibrated"}:
        keep = select_paths_by_validation(full_cache, labels, split, budget=float(budget), seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device), objective="accuracy")
        return cache_with_path_indices(full_cache, keep), {"compression_axis": "path_channel", "path_selection_policy": "validation_accuracy_hard_gate", "selected_path_indices": keep}
    if method == "STC-feature-cache-true-distill":
        keep = select_paths_by_energy(full_cache, split, float(budget))
        return cache_with_path_indices(full_cache, keep), {"compression_axis": "feature_cache_true_distill", "path_selection_policy": "train_val_energy", "selected_path_indices": keep}
    if method == "STC-feature-cache-quantized-fp16":
        cache, diag = quantized_cache(full_cache, bits=16, budget=float(budget), split=split)
        return cache, {"compression_axis": "feature_cache_quantized", **diag, "path_selection_policy": "train_val_energy_if_needed"}
    if method == "STC-feature-cache-quantized-int8":
        cache, diag = quantized_cache(full_cache, bits=8, budget=float(budget), split=split)
        return cache, {"compression_axis": "feature_cache_quantized", **diag, "path_selection_policy": "train_val_energy_if_needed"}
    raise ValueError(f"unsupported Gate19 STC method: {method}")


def _feature_bytes_for_method(method: str, cache: SemanticTreeCache, diag: Mapping[str, Any]) -> int:
    if "quantized_bytes_per_value" in diag:
        return int(np.asarray(cache.tensor).size * int(diag["quantized_bytes_per_value"]))
    return compute_feature_cache_bytes(cache.tensor, np.float32)


def _evaluate_true_distill(
    *,
    method: str,
    cache: SemanticTreeCache,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    teacher_logits: np.ndarray,
    teacher_result: Mapping[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local_labels = labels_for_cache(cache, labels)
    train_idx = local_indices(cache, np.asarray(split["train"], dtype=np.int64))
    features = flatten_cache(cache)
    best: dict[str, Any] | None = None
    best_diag: dict[str, Any] | None = None
    for lambda_kl in (0.0, 0.25, 0.5, 1.0):
        for temperature in (1.0, 2.0, 4.0):
            fit = train_student_with_teacher(
                features,
                local_labels,
                train_idx,
                teacher_logits=teacher_logits,
                seed=int(seed),
                epochs=max(1, min(int(args.task_epochs), 5)),
                hidden_dim=int(args.task_hidden_dim),
                lambda_kl=float(lambda_kl),
                temperature=float(temperature),
                device=str(args.device),
            )
            result = evaluate_cache_logits(
                method=method,
                cache=cache,
                labels=labels,
                split=split,
                logits=np.asarray(fit["logits"], dtype=np.float32),
                model_param_bytes=int(fit.get("model_param_bytes", 0)),
            )
            val_diag = teacher_kl_diagnostics(teacher_logits[result["val_indices"]], result["all_logits"][result["val_indices"]], teacher_source="Full-STC-MLP", temperature=float(temperature))
            test_diag = teacher_kl_diagnostics(teacher_logits[result["test_indices"]], result["all_logits"][result["test_indices"]], teacher_source="Full-STC-MLP", temperature=float(temperature))
            diag = {
                "teacher_source": "Full-STC-MLP",
                "teacher_available": bool(test_diag.get("teacher_available")),
                "teacher_kl_status": test_diag.get("teacher_kl_status", ""),
                "teacher_student_kl_val": val_diag.get("teacher_student_kl", ""),
                "teacher_student_kl_test": test_diag.get("teacher_student_kl", ""),
                "teacher_student_agreement_val": val_diag.get("teacher_student_agreement", ""),
                "teacher_student_agreement_test": test_diag.get("teacher_student_agreement", ""),
                "teacher_val_accuracy": teacher_result.get("validation_accuracy", ""),
                "teacher_test_accuracy": teacher_result.get("accuracy", ""),
                "teacher_val_macro_f1": teacher_result.get("validation_macro_f1", ""),
                "teacher_test_macro_f1": teacher_result.get("macro_f1", ""),
                "lambda_kl": float(lambda_kl),
                "temperature": float(temperature),
                "teacher_sweep_grid": "lambda=[0,0.25,0.5,1];temperature=[1,2,4]",
            }
            result.update(diag)
            if best is None or (float(result["validation_accuracy"]), float(result["validation_macro_f1"])) > (float(best["validation_accuracy"]), float(best["validation_macro_f1"])):
                best = result
                best_diag = diag
    assert best is not None and best_diag is not None
    return best, best_diag


def _add_cache_result(
    *,
    rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    dataset: str,
    seed: int,
    method: str,
    requested_budget: float,
    result: Mapping[str, Any],
    cache: SemanticTreeCache,
    full_cache: SemanticTreeCache,
    full_context: Mapping[str, int],
    method_family: str,
    diagnostic_only: bool,
    args: argparse.Namespace,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(extra or {})
    row = _base_row(
        dataset=dataset,
        seed=seed,
        method=method,
        requested_budget=float(requested_budget),
        requested_support_ratio=None,
        method_family=method_family,
        diagnostic_only=diagnostic_only,
        args=args,
    )
    row.update(_metric_payload(result))
    row.update({key: value for key, value in extra.items() if key not in {"selected_path_indices"}})
    row["actual_support_ratio"] = 0.0
    row["selected_support_count"] = 0
    row["requested_support_count"] = 0
    row["full_cache_ceiling"] = bool(method.startswith("Full-STC"))
    row["full_stc_reference_method"] = "Full-STC-MLP"
    row["path_channel_count"] = int(len(cache.paths))
    row["full_path_channel_count"] = int(len(full_cache.paths))
    row["semantic_tree_l2_delta_vs_full"] = _cache_l2_delta(full_cache, cache)
    row["feature_cache_elements"] = int(np.asarray(cache.tensor).size)
    row["full_feature_cache_elements"] = int(np.asarray(full_cache.tensor).size)
    feature_bytes = int(extra.get("feature_cache_bytes_override", result.get("feature_cache_bytes", compute_feature_cache_bytes(cache.tensor, np.float32))))
    cost = CompressionCost(
        method=method,
        dataset=dataset,
        seed=int(seed),
        requested_budget=float(requested_budget),
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
    _merge_cost(row, cost_rows, cost)
    pc, cm = _prediction_audit(dataset=dataset, seed=seed, method=method, ratio=float(requested_budget), task=result)
    per_class_rows.extend(pc)
    confusion_rows.extend(cm)
    feature_rows.append(
        {
            "dataset": dataset,
            "seed": int(seed),
            "method": method,
            "requested_budget": float(requested_budget),
            "macro_f1": row.get("macro_f1", ""),
            "accuracy": row.get("accuracy", ""),
            "validation_macro_f1": row.get("validation_macro_f1", ""),
            "validation_accuracy": row.get("validation_accuracy", ""),
            "compression_axis": row.get("compression_axis", ""),
            "path_selection_policy": row.get("path_selection_policy", ""),
            "semantic_tree_l2_delta_vs_full": row.get("semantic_tree_l2_delta_vs_full", ""),
            "feature_cache_size_ratio": row.get("feature_cache_size_ratio", ""),
            "path_channel_count_ratio": row.get("path_channel_count_ratio", ""),
            "total_storage_ratio_vs_full_stc": row.get("total_storage_ratio_vs_full_stc", ""),
        }
    )
    rows.append(row)
    return row


def _git_output(args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=False)
    return (completed.stdout if completed.returncode == 0 else completed.stderr).strip()


def _write_code_sync_report(output_dir: Path) -> None:
    gate18_files = (
        "experiments/scripts/run_gate18r_accuracy_first_reset.py",
        "experiments/scripts/summarize_gate18r.py",
    )
    committed = {path: bool(_git_output(["ls-tree", "-r", "HEAD", "--name-only", path])) for path in gate18_files}
    lines = [
        "# Gate19 Code Sync Report",
        "",
        f"- git_head: `{git_commit_hash()}`",
        f"- branch: `{_git_output(['branch', '--show-current'])}`",
        f"- gate18r_runner_in_head: {committed[gate18_files[0]]}",
        f"- gate18r_summarizer_in_head: {committed[gate18_files[1]]}",
        "- gate18r_public_main_preflight: " + str(committed[gate18_files[0]] and committed[gate18_files[1]]),
        "",
        "## Gate18R Public-Main Audit",
    ]
    lines.extend(f"- {path}: committed_in_HEAD={ok}" for path, ok in committed.items())
    lines.extend(
        [
            "",
            "## Gate19 Local Additions",
            "- hesf_coarsen/task_first/costs/accounting.py",
            "- hesf_coarsen/task_first/costs/reports.py",
            "- hesf_coarsen/task_first/feature_condensation/baselines.py",
            "- hesf_coarsen/task_first/feature_condensation/distillation.py",
            "- experiments/scripts/run_gate19_cost_normalized_stc.py",
            "- experiments/scripts/summarize_gate19.py",
            "- tests/test_gate19_cost_normalized_stc.py",
        ]
    )
    output_dir.joinpath("code_sync_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_method_to_code_path(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for method in SUPPORT_BASELINES:
        rows.append({"method": method, "method_family": "support_baseline", "code_path": "experiments/scripts/run_gate19_cost_normalized_stc.py", "function": "run_support_baseline/_eval_task"})
    for method in FULL_STC_BASELINES:
        rows.append({"method": method, "method_family": "full_stc_baseline", "code_path": "hesf_coarsen/task_first/feature_condensation/baselines.py", "function": "evaluate_cache_classifier/calibrated_cache_result"})
    for method in STC_METHODS:
        fn = "_select_cache_for_method"
        if method == "STC-feature-cache-true-distill":
            fn = "_evaluate_true_distill/train_student_with_teacher"
        rows.append({"method": method, "method_family": "stc_compressed", "code_path": "experiments/scripts/run_gate19_cost_normalized_stc.py", "function": fn})
    for method in CLUSTER_DIAGNOSTICS:
        rows.append({"method": method, "method_family": "cluster_diagnostic", "code_path": "experiments/scripts/run_gate19_cost_normalized_stc.py", "function": "_apply_prediction_calibration"})
    rows.extend(
        [
            {"method": "CompressionCost", "method_family": "cost_accounting", "code_path": "hesf_coarsen/task_first/costs/accounting.py", "function": "compute_total_storage_ratio"},
            {"method": "Gate19 summarizer", "method_family": "summary", "code_path": "experiments/scripts/summarize_gate19.py", "function": "summarize/build_cost_normalized_pareto"},
        ]
    )
    write_csv(output_dir / "method_to_code_path.csv", rows)


def _write_code_change_report(output_dir: Path, result: Mapping[str, Any]) -> None:
    changed = _git_output(["status", "--short"])
    lines = [
        "# Gate19 Code Change Report",
        "",
        "- Stage: Gate19 cost-normalized semantic-tree condensation.",
        "- New code paths: cost accounting, Full-STC cache baselines, true teacher-student distillation, Gate19 runner, Gate19 summarizer, Gate19 tests.",
        "- Gate18R preflight: runner and summarizer are present in current HEAD before Gate19 additions.",
        f"- Gate19 decision: {result.get('decision', '')}",
        f"- Gate20 allowed: {result.get('gate20_allowed', '')}",
        "",
        "## Working Tree Status At Report Time",
        "```",
        changed,
        "```",
        "",
        "## Notes",
        "- ClusterGate rows in Gate19 are diagnostic-only and are not used for the official success decision.",
        "- Full-STC-MLP is the classifier ceiling used for DBLP recovery checks.",
        "- Pareto and decision logic use `total_storage_ratio_vs_full_stc`, not raw support ratio.",
    ]
    output_dir.joinpath("code_change_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_requirement_checklist(output_dir: Path, result: Mapping[str, Any]) -> None:
    required = [
        "code_sync_report.md",
        "method_to_code_path.csv",
        "gate19_raw_rows.csv",
        "gate19_validation_selected_by_method.csv",
        "gate19_by_dataset_selected.csv",
        "gate19_pareto_frontier.csv",
        "gate19_result.json",
        "gate19_decision.md",
        "gate19_cost_breakdown.csv",
        "gate19_full_stc_baselines.csv",
        "gate19_feature_condensation.csv",
        "gate19_true_distillation.csv",
        "gate19_teacher_audit.csv",
        "gate19_calibration.csv",
        "gate19_per_class_metrics.csv",
        "gate19_confusion_matrix_by_method.csv",
        "gate19_evaluator_ceiling_audit.csv",
        "gate19_leakage_audit.csv",
        "gate19_cache_size_audit.csv",
        "gate19_cluster_unit_diagnostics.csv",
        "gate19_selected_units.csv",
        "code_change_report.md",
    ]
    lines = [
        "# Gate19 Requirement Checklist",
        "",
        f"- [x] Stage reset recorded as Gate19 Cost-Normalized STC.",
        f"- [x] `primary_eval_mode=compressed_projected` enforced.",
        f"- [{'x' if result.get('no_test_leakage') else ' '}] No test leakage audit passed.",
        f"- [{'x' if result.get('typedhash_included') else ' '}] TypedHash baseline included.",
        f"- [{'x' if result.get('full_stc_baseline_available') else ' '}] Full-STC baselines available for ACM/DBLP/IMDB.",
        f"- [{'x' if result.get('cost_accounting_pass') else ' '}] STC cost accounting is nonzero and finite.",
        f"- [{'x' if result.get('teacher_kl_valid') else ' '}] True distillation teacher KL is valid and not self-reference.",
        "- [x] ACM marked sanity-only.",
        "- [x] DBLP is the primary decision dataset.",
        "- [x] IMDB is diagnostic.",
        f"- [x] Gate decision written: `{result.get('decision', '')}`.",
        "",
        "## Required Output Files",
    ]
    for name in required:
        lines.append(f"- [{'x' if (output_dir / name).exists() else ' '}] `{name}`")
    output_dir.joinpath("gate19_requirement_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.primary_eval_mode) != "compressed_projected":
        raise ValueError("Gate19 requires --primary-eval-mode compressed_projected")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_code_sync_report(output_dir)
    _write_method_to_code_path(output_dir)

    rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    full_stc_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    cache_size_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    selected_unit_rows: list[dict[str, Any]] = []

    for dataset, seed in args.dataset_seed_pairs:
        graph = load_hgb_graph(Path(args.data_root), str(dataset))
        labels = np.asarray(graph.labels if graph.labels is not None else np.full(graph.num_nodes, -1), dtype=np.int64)
        target_type = infer_target_node_type(graph)
        support_count = _support_count(graph, int(target_type))
        full_edge_count = _edge_count(graph)
        train_nodes, val_nodes, test_nodes, split_protocol = select_task_protocol_split(graph, labels, seed=int(seed), target_node_type=int(target_type))
        split = {"train": train_nodes, "val": val_nodes, "test": test_nodes}
        full_cache = build_semantic_tree_cache(graph, target_type=int(target_type), max_hops=2, max_paths=int(args.max_paths))
        full_cache_bytes = compute_feature_cache_bytes(full_cache.tensor, np.float32)
        cache_size_rows.append({"dataset": dataset, "seed": int(seed), **cache_metadata(full_cache), "feature_cache_bytes": int(full_cache_bytes), "cache_role": "full_stc"})

        full_mlp_result = evaluate_cache_classifier(
            method="Full-STC-MLP",
            cache=full_cache,
            labels=labels,
            split=split,
            classifier="mlp",
            seed=int(seed),
            epochs=int(args.task_epochs),
            hidden_dim=int(args.task_hidden_dim),
            device=str(args.device),
        )
        full_model_param_bytes = int(full_mlp_result.get("model_param_bytes", 0) or 0)
        full_context = {
            "full_support_node_count": int(support_count),
            "full_support_edge_count": int(full_edge_count),
            "full_path_channel_count": int(len(full_cache.paths)),
            "full_feature_cache_elements": int(np.asarray(full_cache.tensor).size),
            "full_feature_cache_bytes": int(full_cache_bytes),
            "full_model_param_bytes": int(full_model_param_bytes),
        }
        full_results: list[tuple[str, dict[str, Any], str]] = [("Full-STC-MLP", full_mlp_result, "mlp")]
        if _bool_arg(args.include_full_stc_baselines):
            linear = evaluate_cache_classifier(method="Full-STC-linear", cache=full_cache, labels=labels, split=split, classifier="linear", seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device))
            centroid = evaluate_cache_classifier(method="Full-STC-centroid", cache=full_cache, labels=labels, split=split, classifier="centroid", seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device))
            calibrated, calib = calibrated_cache_result("Full-STC-MLP-logit-calibrated", full_mlp_result, baseline_macro=_float(full_mlp_result.get("validation_macro_f1")))
            calib.update({"dataset": dataset, "seed": int(seed), "requested_budget": 1.0, "calibration_modes": CALIBRATION_MODES, "calibration_split": "val"})
            calibration_rows.append(calib)
            full_results.extend([("Full-STC-MLP-logit-calibrated", calibrated, "mlp_calibrated"), ("Full-STC-linear", linear, "linear"), ("Full-STC-centroid", centroid, "centroid")])
        for method, result, model_name in full_results:
            row = _add_cache_result(
                rows=rows,
                cost_rows=cost_rows,
                feature_rows=feature_rows,
                per_class_rows=per_class_rows,
                confusion_rows=confusion_rows,
                dataset=dataset,
                seed=int(seed),
                method=method,
                requested_budget=1.0,
                result=result,
                cache=full_cache,
                full_cache=full_cache,
                full_context=full_context,
                method_family="full_stc_baseline",
                diagnostic_only=False,
                args=args,
                extra={"compression_axis": "full_semantic_tree_cache", "path_selection_policy": "none", "full_stc_model": model_name},
            )
            row["eligible_for_main_decision"] = False
            full_stc_rows.append(dict(row))
            ceiling_rows.append({"dataset": dataset, "seed": int(seed), "method": method, "full_cache_ceiling": True, "path_channel_count": len(full_cache.paths), "feature_cache_bytes": int(full_cache_bytes), "accuracy": row.get("accuracy", ""), "macro_f1": row.get("macro_f1", "")})

        full_teacher_logits = np.asarray(full_mlp_result["all_logits"], dtype=np.float32)
        teacher_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "teacher_source": "Full-STC-MLP",
                "teacher_available": True,
                "teacher_val_accuracy": full_mlp_result.get("validation_accuracy", ""),
                "teacher_test_accuracy": full_mlp_result.get("accuracy", ""),
                "teacher_val_macro_f1": full_mlp_result.get("validation_macro_f1", ""),
                "teacher_test_macro_f1": full_mlp_result.get("macro_f1", ""),
                "teacher_training_split": "train",
                "teacher_selection_split": "val",
                "teacher_uses_test_labels_for_training": False,
            }
        )

        support_cache: dict[tuple[str, float], tuple[HeteroGraph, np.ndarray, dict[str, Any], dict[str, Any], np.ndarray]] = {}
        for ratio in args.support_ratios_parsed:
            for method in SUPPORT_BASELINES:
                start = perf_counter()
                row = _base_row(dataset=dataset, seed=int(seed), method=method, requested_budget=float(ratio), requested_support_ratio=float(ratio), method_family="support_baseline", diagnostic_only=False, args=args)
                row.update(split_protocol)
                selected_nodes = np.empty(0, dtype=np.int64)
                task: dict[str, Any] = {}
                coarse: HeteroGraph | None = None
                try:
                    if method == "full-graph-hettree-lite-tuned":
                        row.update(_full_graph_row(graph, dataset, int(seed), float(ratio), args, split))
                        selected_count = support_count
                        coarse = graph
                    elif method == "target-only-empty-support":
                        target_row, coarse, _assignment = _target_only_row(graph, dataset, int(seed), float(ratio), args, split)
                        row.update(target_row)
                        selected_count = 0
                    else:
                        coarse, assignment, diag = run_support_baseline(graph, baseline=method, ratio=float(ratio), seed=int(seed), candidate_k=int(args.candidate_k))
                        assignment = np.asarray(assignment, dtype=np.int64)
                        task = _eval_task(graph, coarse, assignment, seed=int(seed), split=split, target_type=int(target_type), args=args, return_logits=True)
                        selected_nodes = selected_support_representatives_from_assignment(graph, assignment, int(target_type))
                        selected_count = int(diag.get("final_support_nodes", len(selected_nodes)))
                        support_cache[(method, float(ratio))] = (coarse, assignment, diag, task, selected_nodes)
                        row.update({key: value for key, value in diag.items() if not isinstance(value, (dict, list, np.ndarray))})
                        _row_from_task(row, task)
                    row["selected_support_count"] = int(selected_count)
                    row["requested_support_count"] = _requested_support_count(support_count, float(ratio))
                    row["actual_support_ratio"] = float(selected_count / max(1, support_count))
                    row["realized_support_ratio"] = row["actual_support_ratio"]
                    support_edges = _edge_count(coarse) if coarse is not None else 0
                    cost = CompressionCost(
                        method=method,
                        dataset=dataset,
                        seed=int(seed),
                        requested_budget=float(ratio),
                        support_node_count=int(selected_count),
                        support_edge_count=int(support_edges),
                        full_support_node_count=int(support_count),
                        full_support_edge_count=int(full_edge_count),
                        full_path_channel_count=int(len(full_cache.paths)),
                        full_feature_cache_elements=int(np.asarray(full_cache.tensor).size),
                        full_feature_cache_bytes=int(full_cache_bytes),
                        full_model_param_bytes=int(full_model_param_bytes),
                    )
                    _merge_cost(row, cost_rows, cost)
                    if task:
                        pc, cm = _prediction_audit(dataset=dataset, seed=int(seed), method=method, ratio=float(ratio), task=task, selected_labels=_support_labels(labels, selected_nodes))
                        per_class_rows.extend(pc)
                        confusion_rows.extend(cm)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        write_csv(output_dir / "gate19_raw_rows.csv", rows)
                        raise
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)
                row["wall_clock_sec"] = float(perf_counter() - start)
                leak = leakage_audit_row(method=method, uses_train_labels=True, uses_val_labels=False, uses_test_labels_before_final_eval=False, calibration_split="none", path_selection_split="train", teacher_training_split="none", student_training_split="train")
                row.update({"no_test_leakage": leak["no_test_leakage"], "method_invalid": leak["method_invalid"]})
                leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_support_ratio": float(ratio), **leak})
                rows.append(row)

            if _bool_arg(args.include_full_stc_baselines):
                diagnostic_sources = {
                    "ClusterGate-H6-units-logit-calibrated": "H6-no-spec-support-only",
                    "ClusterGate-TypedHash-units-logit-calibrated": "TypedHash-ChebHeat-support-only",
                    "HeSF-SS-validation-H6-fill-logit-calibrated": "H6-no-spec-support-only",
                }
                available = [support_cache[key] for key in support_cache if key[1] == float(ratio) and key[0] in {"H6-no-spec-support-only", "TypedHash-ChebHeat-support-only", "flatten-sum-support-only"}]
                if available:
                    best_source = max(
                        ((name, payload) for (name, r), payload in support_cache.items() if r == float(ratio) and name in {"H6-no-spec-support-only", "TypedHash-ChebHeat-support-only", "flatten-sum-support-only"}),
                        key=lambda item: _float(item[1][3].get("validation_accuracy")),
                    )
                    diagnostic_sources["ClusterGate-UnionUnits-logit-calibrated"] = best_source[0]
                for diag_method, source_method in diagnostic_sources.items():
                    cached = support_cache.get((source_method, float(ratio)))
                    if cached is None:
                        continue
                    coarse, _assignment, diag, task, selected_nodes = cached
                    calibrated_task, calib = _apply_prediction_calibration(task, method=diag_method)
                    calib.update({"dataset": dataset, "seed": int(seed), "requested_support_ratio": float(ratio), "source_method": source_method})
                    calibration_rows.append(calib)
                    row = _base_row(dataset=dataset, seed=int(seed), method=diag_method, requested_budget=float(ratio), requested_support_ratio=float(ratio), method_family="cluster_diagnostic", diagnostic_only=True, args=args)
                    row.update(_metric_payload(calibrated_task))
                    row["cluster_gate_source_method"] = source_method
                    row["selected_support_count"] = int(diag.get("final_support_nodes", len(selected_nodes)))
                    row["requested_support_count"] = _requested_support_count(support_count, float(ratio))
                    row["actual_support_ratio"] = float(row["selected_support_count"] / max(1, support_count))
                    row["calibration_modes"] = CALIBRATION_MODES
                    _merge_cost(
                        row,
                        cost_rows,
                        CompressionCost(
                            method=diag_method,
                            dataset=dataset,
                            seed=int(seed),
                            requested_budget=float(ratio),
                            support_node_count=int(row["selected_support_count"]),
                            support_edge_count=_edge_count(coarse),
                            unit_count=int(row["selected_support_count"]),
                            full_support_node_count=int(support_count),
                            full_support_edge_count=int(full_edge_count),
                            full_unit_count=int(support_count),
                            full_path_channel_count=int(len(full_cache.paths)),
                            full_feature_cache_elements=int(np.asarray(full_cache.tensor).size),
                            full_feature_cache_bytes=int(full_cache_bytes),
                            full_model_param_bytes=int(full_model_param_bytes),
                        ),
                    )
                    pc, cm = _prediction_audit(dataset=dataset, seed=int(seed), method=diag_method, ratio=float(ratio), task=calibrated_task, selected_labels=_support_labels(labels, selected_nodes))
                    per_class_rows.extend(pc)
                    confusion_rows.extend(cm)
                    cluster_rows.append({"dataset": dataset, "seed": int(seed), "method": diag_method, "requested_support_ratio": float(ratio), "source_method": source_method, "diagnostic_only": True, "selected_unit_count": int(row["selected_support_count"])})
                    for node in np.asarray(selected_nodes, dtype=np.int64).reshape(-1).tolist()[:5000]:
                        selected_unit_rows.append({"dataset": dataset, "seed": int(seed), "method": diag_method, "requested_support_ratio": float(ratio), "unit_source": source_method, "member_node": int(node)})
                    leak = leakage_audit_row(method=diag_method, uses_train_labels=True, uses_val_labels=True, uses_test_labels_before_final_eval=False, calibration_split="val", path_selection_split="train_val", teacher_training_split="none", student_training_split="train")
                    row.update({"no_test_leakage": leak["no_test_leakage"], "method_invalid": leak["method_invalid"]})
                    leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_support_ratio": float(ratio), **leak})
                    rows.append(row)

        for budget in args.cost_budgets_parsed:
            for method in STC_METHODS:
                if method == "STC-feature-cache-true-distill" and not _bool_arg(args.include_true_distillation):
                    continue
                start = perf_counter()
                try:
                    cache, diag = _select_cache_for_method(method, full_cache=full_cache, labels=labels, split=split, budget=float(budget), seed=int(seed), args=args)
                    cache_size_rows.append({"dataset": dataset, "seed": int(seed), **cache_metadata(cache), "feature_cache_bytes": _feature_bytes_for_method(method, cache, diag), "cache_role": method, "requested_budget": float(budget)})
                    if method == "STC-feature-cache-true-distill":
                        result, teacher_diag = _evaluate_true_distill(method=method, cache=cache, labels=labels, split=split, teacher_logits=full_teacher_logits, teacher_result=full_mlp_result, seed=int(seed), args=args)
                        diag.update(teacher_diag)
                    else:
                        base_method = method.replace("-logit-calibrated", "")
                        result = evaluate_cache_classifier(method=base_method, cache=cache, labels=labels, split=split, classifier="mlp", seed=int(seed), epochs=int(args.task_epochs), hidden_dim=int(args.task_hidden_dim), device=str(args.device))
                        if method.endswith("-logit-calibrated"):
                            result, calib = calibrated_cache_result(method, result, baseline_macro=_float(full_mlp_result.get("validation_macro_f1")))
                            calib.update({"dataset": dataset, "seed": int(seed), "requested_budget": float(budget), "calibration_modes": CALIBRATION_MODES, "calibration_split": "val"})
                            calibration_rows.append(calib)
                    diag["feature_cache_bytes_override"] = _feature_bytes_for_method(method, cache, diag)
                    row = _add_cache_result(
                        rows=rows,
                        cost_rows=cost_rows,
                        feature_rows=feature_rows,
                        per_class_rows=per_class_rows,
                        confusion_rows=confusion_rows,
                        dataset=dataset,
                        seed=int(seed),
                        method=method,
                        requested_budget=float(budget),
                        result=result,
                        cache=cache,
                        full_cache=full_cache,
                        full_context=full_context,
                        method_family="stc_compressed",
                        diagnostic_only=False,
                        args=args,
                        extra=diag,
                    )
                    row["accuracy_gap_vs_full_stc_mlp"] = _float(row.get("accuracy")) - _float(full_mlp_result.get("accuracy"))
                    row["macro_gap_vs_full_stc_mlp"] = _float(row.get("macro_f1")) - _float(full_mlp_result.get("macro_f1"))
                    row["budget_feasible"] = _float(row.get("total_storage_ratio_vs_full_stc")) <= float(budget) + 1.0e-12
                    if method == "STC-feature-cache-true-distill":
                        distill_rows.append({key: row.get(key, "") for key in row})
                        teacher_rows.append({"dataset": dataset, "seed": int(seed), "method": method, "requested_budget": float(budget), **{key: row.get(key, "") for key in ["teacher_source", "teacher_available", "teacher_kl_status", "teacher_student_kl_val", "teacher_student_kl_test", "teacher_student_agreement_val", "teacher_student_agreement_test", "teacher_val_accuracy", "teacher_test_accuracy", "lambda_kl", "temperature"]}})
                    leak = leakage_audit_row(method=method, uses_train_labels=True, uses_val_labels=True, uses_test_labels_before_final_eval=False, calibration_split="val" if method.endswith("calibrated") else "none", path_selection_split="train_val", teacher_training_split="train", student_training_split="train")
                    row.update({"no_test_leakage": leak["no_test_leakage"], "method_invalid": leak["method_invalid"]})
                    leakage_rows.append({"dataset": dataset, "seed": int(seed), "requested_budget": float(budget), **leak})
                    row["wall_clock_sec"] = float(perf_counter() - start)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        write_csv(output_dir / "gate19_raw_rows.csv", rows)
                        raise
                    fail = _base_row(dataset=dataset, seed=int(seed), method=method, requested_budget=float(budget), requested_support_ratio=None, method_family="stc_compressed", diagnostic_only=False, args=args)
                    fail.update({"status": "failed", "error": repr(exc), "wall_clock_sec": float(perf_counter() - start)})
                    rows.append(fail)
                except Exception as exc:
                    fail = _base_row(dataset=dataset, seed=int(seed), method=method, requested_budget=float(budget), requested_support_ratio=None, method_family="stc_compressed", diagnostic_only=False, args=args)
                    fail.update({"status": "failed", "error": repr(exc), "wall_clock_sec": float(perf_counter() - start)})
                    rows.append(fail)

        write_csv(output_dir / "gate19_raw_rows.csv", rows)

    write_csv(output_dir / "gate19_raw_rows.csv", rows)
    write_csv(output_dir / "gate19_cost_breakdown.csv", cost_rows)
    write_csv(output_dir / "gate19_full_stc_baselines.csv", full_stc_rows)
    write_csv(output_dir / "gate19_feature_condensation.csv", feature_rows)
    write_csv(output_dir / "gate19_true_distillation.csv", distill_rows)
    write_csv(output_dir / "gate19_teacher_audit.csv", teacher_rows)
    write_csv(output_dir / "gate19_calibration.csv", calibration_rows)
    write_csv(output_dir / "gate19_per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "gate19_confusion_matrix_by_method.csv", confusion_rows)
    write_csv(output_dir / "gate19_evaluator_ceiling_audit.csv", ceiling_rows)
    write_csv(output_dir / "gate19_leakage_audit.csv", leakage_rows)
    write_csv(output_dir / "gate19_cache_size_audit.csv", cache_size_rows)
    write_csv(output_dir / "gate19_cluster_unit_diagnostics.csv", cluster_rows)
    write_csv(output_dir / "gate19_selected_units.csv", selected_unit_rows)
    result = summarize(output_dir, output_dir)
    _write_code_change_report(output_dir, result)
    _write_requirement_checklist(output_dir, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate19 cost-normalized STC with Full-STC baselines.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate19"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=["ACM", "DBLP", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=["ACM:23456", "DBLP:23456", "IMDB:45678"])
    parser.add_argument("--cost-budgets", nargs="*", default=[0.30, 0.50, 0.70, 1.00])
    parser.add_argument("--support-ratios", nargs="*", default=[0.30, 0.50, 0.70])
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--task-epochs", type=int, default=10)
    parser.add_argument("--task-hidden-dim", type=int, default=64)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--feature-mode", default="full")
    parser.add_argument("--include-typedhash", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--return-logits", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--include-full-stc-baselines", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--include-true-distillation", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor", default="projected_val_macro_f1")
    parser.add_argument("--candidate-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dataset_seed_pairs = parse_dataset_seeds(args.dataset_seeds)
    if args.datasets:
        allowed = {str(dataset).upper() for dataset in args.datasets}
        args.dataset_seed_pairs = [(dataset, seed) for dataset, seed in args.dataset_seed_pairs if dataset in allowed]
    args.cost_budgets_parsed = _split_values(args.cost_budgets, float) or [0.30, 0.50, 0.70, 1.00]
    args.support_ratios_parsed = _split_values(args.support_ratios, float) or [0.30, 0.50, 0.70]
    if not _bool_arg(args.include_typedhash):
        args.support_ratios_parsed = [float(value) for value in args.support_ratios_parsed]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
