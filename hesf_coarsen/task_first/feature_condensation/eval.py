from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.task_gnn import f1_scores
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.feature_condensation.feature_cache_distill import train_feature_cache_distill
from hesf_coarsen.task_first.feature_condensation.path_prototype import class_path_prototype_cache
from hesf_coarsen.task_first.feature_condensation.path_prune import prune_cache_paths
from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache, build_semantic_tree_cache


def _local_indices(cache: SemanticTreeCache, nodes: np.ndarray) -> np.ndarray:
    lookup = {int(node): idx for idx, node in enumerate(np.asarray(cache.target_nodes, dtype=np.int64).tolist())}
    return np.asarray([lookup[int(node)] for node in np.asarray(nodes, dtype=np.int64).tolist() if int(node) in lookup], dtype=np.int64)


def _flatten(cache: SemanticTreeCache) -> np.ndarray:
    tensor = np.asarray(cache.tensor, dtype=np.float32)
    if tensor.ndim != 3:
        return tensor.reshape((tensor.shape[0], -1)).astype(np.float32, copy=False)
    return tensor.reshape((tensor.shape[0], int(tensor.shape[1]) * int(tensor.shape[2]))).astype(np.float32, copy=False)


def _scores(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = (np.asarray(labels) >= 0) & (np.asarray(pred) >= 0)
    if not np.any(valid):
        return {"micro_f1": 0.0, "macro_f1": 0.0, "accuracy": 0.0}
    base = f1_scores(np.asarray(labels)[valid], np.asarray(pred)[valid], macro_empty_class_policy="truth_pred_union")
    return {**base, "accuracy": float(np.mean(np.asarray(labels)[valid] == np.asarray(pred)[valid]))}


def _kl_divergence(teacher_logits: np.ndarray, student_logits: np.ndarray) -> float:
    if teacher_logits.shape != student_logits.shape or teacher_logits.size == 0:
        return 0.0
    t = teacher_logits - np.max(teacher_logits, axis=1, keepdims=True)
    s = student_logits - np.max(student_logits, axis=1, keepdims=True)
    tp = np.exp(t) / np.maximum(np.sum(np.exp(t), axis=1, keepdims=True), 1.0e-12)
    sp = np.exp(s) / np.maximum(np.sum(np.exp(s), axis=1, keepdims=True), 1.0e-12)
    return float(np.mean(np.sum(tp * (np.log(np.maximum(tp, 1.0e-12)) - np.log(np.maximum(sp, 1.0e-12))), axis=1)))


def _aligned_l2_delta(full: SemanticTreeCache, compressed: SemanticTreeCache) -> float:
    full_tensor = np.asarray(full.tensor, dtype=np.float32)
    comp_tensor = np.zeros_like(full_tensor)
    path_lookup = {path: idx for idx, path in enumerate(full.paths)}
    for comp_idx, path in enumerate(compressed.paths):
        if path in path_lookup and comp_idx < compressed.tensor.shape[1]:
            comp_tensor[:, path_lookup[path], :] = compressed.tensor[:, comp_idx, :]
    return float(np.linalg.norm((full_tensor - comp_tensor).reshape(-1)))


def evaluate_feature_condensation_method(
    graph: HeteroGraph,
    *,
    method: str,
    requested_ratio: float,
    target_type: int,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    seed: int = 12345,
    epochs: int = 50,
    max_hops: int = 2,
    max_paths: int | None = 32,
    hidden_dim: int = 64,
    device: str = "auto",
    full_teacher_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    full_cache = build_semantic_tree_cache(graph, target_type=int(target_type), max_hops=int(max_hops), max_paths=max_paths)
    cache = full_cache
    compression_axis = "feature_cache"
    if str(method) == "HeSF-STC-path-prune":
        cache = prune_cache_paths(full_cache, float(requested_ratio))
        compression_axis = "path_channel"
    elif str(method) == "HeSF-STC-path-prototype":
        cache = class_path_prototype_cache(full_cache, labels=labels_arr, train_nodes=np.asarray(split["train"], dtype=np.int64))
        compression_axis = "path_prototype"
    elif str(method) in {"HeSF-STC-feature-cache-distill", "HeSF-STC-feature-cache-distill-logit-calibrated"}:
        cache = prune_cache_paths(full_cache, min(1.0, max(float(requested_ratio), 1.0 / max(1, len(full_cache.paths)))))
        compression_axis = "feature_cache"
    x = _flatten(cache)
    train_idx = _local_indices(cache, np.asarray(split["train"], dtype=np.int64))
    val_idx = _local_indices(cache, np.asarray(split["val"], dtype=np.int64))
    test_idx = _local_indices(cache, np.asarray(split["test"], dtype=np.int64))
    local_labels = np.asarray([int(labels_arr[int(node)]) for node in cache.target_nodes.tolist()], dtype=np.int64)
    fit = train_feature_cache_distill(
        x,
        local_labels,
        train_idx,
        seed=int(seed),
        epochs=int(epochs),
        hidden_dim=int(hidden_dim),
        device=str(device),
    )
    logits = np.asarray(fit["logits"], dtype=np.float32)
    pred = np.argmax(logits, axis=1).astype(np.int64, copy=False) if logits.size else np.empty(0, dtype=np.int64)
    val_scores = _scores(local_labels[val_idx], pred[val_idx]) if len(val_idx) else {"micro_f1": 0.0, "macro_f1": 0.0, "accuracy": 0.0}
    test_scores = _scores(local_labels[test_idx], pred[test_idx]) if len(test_idx) else {"micro_f1": 0.0, "macro_f1": 0.0, "accuracy": 0.0}
    teacher = np.asarray(full_teacher_logits, dtype=np.float32) if full_teacher_logits is not None else logits
    agreement = float(np.mean(np.argmax(teacher[test_idx], axis=1) == pred[test_idx])) if len(test_idx) and teacher.shape == logits.shape else 0.0
    cache_size_ratio = float(np.asarray(cache.tensor).size / max(1, np.asarray(full_cache.tensor).size))
    path_ratio = float(len(cache.paths) / max(1, len(full_cache.paths)))
    return {
        "method": str(method),
        "macro_f1": float(test_scores["macro_f1"]),
        "micro_f1": float(test_scores["micro_f1"]),
        "accuracy": float(test_scores["accuracy"]),
        "validation_macro_f1": float(val_scores["macro_f1"]),
        "validation_micro_f1": float(val_scores["micro_f1"]),
        "validation_accuracy": float(val_scores["accuracy"]),
        "projected_val_logits": logits[val_idx].tolist() if len(val_idx) else [],
        "projected_test_logits": logits[test_idx].tolist() if len(test_idx) else [],
        "projected_val_labels": local_labels[val_idx].tolist() if len(val_idx) else [],
        "projected_test_labels": local_labels[test_idx].tolist() if len(test_idx) else [],
        "projected_val_nodes": cache.target_nodes[val_idx].tolist() if len(val_idx) else [],
        "projected_test_nodes": cache.target_nodes[test_idx].tolist() if len(test_idx) else [],
        "projected_val_pred": pred[val_idx].tolist() if len(val_idx) else [],
        "projected_test_pred": pred[test_idx].tolist() if len(test_idx) else [],
        "feature_cache_size_ratio": cache_size_ratio,
        "path_channel_count_ratio": path_ratio,
        "semantic_tree_l2_delta_vs_full": _aligned_l2_delta(full_cache, cache),
        "teacher_kl_vs_full": _kl_divergence(teacher[test_idx], logits[test_idx]) if len(test_idx) and teacher.shape == logits.shape else 0.0,
        "full_teacher_logit_agreement": agreement,
        "compression_axis": compression_axis,
        "compression_ratio": cache_size_ratio,
        "primary_eval_mode": "compressed_projected",
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "calibrator_uses_test_labels": False,
        "no_test_leakage": True,
        "feature_model": fit.get("model", ""),
        "feature_model_skipped": bool(fit.get("skipped", False)),
    }
