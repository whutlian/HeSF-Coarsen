from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from hesf_coarsen.eval.calibration import apply_calibrator, fit_macro_constrained_accuracy_calibrator
from hesf_coarsen.eval.task_gnn import f1_scores
from hesf_coarsen.task_first.costs.accounting import compute_feature_cache_bytes, count_model_parameters_bytes
from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache


def local_indices(cache: SemanticTreeCache, nodes: np.ndarray) -> np.ndarray:
    lookup = {int(node): idx for idx, node in enumerate(np.asarray(cache.target_nodes, dtype=np.int64).tolist())}
    return np.asarray([lookup[int(node)] for node in np.asarray(nodes, dtype=np.int64).reshape(-1).tolist() if int(node) in lookup], dtype=np.int64)


def flatten_cache(cache: SemanticTreeCache) -> np.ndarray:
    tensor = np.asarray(cache.tensor, dtype=np.float32)
    return tensor.reshape((int(tensor.shape[0]), -1)).astype(np.float32, copy=False)


def labels_for_cache(cache: SemanticTreeCache, labels: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    return np.asarray([int(labels_arr[int(node)]) for node in np.asarray(cache.target_nodes, dtype=np.int64).tolist()], dtype=np.int64)


def classification_scores(labels: np.ndarray, logits: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    local_idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if len(local_idx) == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "accuracy": 0.0, "pred": []}
    y = np.asarray(labels, dtype=np.int64).reshape(-1)[local_idx]
    pred = np.argmax(np.asarray(logits, dtype=np.float32)[local_idx], axis=1).astype(np.int64, copy=False)
    valid = (y >= 0) & (pred >= 0)
    if not np.any(valid):
        return {"micro_f1": 0.0, "macro_f1": 0.0, "accuracy": 0.0, "pred": pred.tolist()}
    scores = f1_scores(y[valid], pred[valid], macro_empty_class_policy="truth_pred_union")
    return {**scores, "accuracy": float(np.mean(y[valid] == pred[valid])), "pred": pred.tolist()}


def cross_entropy_on_indices(labels: np.ndarray, logits: np.ndarray, idx: np.ndarray) -> float:
    local_idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if len(local_idx) == 0:
        return 0.0
    y = np.asarray(labels, dtype=np.int64).reshape(-1)[local_idx]
    raw = np.asarray(logits, dtype=np.float32)[local_idx]
    valid = y >= 0
    if not np.any(valid):
        return 0.0
    raw = raw[valid]
    y = y[valid]
    shifted = raw - np.max(raw, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.maximum(np.sum(np.exp(shifted), axis=1, keepdims=True), 1.0e-12))
    return float(-np.mean(log_probs[np.arange(len(y)), y]))


def _centroid_logits(x: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = np.zeros((int(num_classes), int(x.shape[1])), dtype=np.float32)
    counts = np.zeros(int(num_classes), dtype=np.float32)
    for idx in np.asarray(train_idx, dtype=np.int64).reshape(-1).tolist():
        cls = int(labels[int(idx)])
        if cls >= 0:
            centroids[cls] += x[int(idx)]
            counts[cls] += 1.0
    for cls in range(int(num_classes)):
        if counts[cls] > 0:
            centroids[cls] /= counts[cls]
    logits = np.empty((int(x.shape[0]), int(num_classes)), dtype=np.float32)
    for cls in range(int(num_classes)):
        diff = x - centroids[cls].reshape(1, -1)
        logits[:, cls] = -np.sum(diff * diff, axis=1)
    return logits


def train_cache_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    *,
    classifier: str = "mlp",
    seed: int = 12345,
    epochs: int = 50,
    hidden_dim: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    train = np.asarray([int(idx) for idx in np.asarray(train_idx, dtype=np.int64).reshape(-1) if int(labels_arr[int(idx)]) >= 0], dtype=np.int64)
    num_classes = int(labels_arr[labels_arr >= 0].max(initial=0)) + 1
    if len(train) == 0:
        return {"logits": np.zeros((int(x.shape[0]), int(num_classes)), dtype=np.float32), "model": f"{classifier}_empty", "model_param_bytes": 0}
    if str(classifier) == "centroid":
        return {"logits": _centroid_logits(x, labels_arr, train, num_classes), "model": "full_stc_centroid", "model_param_bytes": 0}
    try:
        import torch
        from torch import nn
    except Exception:
        return {"logits": _centroid_logits(x, labels_arr, train, num_classes), "model": f"{classifier}_centroid_fallback", "model_param_bytes": 0}
    if str(device) == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = str(device)
    dev = torch.device(device_name)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if str(classifier) == "linear":
        model = nn.Linear(int(x.shape[1]), int(num_classes)).to(dev)
    else:
        model = nn.Sequential(
            nn.Linear(int(x.shape[1]), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(num_classes)),
        ).to(dev)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(labels_arr, dtype=torch.long, device=dev)
    idx_t = torch.as_tensor(train, dtype=torch.long, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(max(1, int(epochs))):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits_t = model(x_t)
        loss = loss_fn(logits_t[idx_t], y_t[idx_t])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_t).detach().cpu().numpy().astype(np.float32, copy=False)
    return {"logits": logits, "model": f"full_stc_{classifier}", "device": device_name, "model_param_bytes": count_model_parameters_bytes(model)}


def evaluate_cache_logits(
    *,
    method: str,
    cache: SemanticTreeCache,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    logits: np.ndarray,
    model_param_bytes: int = 0,
) -> dict[str, Any]:
    local_labels = labels_for_cache(cache, labels)
    train_idx = local_indices(cache, np.asarray(split["train"], dtype=np.int64))
    val_idx = local_indices(cache, np.asarray(split["val"], dtype=np.int64))
    test_idx = local_indices(cache, np.asarray(split["test"], dtype=np.int64))
    val_scores = classification_scores(local_labels, logits, val_idx)
    test_scores = classification_scores(local_labels, logits, test_idx)
    return {
        "method": str(method),
        "macro_f1": float(test_scores["macro_f1"]),
        "micro_f1": float(test_scores["micro_f1"]),
        "accuracy": float(test_scores["accuracy"]),
        "validation_macro_f1": float(val_scores["macro_f1"]),
        "validation_micro_f1": float(val_scores["micro_f1"]),
        "validation_accuracy": float(val_scores["accuracy"]),
        "projected_val_logits": np.asarray(logits, dtype=np.float32)[val_idx].tolist(),
        "projected_test_logits": np.asarray(logits, dtype=np.float32)[test_idx].tolist(),
        "projected_val_labels": local_labels[val_idx].tolist(),
        "projected_test_labels": local_labels[test_idx].tolist(),
        "projected_val_nodes": np.asarray(cache.target_nodes, dtype=np.int64)[val_idx].tolist(),
        "projected_test_nodes": np.asarray(cache.target_nodes, dtype=np.int64)[test_idx].tolist(),
        "projected_val_pred": val_scores["pred"],
        "projected_test_pred": test_scores["pred"],
        "all_logits": np.asarray(logits, dtype=np.float32),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "local_labels": local_labels,
        "feature_cache_bytes": compute_feature_cache_bytes(cache.tensor, np.asarray(cache.tensor).dtype),
        "feature_cache_elements": int(np.asarray(cache.tensor).size),
        "path_channel_count": int(len(cache.paths)),
        "model_param_bytes": int(model_param_bytes),
        "primary_eval_mode": "compressed_projected",
        "no_test_leakage": True,
        "selector_uses_test_labels": False,
        "teacher_uses_test_labels_for_training": False,
        "calibrator_uses_test_labels": False,
    }


def evaluate_cache_classifier(
    *,
    method: str,
    cache: SemanticTreeCache,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    classifier: str = "mlp",
    seed: int = 12345,
    epochs: int = 50,
    hidden_dim: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    x = flatten_cache(cache)
    local_labels = labels_for_cache(cache, labels)
    train_idx = local_indices(cache, np.asarray(split["train"], dtype=np.int64))
    fit = train_cache_classifier(x, local_labels, train_idx, classifier=classifier, seed=seed, epochs=epochs, hidden_dim=hidden_dim, device=device)
    out = evaluate_cache_logits(method=method, cache=cache, labels=labels, split=split, logits=np.asarray(fit["logits"], dtype=np.float32), model_param_bytes=int(fit.get("model_param_bytes", 0)))
    out["feature_model"] = fit.get("model", "")
    out["feature_model_skipped"] = bool(fit.get("skipped", False))
    return out


def calibrated_cache_result(method: str, result: Mapping[str, Any], *, baseline_macro: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = fit_macro_constrained_accuracy_calibrator(
        result.get("projected_val_logits", []),
        result.get("projected_val_labels", []),
        baseline_macro=baseline_macro,
        macro_epsilon=0.01 if baseline_macro is None else 0.02,
        temperatures=(0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
        class_bias_values=(-0.5, -0.25, 0.0, 0.25, 0.5),
    )
    val_logits = apply_calibrator(result.get("projected_val_logits", []), fit)
    test_logits = apply_calibrator(result.get("projected_test_logits", []), fit)
    out = dict(result)
    val_labels = np.asarray(result.get("projected_val_labels", []), dtype=np.int64)
    test_labels = np.asarray(result.get("projected_test_labels", []), dtype=np.int64)
    val_pred = np.argmax(val_logits, axis=1).astype(np.int64, copy=False) if val_logits.size else np.empty(0, dtype=np.int64)
    test_pred = np.argmax(test_logits, axis=1).astype(np.int64, copy=False) if test_logits.size else np.empty(0, dtype=np.int64)
    val_scores = f1_scores(val_labels, val_pred, macro_empty_class_policy="truth_pred_union") if len(val_labels) else {"micro_f1": 0.0, "macro_f1": 0.0}
    test_scores = f1_scores(test_labels, test_pred, macro_empty_class_policy="truth_pred_union") if len(test_labels) else {"micro_f1": 0.0, "macro_f1": 0.0}
    out.update(
        {
            "method": str(method),
            "uncalibrated_macro_f1": float(result.get("macro_f1", 0.0) or 0.0),
            "uncalibrated_accuracy": float(result.get("accuracy", 0.0) or 0.0),
            "uncalibrated_validation_macro_f1": float(result.get("validation_macro_f1", 0.0) or 0.0),
            "uncalibrated_validation_accuracy": float(result.get("validation_accuracy", 0.0) or 0.0),
            "macro_f1": float(test_scores["macro_f1"]),
            "micro_f1": float(test_scores["micro_f1"]),
            "accuracy": float(np.mean(test_labels == test_pred)) if len(test_labels) else 0.0,
            "validation_macro_f1": float(val_scores["macro_f1"]),
            "validation_micro_f1": float(val_scores["micro_f1"]),
            "validation_accuracy": float(np.mean(val_labels == val_pred)) if len(val_labels) else 0.0,
            "projected_val_pred": val_pred.tolist(),
            "projected_test_pred": test_pred.tolist(),
            "calibrator_uses_test_labels": False,
        }
    )
    return out, {
        "method": str(method),
        "calibration_temperature": float(fit.get("temperature", 1.0)),
        "calibration_class_bias_vector": fit.get("class_bias", {}),
        "uncalibrated_macro_f1": out["uncalibrated_macro_f1"],
        "uncalibrated_accuracy": out["uncalibrated_accuracy"],
        "calibrated_macro_f1": out["macro_f1"],
        "calibrated_accuracy": out["accuracy"],
        "calibrator_uses_test_labels": False,
        **fit,
    }


def cache_with_path_indices(cache: SemanticTreeCache, indices: Sequence[int]) -> SemanticTreeCache:
    keep = sorted({int(idx) for idx in indices if 0 <= int(idx) < len(cache.paths)})
    if not keep:
        keep = [0]
    return replace(
        cache,
        tensor=np.asarray(cache.tensor[:, keep, :], dtype=np.float32),
        paths=[cache.paths[idx] for idx in keep],
    )


def select_paths_by_energy(cache: SemanticTreeCache, split: Mapping[str, np.ndarray], budget: float) -> list[int]:
    train_val_nodes = np.concatenate([np.asarray(split["train"], dtype=np.int64), np.asarray(split["val"], dtype=np.int64)])
    idx = local_indices(cache, train_val_nodes)
    tensor = np.asarray(cache.tensor, dtype=np.float32)
    path_count = int(tensor.shape[1])
    keep_count = max(1, min(path_count, int(np.floor(path_count * float(budget) + 1.0e-12))))
    local = tensor[idx] if len(idx) else tensor
    energy = np.mean(np.square(local), axis=(0, 2))
    return sorted(int(i) for i in np.argsort(-energy, kind="mergesort")[:keep_count].tolist())


def select_paths_by_validation(
    cache: SemanticTreeCache,
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    *,
    budget: float,
    seed: int,
    epochs: int,
    hidden_dim: int,
    device: str,
    objective: str = "accuracy",
) -> list[int]:
    path_count = len(cache.paths)
    keep_count = max(1, min(path_count, int(np.floor(path_count * float(budget) + 1.0e-12))))
    scored: list[tuple[float, int]] = []
    for idx in range(path_count):
        candidate = cache_with_path_indices(cache, [idx])
        result = evaluate_cache_classifier(method=f"path_{idx}", cache=candidate, labels=labels, split=split, classifier="linear", seed=seed, epochs=max(1, min(epochs, 5)), hidden_dim=hidden_dim, device=device)
        if objective == "accuracy":
            score = float(result["validation_accuracy"])
        elif objective == "loss":
            score = -cross_entropy_on_indices(result["local_labels"], result["all_logits"], result["val_indices"])
        else:
            score = float(result["validation_macro_f1"])
        scored.append((score, idx))
    return sorted(idx for _score, idx in sorted(scored, key=lambda item: (-item[0], item[1]))[:keep_count])


def quantized_cache(cache: SemanticTreeCache, *, bits: int = 16, budget: float = 1.0, split: Mapping[str, np.ndarray] | None = None) -> tuple[SemanticTreeCache, dict[str, Any]]:
    if bits == 16:
        base = np.asarray(cache.tensor, dtype=np.float16).astype(np.float32)
        bytes_per_value = 2
        scale_type = "float16"
    else:
        tensor = np.asarray(cache.tensor, dtype=np.float32)
        mins = np.min(tensor, axis=(0, 2), keepdims=True)
        maxs = np.max(tensor, axis=(0, 2), keepdims=True)
        scale = np.maximum(maxs - mins, 1.0e-6) / 255.0
        q = np.clip(np.round((tensor - mins) / scale), 0, 255).astype(np.uint8)
        base = (q.astype(np.float32) * scale + mins).astype(np.float32)
        bytes_per_value = 1
        scale_type = "per_path_uniform_int8"
    dtype_ratio = bytes_per_value / 4.0
    max_path_ratio = min(1.0, float(budget) / max(dtype_ratio, 1.0e-12))
    if max_path_ratio < 1.0:
        keep = select_paths_by_energy(cache, split or {"train": cache.target_nodes, "val": np.empty(0, dtype=np.int64)}, max_path_ratio)
        out = cache_with_path_indices(replace(cache, tensor=base), keep)
    else:
        out = replace(cache, tensor=base)
    delta = np.asarray(cache.tensor[:, : out.tensor.shape[1], :], dtype=np.float32) - np.asarray(out.tensor, dtype=np.float32)
    return out, {
        "quantization_bits": int(bits),
        "quantization_scale_type": scale_type,
        "quantized_bytes_per_value": int(bytes_per_value),
        "feature_l2_error_vs_full": float(np.linalg.norm(delta.reshape(-1))),
        "semantic_tree_l2_delta_vs_full": float(np.linalg.norm(delta.reshape(-1))),
    }
