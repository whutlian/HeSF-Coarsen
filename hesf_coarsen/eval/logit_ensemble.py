from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from hesf_coarsen.eval.task_gnn import f1_scores


def as_logits(logits: Any) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {arr.shape}")
    return arr


def as_labels(labels: Any) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def softmax(logits: Any) -> np.ndarray:
    arr = as_logits(logits)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1.0e-12)


def scores_from_logits(logits: Any, labels: Any) -> dict[str, Any]:
    arr = as_logits(logits)
    y = as_labels(labels)
    if arr.shape[0] != y.shape[0]:
        raise ValueError(f"logits/labels length mismatch: {arr.shape[0]} vs {y.shape[0]}")
    pred = np.argmax(arr, axis=1).astype(np.int64, copy=False)
    valid = (y >= 0) & (pred >= 0)
    if not np.any(valid):
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": pred.tolist()}
    base = f1_scores(y[valid], pred[valid], macro_empty_class_policy="truth_pred_union")
    return {**base, "accuracy": float(np.mean(y[valid] == pred[valid])), "pred": pred.tolist()}


def calibration_metrics(logits: Any, labels: Any, *, bins: int = 10) -> dict[str, float]:
    arr = as_logits(logits)
    y = as_labels(labels)
    if arr.shape[0] != y.shape[0] or arr.size == 0:
        return {"ECE": 0.0, "NLL": 0.0, "Brier": 0.0}
    valid = y >= 0
    if not np.any(valid):
        return {"ECE": 0.0, "NLL": 0.0, "Brier": 0.0}
    probs = softmax(arr[valid])
    truth = y[valid]
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == truth).astype(np.float32)
    ece = 0.0
    for start in np.linspace(0.0, 1.0, int(bins), endpoint=False):
        stop = start + 1.0 / int(bins)
        if stop >= 1.0:
            mask = (conf >= start) & (conf <= stop)
        else:
            mask = (conf >= start) & (conf < stop)
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(acc[mask]) - np.mean(conf[mask])))
    nll = float(-np.mean(np.log(np.maximum(probs[np.arange(len(truth)), truth], 1.0e-12))))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(truth)), truth] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {"ECE": float(ece), "NLL": nll, "Brier": brier}


def _weight_grid(count: int, step: float) -> list[np.ndarray]:
    slots = int(round(1.0 / float(step)))
    out: list[np.ndarray] = []

    def _recurse(prefix: list[int], remaining: int, left: int) -> None:
        if remaining == 1:
            out.append(np.asarray([*prefix, left], dtype=np.float32) / float(slots))
            return
        for value in range(left + 1):
            _recurse([*prefix, value], remaining - 1, left - value)

    _recurse([], int(count), slots)
    return out


def _stack_sources(source_logits: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    names = sorted(str(name) for name in source_logits)
    if not names:
        raise ValueError("at least one source logit matrix is required")
    arrays = [as_logits(source_logits[name]) for name in names]
    first_shape = arrays[0].shape
    for name, arr in zip(names, arrays):
        if arr.shape != first_shape:
            raise ValueError(f"source {name} shape {arr.shape} does not match {first_shape}")
    return names, np.stack(arrays, axis=0).astype(np.float32, copy=False)


def _weighted_logits(stacked: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights.astype(np.float32), stacked, axes=(0, 0)).astype(np.float32, copy=False)


def search_global_convex_ensemble(
    val_logits_by_method: Mapping[str, Any],
    val_labels: Any,
    test_logits_by_method: Mapping[str, Any],
    test_labels: Any,
    *,
    macro_floor: float,
    grid_step: float = 0.1,
) -> dict[str, Any]:
    names, val_stack = _stack_sources(val_logits_by_method)
    test_names, test_stack = _stack_sources(test_logits_by_method)
    if test_names != names:
        raise ValueError("validation and test source names must match")
    y_val = as_labels(val_labels)
    y_test = as_labels(test_labels)
    best: dict[str, Any] | None = None
    for weights in _weight_grid(len(names), float(grid_step)):
        val_logits = _weighted_logits(val_stack, weights)
        val_scores = scores_from_logits(val_logits, y_val)
        satisfied = float(val_scores["macro_f1"]) >= float(macro_floor)
        key = (
            bool(satisfied),
            float(val_scores["accuracy"]),
            float(val_scores["macro_f1"]),
            -float(np.linalg.norm(weights)),
        )
        if best is None or key > best["_key"]:
            test_logits = _weighted_logits(test_stack, weights)
            test_scores = scores_from_logits(test_logits, y_test)
            best = {
                "_key": key,
                "ensemble_mode": "global_convex",
                "source_methods": names,
                "weights": {name: float(weight) for name, weight in zip(names, weights.tolist())},
                "per_class_weights": {},
                "confidence_threshold": "",
                "constraint_satisfied": bool(satisfied),
                "val_logits": val_logits,
                "test_logits": test_logits,
                "val_macro": float(val_scores["macro_f1"]),
                "val_accuracy": float(val_scores["accuracy"]),
                "test_macro": float(test_scores["macro_f1"]),
                "test_accuracy": float(test_scores["accuracy"]),
                **calibration_metrics(test_logits, y_test),
            }
    assert best is not None
    best.pop("_key", None)
    return best


def search_per_class_ensemble(
    val_logits_by_method: Mapping[str, Any],
    val_labels: Any,
    test_logits_by_method: Mapping[str, Any],
    test_labels: Any,
    *,
    macro_floor: float,
    l2_penalty: float = 0.01,
) -> dict[str, Any]:
    names, val_stack = _stack_sources(val_logits_by_method)
    test_names, test_stack = _stack_sources(test_logits_by_method)
    if test_names != names:
        raise ValueError("validation and test source names must match")
    class_count = int(val_stack.shape[2])
    y_val = as_labels(val_labels)
    y_test = as_labels(test_labels)
    best: dict[str, Any] | None = None
    max_assignments = len(names) ** class_count
    assignments = product(range(len(names)), repeat=class_count) if max_assignments <= 20000 else []
    for assignment in assignments:
        weights = np.zeros((len(names), class_count), dtype=np.float32)
        for cls, source_idx in enumerate(assignment):
            weights[int(source_idx), int(cls)] = 1.0
        val_logits = np.sum(val_stack * weights[:, None, :], axis=0)
        val_scores = scores_from_logits(val_logits, y_val)
        satisfied = float(val_scores["macro_f1"]) >= float(macro_floor)
        key = (
            bool(satisfied),
            float(val_scores["accuracy"]),
            float(val_scores["macro_f1"]) - float(l2_penalty) * float(np.linalg.norm(weights)),
        )
        if best is None or key > best["_key"]:
            test_logits = np.sum(test_stack * weights[:, None, :], axis=0)
            test_scores = scores_from_logits(test_logits, y_test)
            best = {
                "_key": key,
                "ensemble_mode": "per_class",
                "source_methods": names,
                "weights": {},
                "per_class_weights": {
                    str(cls): {names[src]: float(weights[src, cls]) for src in range(len(names))}
                    for cls in range(class_count)
                },
                "confidence_threshold": "",
                "constraint_satisfied": bool(satisfied),
                "val_logits": val_logits,
                "test_logits": test_logits,
                "val_macro": float(val_scores["macro_f1"]),
                "val_accuracy": float(val_scores["accuracy"]),
                "test_macro": float(test_scores["macro_f1"]),
                "test_accuracy": float(test_scores["accuracy"]),
                **calibration_metrics(test_logits, y_test),
            }
    if best is None:
        return search_global_convex_ensemble(
            val_logits_by_method,
            val_labels,
            test_logits_by_method,
            test_labels,
            macro_floor=macro_floor,
            grid_step=1.0,
        )
    best.pop("_key", None)
    return best


def search_confidence_gated_ensemble(
    val_logits_by_method: Mapping[str, Any],
    val_labels: Any,
    test_logits_by_method: Mapping[str, Any],
    test_labels: Any,
    *,
    macro_floor: float,
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
    base_method: str | None = None,
) -> dict[str, Any]:
    names, val_stack = _stack_sources(val_logits_by_method)
    test_names, test_stack = _stack_sources(test_logits_by_method)
    if test_names != names:
        raise ValueError("validation and test source names must match")
    base = str(base_method or ("HeSF-CAL-H6" if "HeSF-CAL-H6" in names else names[0]))
    base_idx = names.index(base) if base in names else 0
    blend_weights = np.full(len(names), 1.0 / float(len(names)), dtype=np.float32)
    y_val = as_labels(val_labels)
    y_test = as_labels(test_labels)
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        base_prob_val = np.max(softmax(val_stack[base_idx]), axis=1)
        base_prob_test = np.max(softmax(test_stack[base_idx]), axis=1)
        blend_val = _weighted_logits(val_stack, blend_weights)
        blend_test = _weighted_logits(test_stack, blend_weights)
        val_logits = np.where((base_prob_val >= float(threshold))[:, None], val_stack[base_idx], blend_val)
        test_logits = np.where((base_prob_test >= float(threshold))[:, None], test_stack[base_idx], blend_test)
        val_scores = scores_from_logits(val_logits, y_val)
        satisfied = float(val_scores["macro_f1"]) >= float(macro_floor)
        key = (bool(satisfied), float(val_scores["accuracy"]), float(val_scores["macro_f1"]), -abs(float(threshold) - 0.7))
        if best is None or key > best["_key"]:
            test_scores = scores_from_logits(test_logits, y_test)
            best = {
                "_key": key,
                "ensemble_mode": "confidence_gated",
                "source_methods": names,
                "weights": {name: float(weight) for name, weight in zip(names, blend_weights.tolist())},
                "per_class_weights": {},
                "confidence_threshold": float(threshold),
                "base_method": names[base_idx],
                "constraint_satisfied": bool(satisfied),
                "val_logits": val_logits,
                "test_logits": test_logits,
                "val_macro": float(val_scores["macro_f1"]),
                "val_accuracy": float(val_scores["accuracy"]),
                "test_macro": float(test_scores["macro_f1"]),
                "test_accuracy": float(test_scores["accuracy"]),
                **calibration_metrics(test_logits, y_test),
            }
    assert best is not None
    best.pop("_key", None)
    return best
