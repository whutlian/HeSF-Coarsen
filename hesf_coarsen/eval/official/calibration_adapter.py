from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import numpy as np

from hesf_coarsen.eval.official.metrics import (
    as_labels,
    as_logits,
    calibration_quality,
    classification_metrics_from_logits,
)


DEFAULT_TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_CLASS_BIAS_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)


def _candidate_biases(num_classes: int, values: Sequence[float]) -> list[np.ndarray]:
    vals = tuple(float(v) for v in values)
    if int(num_classes) <= 4:
        return [np.asarray(combo, dtype=np.float32) for combo in product(vals, repeat=int(num_classes))]
    biases = [np.zeros(int(num_classes), dtype=np.float32)]
    for cls in range(int(num_classes)):
        for value in vals:
            if abs(value) <= 1.0e-12:
                continue
            bias = np.zeros(int(num_classes), dtype=np.float32)
            bias[cls] = float(value)
            biases.append(bias)
    return biases


def _apply(logits: np.ndarray, temperature: float, bias: np.ndarray) -> np.ndarray:
    return (logits / max(float(temperature), 1.0e-6) + bias.reshape(1, -1)).astype(np.float32, copy=False)


def _nested_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(labels.size, dtype=np.int64)
    if idx.size <= 2:
        return idx.copy(), idx.copy()
    rng = np.random.default_rng(int(seed) + 21021)
    calib: list[int] = []
    select: list[int] = []
    for cls in sorted({int(v) for v in labels.tolist() if int(v) >= 0}):
        cls_idx = idx[labels == cls].copy()
        rng.shuffle(cls_idx)
        if cls_idx.size >= 2:
            split = max(1, cls_idx.size // 2)
            calib.extend(int(v) for v in cls_idx[:split].tolist())
            select.extend(int(v) for v in cls_idx[split:].tolist())
        elif cls_idx.size == 1:
            (calib if len(calib) <= len(select) else select).append(int(cls_idx[0]))
    assigned = set(calib) | set(select)
    leftovers = [int(v) for v in idx.tolist() if int(v) not in assigned]
    rng.shuffle(leftovers)
    for value in leftovers:
        (calib if len(calib) <= len(select) else select).append(value)
    if not calib:
        calib = select[:1]
    if not select:
        select = calib[:1]
    return np.asarray(sorted(calib), dtype=np.int64), np.asarray(sorted(select), dtype=np.int64)


def _bias_dict(bias: np.ndarray) -> dict[str, float]:
    return {str(i): float(value) for i, value in enumerate(np.asarray(bias, dtype=np.float32).reshape(-1).tolist())}


def _search(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    eval_idx: np.ndarray,
    temperature_grid: Sequence[float],
    class_bias_grid: Sequence[float],
    macro_guard_epsilon: float,
    bias_l2_penalty: float,
    repeat: int,
    split_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_scores = classification_metrics_from_logits(logits[eval_idx], labels[eval_idx])
    base_quality = calibration_quality(logits[eval_idx], labels[eval_idx])
    min_macro = float(base_scores["macro_f1"]) - float(macro_guard_epsilon)
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for temperature in temperature_grid:
        for bias in _candidate_biases(logits.shape[1], class_bias_grid):
            adjusted = _apply(logits, float(temperature), bias)
            scores = classification_metrics_from_logits(adjusted[eval_idx], labels[eval_idx])
            quality = calibration_quality(adjusted[eval_idx], labels[eval_idx])
            bias_l2 = float(np.linalg.norm(bias))
            macro_loss = float(base_scores["macro_f1"] - scores["macro_f1"])
            candidate = {
                "nested_repeat": int(repeat),
                "split_seed": int(split_seed),
                "temperature": float(temperature),
                "class_bias_vector": _bias_dict(bias),
                "class_bias_l2": bias_l2,
                "bias_l2_penalty": float(bias_l2_penalty),
                "validation_accuracy": float(scores["accuracy"]),
                "validation_macro_f1": float(scores["macro_f1"]),
                "validation_nll": float(quality["NLL"]),
                "validation_ece": float(quality["ECE"]),
                "macro_guard_satisfied": bool(float(scores["macro_f1"]) >= min_macro),
                "accuracy_gain": float(scores["accuracy"] - base_scores["accuracy"]),
                "macro_loss": macro_loss,
                "calibration_uses_test_labels": False,
            }
            key = (
                bool(candidate["macro_guard_satisfied"]),
                float(candidate["validation_accuracy"]),
                float(candidate["validation_macro_f1"]),
                -float(candidate["validation_nll"]),
                -float(candidate["validation_ece"]),
                -(bias_l2 + float(bias_l2_penalty) * bias_l2 * bias_l2),
                -abs(float(temperature) - 1.0),
            )
            candidate["_key"] = key
            candidates.append(candidate)
            if best is None or key > best["_key"]:
                best = candidate
    assert best is not None
    best = dict(best)
    best.pop("_key", None)
    for row in candidates:
        row.pop("_key", None)
    best["uncalibrated_validation_accuracy"] = float(base_scores["accuracy"])
    best["uncalibrated_validation_macro_f1"] = float(base_scores["macro_f1"])
    best["uncalibrated_validation_nll"] = float(base_quality["NLL"])
    best["uncalibrated_validation_ece"] = float(base_quality["ECE"])
    return best, candidates


def _bias_from_dict(value: Any, num_classes: int) -> np.ndarray:
    out = np.zeros(int(num_classes), dtype=np.float32)
    if isinstance(value, dict):
        for key, raw in value.items():
            idx = int(key)
            if 0 <= idx < num_classes:
                out[idx] = float(raw)
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out[: min(out.size, arr.size)] = arr[: min(out.size, arr.size)]
    return out


def calibrate_logits_nested(
    val_logits: Any,
    val_labels: Any,
    test_logits: Any,
    *,
    split_seeds: Sequence[int] = (11, 22, 33, 44, 55),
    objective: str = "accuracy_macro_guard",
    macro_guard_epsilon: float = 0.005,
    temperature_grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
    class_bias_grid: Sequence[float] = DEFAULT_CLASS_BIAS_GRID,
    bias_l2_penalty: float = 0.0,
) -> dict[str, Any]:
    if objective != "accuracy_macro_guard":
        raise ValueError(f"unsupported calibration objective: {objective}")
    val_arr = as_logits(val_logits)
    test_arr = as_logits(test_logits)
    labels = as_labels(val_labels)
    if val_arr.shape[0] != labels.shape[0]:
        raise ValueError("validation logit row count must match validation labels")
    if val_arr.shape[1] != test_arr.shape[1]:
        raise ValueError("validation and test logits must have the same class count")

    nested_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for repeat, split_seed in enumerate(split_seeds):
        calib_idx, select_idx = _nested_split(labels, int(split_seed))
        best, split_candidates = _search(
            val_arr,
            labels,
            eval_idx=select_idx,
            temperature_grid=temperature_grid,
            class_bias_grid=class_bias_grid,
            macro_guard_epsilon=float(macro_guard_epsilon),
            bias_l2_penalty=float(bias_l2_penalty),
            repeat=repeat,
            split_seed=int(split_seed),
        )
        bias = _bias_from_dict(best["class_bias_vector"], val_arr.shape[1])
        adjusted_val = _apply(val_arr, float(best["temperature"]), bias)
        calib_scores = classification_metrics_from_logits(adjusted_val[calib_idx], labels[calib_idx])
        select_scores = classification_metrics_from_logits(adjusted_val[select_idx], labels[select_idx])
        nested_rows.append(
            {
                "nested_repeat": int(repeat),
                "split_seed": int(split_seed),
                "val_calib_size": int(calib_idx.size),
                "val_select_size": int(select_idx.size),
                "val_calib_accuracy": float(calib_scores["accuracy"]),
                "val_select_accuracy": float(select_scores["accuracy"]),
                "val_select_macro_f1": float(select_scores["macro_f1"]),
                "temperature": float(best["temperature"]),
                "class_bias_vector": best["class_bias_vector"],
                "constraint_satisfied": bool(best["macro_guard_satisfied"]),
                "calibration_uses_test_labels": False,
            }
        )
        candidates.extend(split_candidates)

    full_best, full_candidates = _search(
        val_arr,
        labels,
        eval_idx=np.arange(labels.size, dtype=np.int64),
        temperature_grid=temperature_grid,
        class_bias_grid=class_bias_grid,
        macro_guard_epsilon=float(macro_guard_epsilon),
        bias_l2_penalty=float(bias_l2_penalty),
        repeat=-1,
        split_seed=-1,
    )
    candidates.extend(full_candidates)
    best_bias = _bias_from_dict(full_best["class_bias_vector"], val_arr.shape[1])
    calibrated_val = _apply(val_arr, float(full_best["temperature"]), best_bias)
    calibrated_test = _apply(test_arr, float(full_best["temperature"]), best_bias)
    before_quality = calibration_quality(val_arr, labels)
    after_quality = calibration_quality(calibrated_val, labels)
    before_scores = classification_metrics_from_logits(val_arr, labels)
    after_scores = classification_metrics_from_logits(calibrated_val, labels)
    nested_acc = np.asarray([float(row["val_select_accuracy"]) for row in nested_rows], dtype=np.float64)
    nested_macro = np.asarray([float(row["val_select_macro_f1"]) for row in nested_rows], dtype=np.float64)
    constraint_rate = float(np.mean([bool(row["constraint_satisfied"]) for row in nested_rows])) if nested_rows else 0.0
    return {
        "calibrated_test_logits": calibrated_test,
        "best_temperature": float(full_best["temperature"]),
        "class_bias_vector": full_best["class_bias_vector"],
        "val_calib_accuracy_mean": float(np.mean([float(row["val_calib_accuracy"]) for row in nested_rows])) if nested_rows else 0.0,
        "val_select_accuracy_mean": float(np.mean(nested_acc)) if nested_acc.size else 0.0,
        "val_select_macro_mean": float(np.mean(nested_macro)) if nested_macro.size else 0.0,
        "test_accuracy": None,
        "test_macro_f1": None,
        "delta_accuracy_from_calibration": None,
        "delta_macro_from_calibration": None,
        "ece_before": float(before_quality["ECE"]),
        "ece_after": float(after_quality["ECE"]),
        "nll_before": float(before_quality["NLL"]),
        "nll_after": float(after_quality["NLL"]),
        "brier_before": float(before_quality["Brier"]),
        "brier_after": float(after_quality["Brier"]),
        "nested_accuracy_mean": float(np.mean(nested_acc)) if nested_acc.size else 0.0,
        "nested_accuracy_std": float(np.std(nested_acc)) if nested_acc.size else 0.0,
        "nested_macro_mean": float(np.mean(nested_macro)) if nested_macro.size else 0.0,
        "nested_macro_std": float(np.std(nested_macro)) if nested_macro.size else 0.0,
        "constraint_satisfied_rate": constraint_rate,
        "calibration_uses_test_labels": False,
        "uncalibrated_validation_accuracy": float(before_scores["accuracy"]),
        "uncalibrated_validation_macro_f1": float(before_scores["macro_f1"]),
        "calibrated_validation_accuracy": float(after_scores["accuracy"]),
        "calibrated_validation_macro_f1": float(after_scores["macro_f1"]),
        "calibration_candidates": candidates,
        "nested_rows": nested_rows,
    }
