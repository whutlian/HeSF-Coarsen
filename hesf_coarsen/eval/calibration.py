from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hesf_coarsen.eval.task_gnn import f1_scores


def _as_logits(logits: Any) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"logits must be a 2-D array, got shape {arr.shape}")
    return arr


def _as_labels(labels: Any) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def _scores(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0}
    pred = np.argmax(np.asarray(logits), axis=1).astype(np.int64, copy=False)
    valid = (labels >= 0) & (pred >= 0)
    if not np.any(valid):
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0}
    base = f1_scores(labels[valid], pred[valid], macro_empty_class_policy="truth_pred_union")
    return {**base, "accuracy": float(np.mean(labels[valid] == pred[valid]))}


def temperature_scale_logits(logits: Any, temperature: float) -> np.ndarray:
    arr = _as_logits(logits)
    temp = max(float(temperature), 1.0e-6)
    return (arr / temp).astype(np.float32, copy=False)


def class_bias_adjust_logits(logits: Any, class_bias: Mapping[str | int, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    arr = _as_logits(logits)
    if isinstance(class_bias, Mapping):
        bias = np.zeros(arr.shape[1], dtype=np.float32)
        for key, value in class_bias.items():
            idx = int(key)
            if 0 <= idx < arr.shape[1]:
                bias[idx] = float(value)
    else:
        bias = np.asarray(class_bias, dtype=np.float32).reshape(-1)
        if bias.size < arr.shape[1]:
            padded = np.zeros(arr.shape[1], dtype=np.float32)
            padded[: bias.size] = bias
            bias = padded
        elif bias.size > arr.shape[1]:
            bias = bias[: arr.shape[1]]
    return (arr + bias.reshape(1, -1)).astype(np.float32, copy=False)


def _candidate_biases(num_classes: int, values: Sequence[float]) -> Iterable[np.ndarray]:
    vals = tuple(float(value) for value in values)
    if int(num_classes) <= 4:
        for combo in product(vals, repeat=int(num_classes)):
            yield np.asarray(combo, dtype=np.float32)
        return
    yield np.zeros(int(num_classes), dtype=np.float32)
    for cls in range(int(num_classes)):
        for value in vals:
            if abs(float(value)) <= 1.0e-12:
                continue
            bias = np.zeros(int(num_classes), dtype=np.float32)
            bias[cls] = float(value)
            yield bias


def fit_temperature_grid(
    val_logits: Any,
    val_labels: Any,
    *,
    temperatures: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    min_macro: float | None = None,
) -> dict[str, Any]:
    logits = _as_logits(val_logits)
    labels = _as_labels(val_labels)
    best: dict[str, Any] | None = None
    for temperature in temperatures:
        adjusted = temperature_scale_logits(logits, float(temperature))
        score = _scores(adjusted, labels)
        satisfied = min_macro is None or float(score["macro_f1"]) >= float(min_macro)
        candidate = {
            "temperature": float(temperature),
            "class_bias": {str(i): 0.0 for i in range(logits.shape[1])},
            "validation_macro_f1": float(score["macro_f1"]),
            "validation_micro_f1": float(score["micro_f1"]),
            "validation_accuracy": float(score["accuracy"]),
            "constraint_satisfied": bool(satisfied),
            "calibrator_uses_test_labels": False,
        }
        if best is None or (
            bool(candidate["constraint_satisfied"]),
            float(candidate["validation_accuracy"]),
            float(candidate["validation_macro_f1"]),
            -abs(float(temperature) - 1.0),
        ) > (
            bool(best["constraint_satisfied"]),
            float(best["validation_accuracy"]),
            float(best["validation_macro_f1"]),
            -abs(float(best["temperature"]) - 1.0),
        ):
            best = candidate
    assert best is not None
    return best


def fit_class_bias_grid(
    val_logits: Any,
    val_labels: Any,
    *,
    temperature: float = 1.0,
    class_bias_values: Sequence[float] = (-0.25, 0.0, 0.25),
    min_macro: float | None = None,
) -> dict[str, Any]:
    logits = temperature_scale_logits(val_logits, float(temperature))
    labels = _as_labels(val_labels)
    best: dict[str, Any] | None = None
    for bias in _candidate_biases(logits.shape[1], class_bias_values):
        adjusted = class_bias_adjust_logits(logits, bias)
        score = _scores(adjusted, labels)
        satisfied = min_macro is None or float(score["macro_f1"]) >= float(min_macro)
        candidate = {
            "temperature": float(temperature),
            "class_bias": {str(i): float(bias[i]) for i in range(logits.shape[1])},
            "validation_macro_f1": float(score["macro_f1"]),
            "validation_micro_f1": float(score["micro_f1"]),
            "validation_accuracy": float(score["accuracy"]),
            "constraint_satisfied": bool(satisfied),
            "calibrator_uses_test_labels": False,
        }
        if best is None or (
            bool(candidate["constraint_satisfied"]),
            float(candidate["validation_accuracy"]),
            float(candidate["validation_macro_f1"]),
            -float(np.linalg.norm(bias)),
        ) > (
            bool(best["constraint_satisfied"]),
            float(best["validation_accuracy"]),
            float(best["validation_macro_f1"]),
            -float(np.linalg.norm(np.asarray(list(best["class_bias"].values()), dtype=np.float32))),
        ):
            best = candidate
    assert best is not None
    return best


def fit_macro_constrained_accuracy_calibrator(
    val_logits: Any,
    val_labels: Any,
    *,
    baseline_macro: float | None = None,
    macro_epsilon: float = 0.005,
    temperatures: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    class_bias_values: Sequence[float] = (-0.25, 0.0, 0.25),
) -> dict[str, Any]:
    logits = _as_logits(val_logits)
    labels = _as_labels(val_labels)
    uncalibrated = _scores(logits, labels)
    reference_macro = float(uncalibrated["macro_f1"] if baseline_macro is None else baseline_macro)
    min_macro = float(reference_macro) - float(macro_epsilon)
    best: dict[str, Any] | None = None
    for temperature in temperatures:
        for bias in _candidate_biases(logits.shape[1], class_bias_values):
            adjusted = class_bias_adjust_logits(temperature_scale_logits(logits, float(temperature)), bias)
            score = _scores(adjusted, labels)
            satisfied = float(score["macro_f1"]) >= min_macro
            candidate = {
                "temperature": float(temperature),
                "class_bias": {str(i): float(bias[i]) for i in range(logits.shape[1])},
                "validation_macro_f1": float(score["macro_f1"]),
                "validation_micro_f1": float(score["micro_f1"]),
                "validation_accuracy": float(score["accuracy"]),
                "uncalibrated_validation_macro_f1": float(uncalibrated["macro_f1"]),
                "uncalibrated_validation_accuracy": float(uncalibrated["accuracy"]),
                "calibration_constraint_macro": float(min_macro),
                "macro_constraint_reference": "baseline" if baseline_macro is not None else "uncalibrated",
                "constraint_satisfied": bool(satisfied),
                "calibration_objective": "validation_accuracy_macro_constrained",
                "calibrator_uses_test_labels": False,
            }
            if best is None or (
                bool(candidate["constraint_satisfied"]),
                float(candidate["validation_accuracy"]),
                float(candidate["validation_macro_f1"]),
                -abs(float(temperature) - 1.0),
                -float(np.linalg.norm(bias)),
            ) > (
                bool(best["constraint_satisfied"]),
                float(best["validation_accuracy"]),
                float(best["validation_macro_f1"]),
                -abs(float(best["temperature"]) - 1.0),
                -float(np.linalg.norm(np.asarray(list(best["class_bias"].values()), dtype=np.float32))),
            ):
                best = candidate
    assert best is not None
    return best


def apply_calibrator(logits: Any, calibrator: Mapping[str, Any]) -> np.ndarray:
    adjusted = temperature_scale_logits(logits, float(calibrator.get("temperature", 1.0)))
    return class_bias_adjust_logits(adjusted, calibrator.get("class_bias", {}))
