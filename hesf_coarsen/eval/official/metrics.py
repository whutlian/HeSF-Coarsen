from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from hesf_coarsen.eval.task_gnn import f1_scores


def as_logits(logits: Any) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits must be finite")
    return arr


def as_labels(labels: Any) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def softmax(logits: Any) -> np.ndarray:
    arr = as_logits(logits).astype(np.float64, copy=False)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.maximum(np.sum(exp, axis=1, keepdims=True), 1.0e-12)
    return (exp / denom).astype(np.float64, copy=False)


def classification_metrics_from_logits(logits: Any, labels: Any) -> dict[str, Any]:
    arr = as_logits(logits)
    y = as_labels(labels)
    if arr.shape[0] != y.shape[0]:
        raise ValueError("logit row count must match labels")
    if y.size == 0:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": []}
    pred = np.argmax(arr, axis=1).astype(np.int64, copy=False)
    valid = (y >= 0) & (pred >= 0)
    if not np.any(valid):
        return {"macro_f1": 0.0, "micro_f1": 0.0, "accuracy": 0.0, "pred": pred.tolist()}
    f1 = f1_scores(y[valid], pred[valid], macro_empty_class_policy="truth_pred_union")
    return {
        "macro_f1": float(f1["macro_f1"]),
        "micro_f1": float(f1["micro_f1"]),
        "accuracy": float(np.mean(y[valid] == pred[valid])),
        "pred": pred.tolist(),
    }


def calibration_quality(logits: Any, labels: Any, *, bins: int = 10) -> dict[str, float]:
    arr = as_logits(logits)
    y = as_labels(labels)
    if arr.shape[0] != y.shape[0]:
        raise ValueError("logit row count must match labels")
    valid = y >= 0
    if not np.any(valid):
        return {"ECE": 0.0, "NLL": 0.0, "Brier": 0.0}
    probs = softmax(arr[valid])
    labels_valid = y[valid]
    clipped = np.clip(probs[np.arange(labels_valid.size), labels_valid], 1.0e-12, 1.0)
    nll = float(-np.mean(np.log(clipped)))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(labels_valid.size), labels_valid] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    confidence = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels_valid).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, max(int(bins), 1) + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        if math.isclose(right, 1.0):
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return {"ECE": float(ece), "NLL": nll, "Brier": brier}


def per_class_metric_rows(
    labels: Any,
    pred: Any,
    *,
    dataset: str,
    model: str,
    method: str,
    seed: int,
    ratio: float | None,
    calibrated: bool,
    uncalibrated_lookup: Mapping[int, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    y = as_labels(labels)
    p = as_labels(pred)
    if y.shape != p.shape:
        raise ValueError("prediction count must match labels")
    classes = sorted({int(v) for v in y.tolist() if int(v) >= 0} | {int(v) for v in p.tolist() if int(v) >= 0})
    baseline = dict(uncalibrated_lookup or {})
    rows: list[dict[str, Any]] = []
    for cls in classes:
        tp = int(np.sum((y == cls) & (p == cls)))
        fp = int(np.sum((y != cls) & (p == cls)))
        fn = int(np.sum((y == cls) & (p != cls)))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = 0.0 if precision + recall <= 0.0 else float(2 * precision * recall / (precision + recall))
        old = baseline.get(int(cls), {})
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "seed": int(seed),
                "ratio": "" if ratio is None else float(ratio),
                "calibrated": bool(calibrated),
                "class_id": int(cls),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(y == cls)),
                "delta_precision_vs_uncalibrated": precision - float(old.get("precision", precision)),
                "delta_recall_vs_uncalibrated": recall - float(old.get("recall", recall)),
                "delta_f1_vs_uncalibrated": f1 - float(old.get("f1", f1)),
            }
        )
    return rows


def confusion_rows(
    labels: Any,
    pred: Any,
    *,
    dataset: str,
    model: str,
    method: str,
    seed: int,
    ratio: float | None,
    calibrated: bool,
) -> list[dict[str, Any]]:
    y = as_labels(labels)
    p = as_labels(pred)
    if y.shape != p.shape:
        raise ValueError("prediction count must match labels")
    rows: list[dict[str, Any]] = []
    for true_class in sorted({int(v) for v in y.tolist() if int(v) >= 0}):
        for pred_class in sorted({int(v) for v in p.tolist() if int(v) >= 0}):
            count = int(np.sum((y == true_class) & (p == pred_class)))
            if count:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": int(seed),
                        "ratio": "" if ratio is None else float(ratio),
                        "calibrated": bool(calibrated),
                        "true_class": int(true_class),
                        "pred_class": int(pred_class),
                        "count": count,
                    }
                )
    return rows


def ratio_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0
