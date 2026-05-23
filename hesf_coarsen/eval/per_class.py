from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _label_set(*arrays: Iterable[int], extra: Sequence[int] | None = None) -> list[int]:
    labels: set[int] = set(int(value) for value in extra or [])
    for arr in arrays:
        labels.update(int(value) for value in np.asarray(list(arr), dtype=np.int64).reshape(-1).tolist() if int(value) >= 0)
    return sorted(labels)


def _support_count(labels: np.ndarray, cls: int) -> int:
    if labels.size == 0:
        return 0
    return int(np.sum(labels == int(cls)))


def _baseline_value(baseline: Mapping[int, Mapping[str, float]] | None, cls: int, key: str, fallback: float) -> float:
    if not baseline:
        return float(fallback)
    return float(baseline.get(int(cls), {}).get(key, fallback))


def per_class_metrics(
    *,
    dataset: str,
    seed: int,
    method: str,
    method_family: str,
    requested_budget: float,
    cost_ratio: float,
    total_storage_ratio_vs_full_stc: float,
    calibrated: bool,
    source_method: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    train_labels: Iterable[int] = (),
    val_labels: Iterable[int] = (),
    baseline_per_class: Mapping[int, Mapping[str, float]] | None = None,
    best_uncalibrated_support_per_class: Mapping[int, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    truth = np.asarray(list(y_true), dtype=np.int64).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=np.int64).reshape(-1)
    valid = (truth >= 0) & (pred >= 0)
    truth = truth[valid]
    pred = pred[valid]
    train_arr = np.asarray(list(train_labels), dtype=np.int64).reshape(-1)
    val_arr = np.asarray(list(val_labels), dtype=np.int64).reshape(-1)
    labels = _label_set(truth.tolist(), pred.tolist(), train_arr.tolist(), val_arr.tolist())
    total = max(1, int(len(truth)))
    rows: list[dict[str, Any]] = []
    for cls in labels:
        tp = int(np.sum((truth == cls) & (pred == cls)))
        fp = int(np.sum((truth != cls) & (pred == cls)))
        fn = int(np.sum((truth == cls) & (pred != cls)))
        precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp))
        recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn))
        source_precision = _baseline_value(baseline_per_class, cls, "precision", precision)
        source_recall = _baseline_value(baseline_per_class, cls, "recall", recall)
        source_f1 = _baseline_value(baseline_per_class, cls, "f1", f1)
        best_precision = _baseline_value(best_uncalibrated_support_per_class, cls, "precision", precision)
        best_recall = _baseline_value(best_uncalibrated_support_per_class, cls, "recall", recall)
        best_f1 = _baseline_value(best_uncalibrated_support_per_class, cls, "f1", f1)
        rows.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "method": str(method),
                "method_family": str(method_family),
                "requested_budget": float(requested_budget),
                "cost_ratio": float(cost_ratio),
                "total_storage_ratio_vs_full_stc": float(total_storage_ratio_vs_full_stc),
                "calibrated": bool(calibrated),
                "source_method": str(source_method),
                "class_id": int(cls),
                "class_support_train": _support_count(train_arr, cls),
                "class_support_val": _support_count(val_arr, cls),
                "class_support_test": _support_count(truth, cls),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy_contribution": float(tp / total),
                "predicted_count": int(np.sum(pred == cls)),
                "true_count": int(np.sum(truth == cls)),
                "delta_precision_vs_uncalibrated_source": float(precision - source_precision),
                "delta_recall_vs_uncalibrated_source": float(recall - source_recall),
                "delta_f1_vs_uncalibrated_source": float(f1 - source_f1),
                "delta_precision_vs_best_uncalibrated_support": float(precision - best_precision),
                "delta_recall_vs_best_uncalibrated_support": float(recall - best_recall),
                "delta_f1_vs_best_uncalibrated_support": float(f1 - best_f1),
            }
        )
    return rows


def confusion_matrix_rows(
    *,
    dataset: str,
    seed: int,
    method: str,
    requested_budget: float,
    calibrated: bool,
    source_method: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> list[dict[str, Any]]:
    truth = np.asarray(list(y_true), dtype=np.int64).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=np.int64).reshape(-1)
    valid = (truth >= 0) & (pred >= 0)
    truth = truth[valid]
    pred = pred[valid]
    labels = _label_set(truth.tolist(), pred.tolist())
    true_totals = {label: int(np.sum(truth == label)) for label in labels}
    pred_totals = {label: int(np.sum(pred == label)) for label in labels}
    rows: list[dict[str, Any]] = []
    for true_label in labels:
        for pred_label in labels:
            count = int(np.sum((truth == true_label) & (pred == pred_label)))
            if count == 0:
                continue
            rows.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "method": str(method),
                    "requested_budget": float(requested_budget),
                    "calibrated": bool(calibrated),
                    "source_method": str(source_method),
                    "true_class": int(true_label),
                    "predicted_class": int(pred_label),
                    "count": int(count),
                    "normalized_by_true": float(count / max(1, true_totals[true_label])),
                    "normalized_by_pred": float(count / max(1, pred_totals[pred_label])),
                }
            )
    return rows


def per_class_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        out[int(row["class_id"])] = {
            "precision": float(row.get("precision", 0.0) or 0.0),
            "recall": float(row.get("recall", 0.0) or 0.0),
            "f1": float(row.get("f1", 0.0) or 0.0),
        }
    return out
