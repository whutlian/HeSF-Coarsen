"""Experimental proxy distillation helpers for deprecated Next17 A5.

The deterministic teacher logits are not a trained teacher and must not be used
as evidence for a real distillation contribution.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1.0e-12)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)


def kl_divergence_from_logits(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    temperature: float = 1.0,
) -> float:
    student = softmax(student_logits, temperature=temperature)
    teacher = softmax(teacher_logits, temperature=temperature)
    value = np.sum(teacher * (np.log(np.maximum(teacher, 1.0e-12)) - np.log(np.maximum(student, 1.0e-12))), axis=1)
    return float(np.mean(value))


def deterministic_teacher_logits(features: np.ndarray, num_classes: int, *, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)
    weight = rng.normal(size=(x.shape[1], int(num_classes))).astype(np.float32)
    return x @ weight
