from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hesf_coarsen.task_first.costs.accounting import count_model_parameters_bytes


@dataclass(frozen=True)
class TeacherLogits:
    source: str
    train_logits: np.ndarray
    val_logits: np.ndarray
    test_logits: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    teacher_val_macro_f1: float
    teacher_val_accuracy: float
    teacher_test_macro_f1: float
    teacher_test_accuracy: float


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32) / max(float(temperature), 1.0e-6)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1.0e-12)


def teacher_kl_diagnostics(
    teacher_logits: np.ndarray | None,
    student_logits: np.ndarray | None,
    *,
    teacher_source: str = "",
    temperature: float = 1.0,
) -> dict[str, Any]:
    if teacher_logits is None or student_logits is None:
        return {
            "teacher_available": False,
            "teacher_kl_status": "unavailable",
            "teacher_student_kl": "",
            "teacher_student_agreement": "",
        }
    teacher = np.asarray(teacher_logits, dtype=np.float32)
    student = np.asarray(student_logits, dtype=np.float32)
    if teacher.shape != student.shape or teacher.size == 0:
        return {
            "teacher_available": False,
            "teacher_kl_status": "shape_mismatch",
            "teacher_student_kl": "",
            "teacher_student_agreement": "",
        }
    if np.array_equal(teacher, student) and str(teacher_source).lower() in {"student", "student_self", "self"}:
        return {
            "teacher_available": True,
            "teacher_kl_status": "self_reference",
            "teacher_student_kl": "",
            "teacher_student_agreement": "",
        }
    tp = _softmax(teacher, temperature)
    sp = _softmax(student, temperature)
    kl = float(np.mean(np.sum(tp * (np.log(np.maximum(tp, 1.0e-12)) - np.log(np.maximum(sp, 1.0e-12))), axis=1)))
    agreement = float(np.mean(np.argmax(teacher, axis=1) == np.argmax(student, axis=1)))
    return {
        "teacher_available": True,
        "teacher_kl_status": "valid",
        "teacher_student_kl": kl,
        "teacher_student_agreement": agreement,
    }


def train_student_with_teacher(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    *,
    teacher_logits: np.ndarray | None,
    seed: int = 12345,
    epochs: int = 50,
    hidden_dim: int = 64,
    lambda_kl: float = 0.5,
    temperature: float = 2.0,
    device: str = "auto",
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    train = np.asarray([int(idx) for idx in np.asarray(train_idx, dtype=np.int64).reshape(-1) if int(labels_arr[int(idx)]) >= 0], dtype=np.int64)
    num_classes = int(labels_arr[labels_arr >= 0].max(initial=0)) + 1
    if teacher_logits is None:
        return {
            "logits": np.zeros((int(x.shape[0]), int(num_classes)), dtype=np.float32),
            "model": "student_without_teacher_skipped",
            "skipped": True,
            "teacher_kl_status": "unavailable",
            "model_param_bytes": 0,
        }
    teacher = np.asarray(teacher_logits, dtype=np.float32)
    if teacher.shape[0] != x.shape[0] or teacher.shape[1] != num_classes:
        return {
            "logits": np.zeros((int(x.shape[0]), int(num_classes)), dtype=np.float32),
            "model": "student_teacher_shape_mismatch",
            "skipped": True,
            "teacher_kl_status": "shape_mismatch",
            "model_param_bytes": 0,
        }
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except Exception:
        return {
            "logits": teacher.astype(np.float32, copy=False),
            "model": "teacher_logits_fallback_no_torch",
            "skipped": False,
            "teacher_kl_status": "valid",
            "model_param_bytes": 0,
        }
    if str(device) == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = str(device)
    dev = torch.device(device_name)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = nn.Sequential(
        nn.Linear(int(x.shape[1]), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(num_classes)),
    ).to(dev)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(labels_arr, dtype=torch.long, device=dev)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=dev)
    idx_t = torch.as_tensor(train, dtype=torch.long, device=dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-4)
    loss_fn = nn.CrossEntropyLoss()
    temp = max(float(temperature), 1.0e-6)
    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits_t = model(x_t)
        ce = loss_fn(logits_t[idx_t], y_t[idx_t])
        student_logprob = F.log_softmax(logits_t[idx_t] / temp, dim=1)
        teacher_prob = F.softmax(teacher_t[idx_t] / temp, dim=1)
        kl = F.kl_div(student_logprob, teacher_prob, reduction="batchmean") * temp * temp
        loss = ce + float(lambda_kl) * kl
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_t).detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "logits": logits,
        "model": "feature_cache_true_distill_mlp",
        "skipped": False,
        "device": device_name,
        "teacher_kl_status": "valid",
        "lambda_kl": float(lambda_kl),
        "temperature": float(temperature),
        "model_param_bytes": count_model_parameters_bytes(model),
    }
