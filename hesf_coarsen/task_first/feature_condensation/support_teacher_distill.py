from __future__ import annotations

from typing import Any

import numpy as np

from hesf_coarsen.task_first.costs.accounting import count_model_parameters_bytes


def teacher_unavailable_result(
    *,
    dataset: str,
    student_method: str,
    teacher_method: str,
    quantization_mode: str,
    feature_cache_size_ratio: float,
    path_channel_count_ratio: float,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset),
        "student_method": str(student_method),
        "teacher_method": str(teacher_method),
        "teacher_macro": "NaN",
        "teacher_accuracy": "NaN",
        "student_macro": "NaN",
        "student_accuracy": "NaN",
        "student_teacher_agreement": "NaN",
        "student_teacher_KL": "NaN",
        "student_cost_ratio": "NaN",
        "lambda_KL": "NaN",
        "teacher_temperature": "NaN",
        "lambda_margin": "NaN",
        "feature_cache_size_ratio": float(feature_cache_size_ratio),
        "path_channel_count_ratio": float(path_channel_count_ratio),
        "quantization_mode": str(quantization_mode),
        "teacher_status": "unavailable",
        "method_failed": True,
    }


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32) / max(float(temperature), 1.0e-6)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1.0e-12)


def teacher_student_diagnostics(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    teacher = np.asarray(teacher_logits, dtype=np.float32)
    student = np.asarray(student_logits, dtype=np.float32)
    if teacher.shape != student.shape or teacher.size == 0:
        return {"student_teacher_KL": "NaN", "student_teacher_agreement": "NaN", "teacher_status": "shape_mismatch"}
    tp = _softmax(teacher, float(temperature))
    sp = _softmax(student, float(temperature))
    kl = float(np.mean(np.sum(tp * (np.log(np.maximum(tp, 1.0e-12)) - np.log(np.maximum(sp, 1.0e-12))), axis=1)))
    agreement = float(np.mean(np.argmax(teacher, axis=1) == np.argmax(student, axis=1)))
    return {"student_teacher_KL": kl, "student_teacher_agreement": agreement, "teacher_status": "valid"}


def train_support_teacher_student(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    *,
    teacher_logits: np.ndarray | None,
    teacher_method: str,
    student_method: str,
    seed: int = 12345,
    epochs: int = 5,
    hidden_dim: int = 64,
    lambda_kl: float = 0.5,
    teacher_temperature: float = 2.0,
    lambda_margin: float = 0.0,
    margin: float = 1.0,
    device: str = "auto",
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    train = np.asarray([int(idx) for idx in np.asarray(train_idx, dtype=np.int64).reshape(-1) if 0 <= int(idx) < len(labels_arr) and int(labels_arr[int(idx)]) >= 0], dtype=np.int64)
    num_classes = int(labels_arr[labels_arr >= 0].max(initial=0)) + 1
    if teacher_logits is None:
        return {"logits": np.zeros((len(x), num_classes), dtype=np.float32), "method_failed": True, "teacher_status": "unavailable", "model_param_bytes": 0}
    teacher = np.asarray(teacher_logits, dtype=np.float32)
    if str(teacher_method) == str(student_method):
        return {"logits": np.zeros((len(x), num_classes), dtype=np.float32), "method_failed": True, "teacher_status": "self_reference", "model_param_bytes": 0}
    if teacher.shape != (int(x.shape[0]), int(num_classes)):
        return {"logits": np.zeros((len(x), num_classes), dtype=np.float32), "method_failed": True, "teacher_status": "shape_mismatch", "model_param_bytes": 0}
    if len(train) == 0:
        return {"logits": np.zeros((len(x), num_classes), dtype=np.float32), "method_failed": True, "teacher_status": "no_train_labels", "model_param_bytes": 0}
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except Exception:
        return {"logits": teacher.astype(np.float32, copy=False), "method_failed": False, "teacher_status": "teacher_fallback_no_torch", "model_param_bytes": 0}
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
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-4)
    temp = max(float(teacher_temperature), 1.0e-6)
    for _ in range(max(1, int(epochs))):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits_t = model(x_t)
        ce = F.cross_entropy(logits_t[idx_t], y_t[idx_t])
        student_logprob = F.log_softmax(logits_t[idx_t] / temp, dim=1)
        teacher_prob = F.softmax(teacher_t[idx_t] / temp, dim=1)
        kl = F.kl_div(student_logprob, teacher_prob, reduction="batchmean") * temp * temp
        teacher_pred = torch.argmax(teacher_t[idx_t], dim=1)
        teacher_logit = logits_t[idx_t].gather(1, teacher_pred.reshape(-1, 1)).squeeze(1)
        masked = logits_t[idx_t].clone()
        masked.scatter_(1, teacher_pred.reshape(-1, 1), -1.0e9)
        margin_loss = torch.relu(float(margin) - (teacher_logit - torch.max(masked, dim=1).values)).mean()
        loss = ce + float(lambda_kl) * kl + float(lambda_margin) * margin_loss
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_t).detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "logits": logits,
        "method_failed": False,
        "teacher_status": "valid",
        "model": "support_teacher_distill_mlp",
        "device": device_name,
        "model_param_bytes": count_model_parameters_bytes(model),
        "lambda_KL": float(lambda_kl),
        "teacher_temperature": float(teacher_temperature),
        "lambda_margin": float(lambda_margin),
    }
