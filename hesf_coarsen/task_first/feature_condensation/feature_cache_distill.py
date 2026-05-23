from __future__ import annotations

from typing import Any

import numpy as np


def _nearest_centroid_logits(x: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = np.zeros((int(num_classes), int(x.shape[1])), dtype=np.float32)
    counts = np.zeros(int(num_classes), dtype=np.float32)
    for idx in np.asarray(train_idx, dtype=np.int64).tolist():
        cls = int(labels[int(idx)])
        if cls < 0:
            continue
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


def train_feature_cache_distill(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int = 12345,
    epochs: int = 50,
    hidden_dim: int = 64,
    lr: float = 0.01,
    weight_decay: float = 1.0e-4,
    device: str = "auto",
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    train = np.asarray([int(idx) for idx in np.asarray(train_idx, dtype=np.int64).reshape(-1) if int(labels_arr[int(idx)]) >= 0], dtype=np.int64)
    num_classes = int(labels_arr[labels_arr >= 0].max(initial=0)) + 1
    if len(train) == 0:
        logits = np.zeros((int(x.shape[0]), int(num_classes)), dtype=np.float32)
        return {"logits": logits, "model": "feature_cache_distill_empty", "skipped": True}
    try:
        import torch
        from torch import nn
    except Exception:
        logits = _nearest_centroid_logits(x, labels_arr, train, num_classes)
        return {"logits": logits, "model": "nearest_centroid_feature_cache_fallback", "skipped": False}
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
    idx_t = torch.as_tensor(train, dtype=torch.long, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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
    return {"logits": logits, "model": "feature_cache_distill_mlp", "skipped": False, "device": device_name}
