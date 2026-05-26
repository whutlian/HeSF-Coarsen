from __future__ import annotations

import re
from typing import Any

import numpy as np


def transform_feature_matrix(features: np.ndarray, transform_name: str, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(features, dtype=np.float32)
    name = str(transform_name)
    audit: dict[str, Any] = {
        "transform_name": name,
        "seed": int(seed),
        "input_shape": list(x.shape),
        "fit_uses_labels": False,
        "fit_uses_test_labels": False,
        "feature_dtype": "fp32",
        "feature_dim": int(x.shape[1]) if x.ndim == 2 else 0,
        "sidecar_metadata_bytes": 0,
        "metadata_keys": "",
    }
    if name in {"raw", "raw-paper", "raw_features_adapter_control"}:
        out = x.copy()
    elif name == "zero-paper":
        out = np.zeros_like(x, dtype=np.float32)
    elif name == "fp16-paper":
        out = x.astype(np.float16).astype(np.float32)
        audit["feature_dtype"] = "fp16"
        audit["sidecar_metadata_bytes"] = 4
    elif name == "int8-paper":
        out, meta = _int8_dequantize(x)
        audit.update(meta)
    elif match := re.fullmatch(r"pca-paper-(\d+)", name):
        out = _pca(x, int(match.group(1)))
        audit["feature_dim"] = int(out.shape[1])
    elif match := re.fullmatch(r"pca_svd_dim(\d+)", name):
        out = _pca(x, int(match.group(1)))
        audit["feature_dim"] = int(out.shape[1])
    elif match := re.fullmatch(r"random_projection_dim(\d+)", name):
        out = _random_projection(x, int(match.group(1)), seed=int(seed))
        audit["feature_dim"] = int(out.shape[1])
    else:
        raise ValueError(f"unsupported paper feature transform: {transform_name!r}")
    audit["output_shape"] = list(out.shape)
    audit["feature_dim"] = int(out.shape[1]) if out.ndim == 2 else 0
    return out.astype(np.float32, copy=False), audit


def _pca(x: np.ndarray, dim: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("PCA transform expects a 2D feature matrix")
    requested_dim = int(dim)
    target_dim = min(requested_dim, int(x.shape[0]), int(x.shape[1]))
    centered = x - x.mean(axis=0, keepdims=True)
    if target_dim <= 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:target_dim].T
    if projected.shape[1] < requested_dim:
        pad = np.zeros((projected.shape[0], requested_dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected.astype(np.float32), pad], axis=1)
    return projected


def _random_projection(x: np.ndarray, dim: int, *, seed: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("random projection expects a 2D feature matrix")
    rng = np.random.default_rng(int(seed))
    projection = rng.normal(0.0, 1.0 / np.sqrt(max(int(dim), 1)), size=(x.shape[1], int(dim))).astype(np.float32)
    return x @ projection


def _int8_dequantize(x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mins = x.min(axis=0, keepdims=True)
    maxs = x.max(axis=0, keepdims=True)
    scales = (maxs - mins) / 255.0
    scales[scales == 0] = 1.0
    quantized = np.clip(np.round((x - mins) / scales), 0, 255).astype(np.uint8)
    restored = quantized.astype(np.float32) * scales.astype(np.float32) + mins.astype(np.float32)
    metadata_bytes = int(scales.size * 4 + mins.size * 4)
    return restored, {
        "feature_dtype": "int8",
        "sidecar_metadata_bytes": metadata_bytes,
        "metadata_keys": "scale,zero_point",
    }
