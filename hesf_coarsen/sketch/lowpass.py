from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.sketch.random_probe import generate_probe


def _row_normalize(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32, copy=False)
    Z = Z - Z.mean(axis=0, keepdims=True)
    return _normalize_only(Z)


def _normalize_only(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, 1e-6)


def compute_lowpass_sketch(graph: HeteroGraph, config: dict) -> np.ndarray:
    sketch_cfg = config.get("sketch", {})
    dim = int(sketch_cfg.get("dim", 32))
    order = int(sketch_cfg.get("order", 5))
    num_scales = int(sketch_cfg.get("num_scales", 2))
    dtype_name = str(sketch_cfg.get("dtype", "float16"))
    seed = int(config.get("seed", 12345))
    probe = str(sketch_cfg.get("probe", "rademacher"))

    current = generate_probe(graph.num_nodes, dim, seed, probe=probe)
    scales: list[np.ndarray] = []
    relation_weights = {relation_id: 1.0 for relation_id in graph.relations}
    for step in range(max(order, 0)):
        current = apply_fused_smoothing(graph, current, relation_weights)
        if step >= max(order - num_scales, 0):
            scales.append(current.copy())
    if not scales:
        scales.append(current)

    Z = np.mean(scales, axis=0).astype(np.float32)
    Z = Z - Z.mean(axis=0, keepdims=True)
    acceleration = config.get("acceleration", {})
    if acceleration.get("dense_backend") == "torch":
        try:
            from hesf_coarsen.ops.torch_dense import torch_row_normalize

            Z = torch_row_normalize(
                Z,
                device=str(acceleration.get("device", "auto")),
                max_bytes=acceleration.get("max_dense_bytes"),
            )
        except (ImportError, RuntimeError):
            if not bool(acceleration.get("fallback_to_numpy", True)):
                raise
            Z = _normalize_only(Z)
    else:
        Z = _normalize_only(Z)
    if dtype_name == "float16":
        return Z.astype(np.float16)
    if dtype_name == "float32":
        return Z.astype(np.float32)
    raise ValueError(f"unsupported sketch dtype: {dtype_name}")
