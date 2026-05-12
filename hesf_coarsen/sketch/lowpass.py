from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.progress import progress_iter
from hesf_coarsen.sketch.chebyshev import chebyshev_heat_filter
from hesf_coarsen.sketch.metapath import compute_metapath_sketch
from hesf_coarsen.sketch.relation_weights import compute_relation_weights
from hesf_coarsen.sketch.random_probe import generate_probe


def _row_normalize(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32, copy=False)
    Z = Z - Z.mean(axis=0, keepdims=True)
    return _normalize_only(Z)


def _normalize_only(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, 1e-6)


def _dtype_cast(Z: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "float16":
        return Z.astype(np.float16)
    if dtype_name == "float32":
        return Z.astype(np.float32)
    raise ValueError(f"unsupported sketch dtype: {dtype_name}")


def _row_norm_stats(Z: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(Z.astype(np.float32, copy=False), axis=1)
    return {
        "min": float(norms.min() if len(norms) else 0.0),
        "max": float(norms.max() if len(norms) else 0.0),
        "mean": float(norms.mean() if len(norms) else 0.0),
    }


def _store_diagnostics(config: dict, diagnostics: dict[str, Any] | None, payload: dict[str, Any]) -> None:
    config["_last_sketch_diagnostics"] = payload
    if diagnostics is not None:
        diagnostics.update(payload)


def _heat_component_dims(sketch_cfg: dict[str, Any], heat_times: list[float], heat_dim: int) -> list[int]:
    component_dims = sketch_cfg.get("component_dims")
    if component_dims:
        dims = [int(component_dims[f"heat_{heat_time}"]) for heat_time in heat_times]
        if sum(dims) != int(heat_dim):
            raise ValueError("sketch.component_dims heat dimensions must sum to the allocated heat dimension")
        return dims
    if not heat_times:
        return []
    base = int(heat_dim) // len(heat_times)
    remainder = int(heat_dim) % len(heat_times)
    return [base + (1 if idx < remainder else 0) for idx in range(len(heat_times))]


def _compute_lazy_sketch(
    graph: HeteroGraph,
    config: dict,
    diagnostics: dict[str, Any] | None,
) -> np.ndarray:
    sketch_cfg = config.get("sketch", {})
    dim = int(sketch_cfg.get("dim", 32))
    order = int(sketch_cfg.get("order", 5))
    num_scales = int(sketch_cfg.get("num_scales", 2))
    dtype_name = str(sketch_cfg.get("dtype", "float16"))
    seed = int(config.get("seed", 12345))
    probe = str(sketch_cfg.get("probe", "rademacher"))

    start = perf_counter()
    current = generate_probe(graph.num_nodes, dim, seed, probe=probe)
    scales: list[np.ndarray] = []
    relation_result = compute_relation_weights(graph, config, basis=current)
    smoothing_steps = max(order, 0)
    for step in progress_iter(
        range(smoothing_steps),
        total=smoothing_steps,
        desc="lazy smoothing",
        config=config,
        unit="step",
    ):
        current = apply_fused_smoothing(graph, current, relation_result.weights)
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
    Z_out = _dtype_cast(Z, dtype_name)
    runtime = float(perf_counter() - start)
    diag = {
        "sketch_method": "lazy",
        "sketch_dim": int(Z_out.shape[1]),
        "sketch_dtype": str(Z_out.dtype),
        "chebyshev_order": None,
        "heat_times": [],
        "sketch_runtime_sec": runtime,
        "sketch_component_runtime_sec": {"lazy_smoothing": runtime},
        "nan_count": int(np.isnan(Z_out).sum()),
        "inf_count": int(np.isinf(Z_out).sum()),
        "row_norm_stats": _row_norm_stats(Z_out),
        "fusion": relation_result.diagnostics,
        "metapath_sketch": {"enabled": False, "num_paths": 0, "paths": []},
    }
    _store_diagnostics(config, diagnostics, diag)
    return Z_out


def _compute_chebyshev_heat_sketch(
    graph: HeteroGraph,
    config: dict,
    diagnostics: dict[str, Any] | None,
) -> np.ndarray:
    sketch_cfg = config.get("sketch", {})
    total_dim = int(sketch_cfg.get("dim", 32))
    order = int(sketch_cfg.get("order", 5))
    heat_times = [float(value) for value in sketch_cfg.get("heat_times", [1.0])]
    dtype_name = str(sketch_cfg.get("dtype", "float16"))
    seed = int(sketch_cfg.get("seed", config.get("seed", 12345)))
    probe = str(sketch_cfg.get("probe", "rademacher"))
    row_normalize = bool(sketch_cfg.get("row_normalize", True))
    quadrature_points = sketch_cfg.get("chebyshev_quadrature_points")
    quadrature_points = None if quadrature_points in (None, "") else int(quadrature_points)
    metapath_cfg = config.get("metapath_sketch", {})
    metapath_enabled = bool(metapath_cfg.get("enabled", False))
    metapath_dim = int(metapath_cfg.get("dim", 8)) if metapath_enabled else 0
    heat_dim = max(total_dim - metapath_dim, 0) if metapath_enabled else total_dim
    if heat_dim <= 0 and not metapath_enabled:
        raise ValueError("sketch.dim must allocate at least one heat or meta-path dimension")
    heat_dims = _heat_component_dims(sketch_cfg, heat_times, heat_dim)
    fusion_cfg = config.get("fusion", {})
    symmetric = bool(fusion_cfg.get("symmetric_relation_operator", True))
    reverse_policy = str(fusion_cfg.get("reverse_relation_policy", "include_all"))

    start_total = perf_counter()
    weight_basis_dim = max(1, min(heat_dim or total_dim, int(sketch_cfg.get("weight_basis_dim", 8))))
    weight_basis = generate_probe(graph.num_nodes, weight_basis_dim, seed, probe=probe)
    relation_result = compute_relation_weights(graph, config, basis=weight_basis)

    components: list[np.ndarray] = []
    component_runtime: dict[str, float] = {}
    heat_components = [
        (idx, heat_time, dim)
        for idx, (heat_time, dim) in enumerate(zip(heat_times, heat_dims))
        if dim > 0
    ]
    for idx, heat_time, dim in progress_iter(
        heat_components,
        total=len(heat_components),
        desc="chebyshev heat components",
        config=config,
        unit="component",
    ):
        component_start = perf_counter()
        basis = generate_probe(graph.num_nodes, dim, seed + 101 * (idx + 1), probe=probe)
        component = chebyshev_heat_filter(
            graph,
            basis,
            relation_result.weights,
            heat_time=heat_time,
            order=order,
            quadrature_points=quadrature_points,
            symmetric_relation_operator=symmetric,
            reverse_relation_policy=reverse_policy,
            progress_config=config,
            progress_desc=f"chebyshev heat t={heat_time}",
        )
        components.append(component)
        component_runtime[f"heat_{heat_time}"] = float(perf_counter() - component_start)

    meta_diag = {"enabled": False, "num_paths": 0, "paths": []}
    if metapath_enabled:
        component_start = perf_counter()
        meta_result = compute_metapath_sketch(graph, config)
        components.append(meta_result.sketch)
        meta_diag = meta_result.diagnostics
        component_runtime["metapath"] = float(perf_counter() - component_start)

    if not components:
        Z = np.empty((graph.num_nodes, 0), dtype=np.float32)
    else:
        Z = np.concatenate(components, axis=1).astype(np.float32)
    if row_normalize and Z.shape[1]:
        Z = _normalize_only(Z)
    if not np.all(np.isfinite(Z)):
        raise FloatingPointError("low-pass sketch contains NaN or Inf")
    Z_out = _dtype_cast(Z, dtype_name)
    diag = {
        "sketch_method": "chebyshev_heat",
        "sketch_dim": int(Z_out.shape[1]),
        "sketch_dtype": str(Z_out.dtype),
        "chebyshev_order": int(order),
        "heat_times": [float(value) for value in heat_times],
        "sketch_runtime_sec": float(perf_counter() - start_total),
        "sketch_component_runtime_sec": component_runtime,
        "nan_count": int(np.isnan(Z_out).sum()),
        "inf_count": int(np.isinf(Z_out).sum()),
        "row_norm_stats": _row_norm_stats(Z_out),
        "fusion": relation_result.diagnostics,
        "metapath_sketch": meta_diag,
    }
    _store_diagnostics(config, diagnostics, diag)
    return Z_out


def compute_lowpass_sketch(
    graph: HeteroGraph,
    config: dict,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    method = str(config.get("sketch", {}).get("method", "lazy")).lower()
    if method == "lazy":
        return _compute_lazy_sketch(graph, config, diagnostics)
    if method == "chebyshev_heat":
        return _compute_chebyshev_heat_sketch(graph, config, diagnostics)
    raise ValueError(f"unsupported sketch.method: {method}")
