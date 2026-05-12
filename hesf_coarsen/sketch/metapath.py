from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.sketch.operators import apply_relation_step
from hesf_coarsen.sketch.random_probe import generate_probe


@dataclass(frozen=True)
class MetaPathSketchResult:
    sketch: np.ndarray
    diagnostics: dict[str, Any]


def _normalize_rows(Z: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, epsilon)


def _type_name_map(graph: HeteroGraph, config: dict[str, Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    configured = config.get("type_names", {}) or config.get("metapath_sketch", {}).get("type_names", {})
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(key, (int, np.integer)) or str(key).lstrip("-").isdigit():
                mapping[str(value)] = int(key)
            else:
                mapping[str(key)] = int(value)
    for spec in graph.relation_specs.values():
        name = str(spec.name)
        if "__" in name:
            parts = name.split("__")
            if len(parts) >= 3:
                mapping.setdefault(parts[0], int(spec.src_type))
                mapping.setdefault(parts[-1], int(spec.dst_type))
        elif "_to_" in name:
            src_name, dst_name = name.split("_to_", 1)
            mapping.setdefault(src_name, int(spec.src_type))
            mapping.setdefault(dst_name, int(spec.dst_type))
    return mapping


def _type_display_name(type_id: int, type_names: dict[str, int]) -> str:
    for name, mapped_id in type_names.items():
        if int(mapped_id) == int(type_id):
            return str(name)
    return str(int(type_id))


def _parse_type(value: Any, type_names: dict[str, int]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    if text in type_names:
        return int(type_names[text])
    raise ValueError(f"unknown meta-path type name: {text}")


def _path_dims(total_dim: int, num_paths: int) -> list[int]:
    if num_paths <= 0 or total_dim <= 0:
        return []
    base = total_dim // num_paths
    remainder = total_dim % num_paths
    return [base + (1 if idx < remainder else 0) for idx in range(num_paths)]


def _mask_type(graph: HeteroGraph, H: np.ndarray, type_id: int) -> np.ndarray:
    out = np.zeros_like(H, dtype=np.float32)
    mask = graph.node_type == int(type_id)
    out[mask] = H[mask]
    return out


def compute_metapath_sketch(
    graph: HeteroGraph,
    config: dict[str, Any],
) -> MetaPathSketchResult:
    cfg = config.get("metapath_sketch", {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return MetaPathSketchResult(
            np.empty((graph.num_nodes, 0), dtype=np.float32),
            {"enabled": False, "num_paths": 0, "paths": []},
        )

    paths = list(cfg.get("paths", []))
    max_paths = int(cfg.get("max_paths", 3))
    max_path_length = int(cfg.get("max_path_length", 3))
    allow_large = bool(cfg.get("allow_large_metapath_sketch", False))
    if not allow_large and len(paths) > max_paths:
        raise ValueError("metapath_sketch.paths exceeds max_paths")
    for path in paths:
        if not allow_large and len(path.get("steps", [])) > max_path_length:
            raise ValueError("meta-path length exceeds max_path_length")

    total_dim = int(cfg.get("dim", 8))
    seed = int(cfg.get("seed", config.get("seed", 12345)))
    row_normalize = bool(cfg.get("row_normalize", True))
    dims = _path_dims(total_dim, len(paths))
    type_names = _type_name_map(graph, config)
    components: list[np.ndarray] = []
    path_diags: list[dict[str, Any]] = []

    for path_index, (path, dim) in enumerate(zip(paths, dims)):
        start = perf_counter()
        name = str(path.get("name", f"metapath_{path_index}"))
        start_type = _parse_type(path["start_type"], type_names)
        end_type = _parse_type(path["end_type"], type_names)
        current_type = start_type
        H = generate_probe(graph.num_nodes, dim, seed + path_index, probe=str(cfg.get("probe", "rademacher")))
        H = _mask_type(graph, H, start_type)
        for step in path.get("steps", []):
            relation_id = int(step["relation_id"])
            direction = str(step.get("direction", "forward")).lower()
            rel = graph.relations[relation_id]
            expected_type = rel.src_type if direction == "forward" else rel.dst_type
            next_type = rel.dst_type if direction == "forward" else rel.src_type
            if current_type != expected_type:
                raise ValueError(
                    f"meta-path {name} expects current type {expected_type} before relation {relation_id}, "
                    f"got {current_type}"
                )
            H = apply_relation_step(graph, H, relation_id, direction)
            H = _mask_type(graph, H, next_type)
            current_type = next_type
        if current_type != end_type:
            raise ValueError(f"meta-path {name} ends at type {current_type}, expected {end_type}")
        if row_normalize:
            H = _normalize_rows(H)
        components.append(H.astype(np.float32, copy=False))
        nonzero_rows = int(np.sum(np.linalg.norm(H, axis=1) > 0.0))
        spec_names = [
            graph.relation_specs[int(step["relation_id"])].name
            for step in path.get("steps", [])
            if int(step["relation_id"]) in graph.relation_specs
        ]
        path_diags.append(
            {
                "name": name,
                "dim": int(dim),
                "length": int(len(path.get("steps", []))),
                "start_type": int(start_type),
                "end_type": int(end_type),
                "start_type_name": _type_display_name(start_type, type_names),
                "end_type_name": _type_display_name(end_type, type_names),
                "relation_names": spec_names,
                "runtime_sec": float(perf_counter() - start),
                "nonzero_rows": nonzero_rows,
            }
        )

    sketch = (
        np.concatenate(components, axis=1).astype(np.float32, copy=False)
        if components
        else np.empty((graph.num_nodes, 0), dtype=np.float32)
    )
    return MetaPathSketchResult(
        sketch,
        {"enabled": True, "num_paths": int(len(paths)), "paths": path_diags},
    )
