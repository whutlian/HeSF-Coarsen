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


def _auto_metapath_paths(
    graph: HeteroGraph,
    max_paths: int,
    type_names: dict[str, int],
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for relation_id in sorted(graph.relations):
        if len(paths) >= max_paths:
            break
        rel = graph.relations[int(relation_id)]
        if rel.src_type == rel.dst_type:
            continue
        spec = graph.relation_specs.get(int(relation_id))
        relation_name = str(spec.name) if spec is not None else f"relation_{relation_id}"
        src_name = _type_display_name(int(rel.src_type), type_names)
        dst_name = _type_display_name(int(rel.dst_type), type_names)
        paths.append(
            {
                "name": f"auto_{src_name}_{relation_name}_{dst_name}_{src_name}",
                "start_type": int(rel.src_type),
                "end_type": int(rel.src_type),
                "steps": [
                    {"relation_id": int(relation_id), "direction": "forward"},
                    {"relation_id": int(relation_id), "direction": "backward"},
                ],
            }
        )
        if len(paths) >= max_paths:
            break
        paths.append(
            {
                "name": f"auto_{dst_name}_{relation_name}_{src_name}_{dst_name}",
                "start_type": int(rel.dst_type),
                "end_type": int(rel.dst_type),
                "steps": [
                    {"relation_id": int(relation_id), "direction": "backward"},
                    {"relation_id": int(relation_id), "direction": "forward"},
                ],
            }
        )

    if paths:
        return paths[:max_paths]

    for relation_id in sorted(graph.relations):
        if len(paths) >= max_paths:
            break
        rel = graph.relations[int(relation_id)]
        if rel.src_type != rel.dst_type:
            continue
        spec = graph.relation_specs.get(int(relation_id))
        relation_name = str(spec.name) if spec is not None else f"relation_{relation_id}"
        src_name = _type_display_name(int(rel.src_type), type_names)
        paths.append(
            {
                "name": f"auto_{src_name}_{relation_name}_{src_name}",
                "start_type": int(rel.src_type),
                "end_type": int(rel.src_type),
                "steps": [{"relation_id": int(relation_id), "direction": "forward"}],
            }
        )
    return paths[:max_paths]


def resolve_metapath_paths(
    graph: HeteroGraph,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    cfg = config.get("metapath_sketch", {})
    max_paths = int(cfg.get("max_paths", 3))
    max_path_length = int(cfg.get("max_path_length", 3))
    type_names = _type_name_map(graph, config)
    raw_paths = list(cfg.get("paths", []))
    auto_generated = False
    if not raw_paths and bool(cfg.get("auto_paths", False)):
        raw_paths = _auto_metapath_paths(graph, max_paths, type_names)
        auto_generated = True
    allow_large = bool(cfg.get("allow_large_metapath_sketch", False))
    if not allow_large and len(raw_paths) > max_paths:
        raise ValueError("metapath_sketch.paths exceeds max_paths")

    paths: list[dict[str, Any]] = []
    for path_index, path in enumerate(raw_paths):
        if not allow_large and len(path.get("steps", [])) > max_path_length:
            raise ValueError("meta-path length exceeds max_path_length")
        resolved = dict(path)
        resolved["name"] = str(path.get("name", f"metapath_{path_index}"))
        resolved["start_type"] = _parse_type(path["start_type"], type_names)
        resolved["end_type"] = _parse_type(path["end_type"], type_names)
        resolved["steps"] = [
            {
                "relation_id": int(step["relation_id"]),
                "direction": str(step.get("direction", "forward")).lower(),
            }
            for step in path.get("steps", [])
        ]
        paths.append(resolved)
    return paths, auto_generated, type_names


def metapath_path_diagnostics(
    graph: HeteroGraph,
    paths: list[dict[str, Any]],
    type_names: dict[str, int],
    *,
    auto_generated: bool,
    path_weights: dict[str, float] | None = None,
    enabled: bool = True,
    operator_mode: str = "chained_sketch_channels",
) -> dict[str, Any]:
    weights = path_weights or {}
    return {
        "enabled": bool(enabled),
        "operator_mode": operator_mode,
        "num_paths": int(len(paths)),
        "auto_generated_paths": bool(auto_generated),
        "operator_weight_total": float(sum(weights.values())),
        "path_weights": {str(name): float(weight) for name, weight in weights.items()},
        "paths": [
            {
                "name": str(path.get("name", f"metapath_{idx}")),
                "length": int(len(path.get("steps", []))),
                "start_type": int(path["start_type"]),
                "end_type": int(path["end_type"]),
                "start_type_name": _type_display_name(int(path["start_type"]), type_names),
                "end_type_name": _type_display_name(int(path["end_type"]), type_names),
                "relation_names": [
                    graph.relation_specs[int(step["relation_id"])].name
                    for step in path.get("steps", [])
                    if int(step["relation_id"]) in graph.relation_specs
                ],
            }
            for idx, path in enumerate(paths)
        ],
    }


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

    paths, auto_generated, type_names = resolve_metapath_paths(graph, config)

    total_dim = int(cfg.get("dim", 8))
    seed = int(cfg.get("seed", config.get("seed", 12345)))
    row_normalize = bool(cfg.get("row_normalize", True))
    dims = _path_dims(total_dim, len(paths))
    components: list[np.ndarray] = []
    path_diags: list[dict[str, Any]] = []

    for path_index, (path, dim) in enumerate(zip(paths, dims)):
        start = perf_counter()
        name = str(path.get("name", f"metapath_{path_index}"))
        start_type = int(path["start_type"])
        end_type = int(path["end_type"])
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
        {
            "enabled": True,
            "num_paths": int(len(paths)),
            "auto_generated_paths": bool(auto_generated),
            "paths": path_diags,
        },
    )
