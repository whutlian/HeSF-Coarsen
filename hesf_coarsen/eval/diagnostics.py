from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph


def _node_counts(graph: HeteroGraph) -> dict[str, int]:
    return {
        str(int(type_id)): int(np.sum(graph.node_type == type_id))
        for type_id in sorted(np.unique(graph.node_type))
    }


def _edge_counts(graph: HeteroGraph) -> dict[str, int]:
    return {
        str(relation_id): int(rel.num_edges)
        for relation_id, rel in sorted(graph.relations.items())
    }


def _relation_weights(graph: HeteroGraph) -> dict[str, float]:
    return {
        str(relation_id): float(rel.weight.sum())
        for relation_id, rel in sorted(graph.relations.items())
    }


def _array_nbytes(array: np.ndarray | None) -> int:
    return 0 if array is None else int(array.nbytes)


def _graph_array_bytes(graph: HeteroGraph) -> int:
    total = _array_nbytes(graph.node_type) + _array_nbytes(graph.labels) + _array_nbytes(graph.partitions)
    for rel in graph.relations.values():
        total += _array_nbytes(rel.src) + _array_nbytes(rel.dst) + _array_nbytes(rel.weight)
    if graph.features is not None:
        total += sum(_array_nbytes(feature) for feature in graph.features.values())
    return int(total)


def _current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    if os.name == "nt":
        try:
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if value > 10_000_000 else value * 1024
    except Exception:
        return None


def _cuda_memory_stats() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return {"available": False}
    try:
        if not torch.cuda.is_available():
            return {"available": False}
        device = torch.cuda.current_device()
        return {
            "available": True,
            "device": int(device),
            "device_name": str(torch.cuda.get_device_name(device)),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception:
        return {"available": False}


def _sample_indices(length: int, sample_size: int) -> np.ndarray:
    if length <= 0 or sample_size <= 0:
        return np.empty(0, dtype=np.int64)
    if length <= sample_size:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=sample_size, dtype=np.int64))


def _dir_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return int(root.stat().st_size)
    return int(sum(file.stat().st_size for file in root.rglob("*") if file.is_file()))


def compute_large_graph_envelope(
    graph: HeteroGraph,
    candidate_counts: np.ndarray | None = None,
    runtime_by_stage: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    artifact_dirs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    config = config or {}
    diagnostics_cfg = config.get("diagnostics", {})
    sample_size = int(diagnostics_cfg.get("edge_sample_size", 1024))
    relation_samples: dict[str, dict[str, Any]] = {}
    total_sampled_edges = 0
    for relation_id, rel in sorted(graph.relations.items()):
        indices = _sample_indices(rel.num_edges, sample_size)
        weights = rel.weight[indices] if len(indices) else np.empty(0, dtype=np.float32)
        src = rel.src[indices] if len(indices) else np.empty(0, dtype=np.int64)
        dst = rel.dst[indices] if len(indices) else np.empty(0, dtype=np.int64)
        total_sampled_edges += int(len(indices))
        relation_samples[str(relation_id)] = {
            "edge_count": int(rel.num_edges),
            "sampled_edges": int(len(indices)),
            "sample_weight_sum": float(weights.sum()) if len(weights) else 0.0,
            "sample_weight_mean": float(weights.mean()) if len(weights) else 0.0,
            "sample_self_loop_count": int(np.sum(src == dst)) if len(indices) else 0,
            "sample_unique_src": int(len(np.unique(src))) if len(indices) else 0,
            "sample_unique_dst": int(len(np.unique(dst))) if len(indices) else 0,
        }

    runtime_by_stage = runtime_by_stage or {}
    runtime_total = float(sum(float(value) for value in runtime_by_stage.values()))
    runtime_max_stage = (
        max(runtime_by_stage, key=lambda key: float(runtime_by_stage[key]))
        if runtime_by_stage
        else None
    )
    candidate_counts = (
        np.asarray(candidate_counts, dtype=np.int64)
        if candidate_counts is not None
        else np.empty(0, dtype=np.int64)
    )
    candidate_cfg = config.get("candidates", {})
    candidate_K = int(candidate_cfg.get("total_budget_K", 0) or 0)
    candidate_store_estimated_bytes = int(
        graph.num_nodes * max(candidate_K, 0) * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize + np.dtype(np.int16).itemsize)
        + graph.num_nodes * np.dtype(np.int32).itemsize
    )
    artifact_dirs = artifact_dirs or {}
    artifact_bytes = {name: _dir_size(path) for name, path in sorted(artifact_dirs.items())}
    rss = _current_rss_bytes()
    max_ram_gb = config.get("hardware", {}).get("max_ram_gb")
    max_ram_bytes = None if max_ram_gb in (None, "") else int(float(max_ram_gb) * (1024**3))

    return {
        "edge_sample_size": sample_size,
        "total_sampled_edges": int(total_sampled_edges),
        "relation_edge_samples": relation_samples,
        "graph_array_bytes": _graph_array_bytes(graph),
        "process_rss_bytes": rss,
        "cuda_memory": _cuda_memory_stats(),
        "hardware_max_ram_bytes": max_ram_bytes,
        "rss_fraction_of_configured_ram": (
            None if rss is None or not max_ram_bytes else float(rss / max_ram_bytes)
        ),
        "candidate_store_estimated_bytes": candidate_store_estimated_bytes,
        "candidate_count_quantiles": {
            "p50": float(np.percentile(candidate_counts, 50)) if len(candidate_counts) else 0.0,
            "p90": float(np.percentile(candidate_counts, 90)) if len(candidate_counts) else 0.0,
            "p95": float(np.percentile(candidate_counts, 95)) if len(candidate_counts) else 0.0,
            "p99": float(np.percentile(candidate_counts, 99)) if len(candidate_counts) else 0.0,
        },
        "runtime_by_stage": {key: float(value) for key, value in sorted(runtime_by_stage.items())},
        "runtime_total_seconds": runtime_total,
        "runtime_max_stage": runtime_max_stage,
        "artifact_bytes_by_name": artifact_bytes,
        "artifact_bytes_total": int(sum(artifact_bytes.values())),
    }


def compute_diagnostics(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    candidate_counts: np.ndarray,
    source_counts: dict[str, int],
    runtime_by_stage: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    artifact_dirs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    sizes = assignment.cluster_sizes()
    original_weights = _relation_weights(original)
    coarse_weights = _relation_weights(coarse)
    weight_error = {
        relation_id: abs(original_weights.get(relation_id, 0.0) - coarse_weights.get(relation_id, 0.0))
        for relation_id in sorted(set(original_weights) | set(coarse_weights))
    }
    diagnostics = {
        "original_nodes": int(original.num_nodes),
        "coarse_nodes": int(coarse.num_nodes),
        "compression_ratio": float(coarse.num_nodes / max(original.num_nodes, 1)),
        "original_node_count_by_type": _node_counts(original),
        "coarse_node_count_by_type": _node_counts(coarse),
        "original_edge_count_by_relation": _edge_counts(original),
        "coarse_edge_count_by_relation": _edge_counts(coarse),
        "candidate_count_total": int(len(candidate_counts) and candidate_counts.sum()),
        "candidate_count_max": int(candidate_counts.max(initial=0)),
        "candidate_count_mean": float(candidate_counts.mean() if len(candidate_counts) else 0.0),
        "candidate_coverage": float(np.mean(candidate_counts > 0) if len(candidate_counts) else 0.0),
        "candidate_count_quantiles": {
            "p50": float(np.percentile(candidate_counts, 50)) if len(candidate_counts) else 0.0,
            "p95": float(np.percentile(candidate_counts, 95)) if len(candidate_counts) else 0.0,
            "p99": float(np.percentile(candidate_counts, 99)) if len(candidate_counts) else 0.0,
        },
        "candidate_source_counts": dict(source_counts),
        "matched_pairs": int(np.sum(sizes == 2)),
        "matched_merges": int(np.sum(np.maximum(sizes - 1, 0))),
        "singleton_ratio": float(np.sum(sizes == 1) / max(len(sizes), 1)),
        "max_cluster_size": int(sizes.max(initial=0)),
        "relation_weight_before": original_weights,
        "relation_weight_after": coarse_weights,
        "relation_weight_abs_error": weight_error,
        "runtime_by_stage": runtime_by_stage or {},
    }
    if (config or {}).get("diagnostics", {}).get("enable_large_graph_envelope", False):
        diagnostics["large_graph_envelope"] = compute_large_graph_envelope(
            original,
            candidate_counts=candidate_counts,
            runtime_by_stage=runtime_by_stage,
            config=config,
            artifact_dirs=artifact_dirs,
        )
    return diagnostics


def save_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
