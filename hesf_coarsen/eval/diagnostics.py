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


def _cluster_size_histogram(sizes: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(sizes, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, float(percentile)))


def _cluster_size_histogram_by_type(assignment: Assignment) -> dict[str, dict[str, int]]:
    hist: dict[str, dict[str, int]] = {}
    sizes = assignment.cluster_sizes()
    for supernode, size in enumerate(sizes):
        type_key = str(int(assignment.supernode_type[supernode]))
        bucket = hist.setdefault(type_key, {})
        size_key = str(int(size))
        bucket[size_key] = int(bucket.get(size_key, 0) + 1)
    return hist


def _assignment_groups(assignment: Assignment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_ids = np.asarray(assignment.assignment, dtype=np.int64).reshape(-1)
    order = np.argsort(group_ids, kind="stable")
    counts = np.bincount(group_ids, minlength=assignment.num_supernodes).astype(np.int64, copy=False)
    starts = np.empty(assignment.num_supernodes, dtype=np.int64)
    if assignment.num_supernodes:
        starts[0] = 0
        if assignment.num_supernodes > 1:
            starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return order.astype(np.int64, copy=False), starts, counts


def _node_reduction_by_type(original: HeteroGraph, coarse: HeteroGraph) -> dict[str, int]:
    before = _node_counts(original)
    after = _node_counts(coarse)
    return {
        type_id: int(before.get(type_id, 0) - after.get(type_id, 0))
        for type_id in sorted(set(before) | set(after), key=lambda item: int(item))
    }


def _cluster_spread_values(
    values: np.ndarray | None,
    assignment: Assignment,
    *,
    mode: str = "spread",
    groups: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if values is None:
        return np.zeros(assignment.num_supernodes, dtype=np.float64)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    out = np.zeros(assignment.num_supernodes, dtype=np.float64)
    order, starts, counts = groups if groups is not None else _assignment_groups(assignment)
    for supernode in np.flatnonzero(counts > 1):
        start = int(starts[int(supernode)])
        members = order[start : start + int(counts[int(supernode)])]
        block = matrix[members]
        if mode == "variance":
            out[supernode] = float(np.mean(np.var(block, axis=0)))
        else:
            center = block.mean(axis=0, keepdims=True)
            out[supernode] = float(np.mean(np.sum((block - center) ** 2, axis=1)))
    return out


def _cluster_label_entropy_values(
    labels: np.ndarray | None,
    assignment: Assignment,
    *,
    groups: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    values = np.zeros(assignment.num_supernodes, dtype=np.float64)
    if labels is None:
        return values
    labels = np.asarray(labels).reshape(-1)
    order, starts, counts = groups if groups is not None else _assignment_groups(assignment)
    for supernode in np.flatnonzero(counts > 0):
        start = int(starts[int(supernode)])
        members = order[start : start + int(counts[int(supernode)])]
        cluster_labels = labels[members]
        cluster_labels = cluster_labels[cluster_labels >= 0]
        if len(cluster_labels) == 0:
            continue
        _label_values, label_counts = np.unique(cluster_labels, return_counts=True)
        probs = label_counts.astype(np.float64) / max(float(label_counts.sum()), 1.0)
        values[supernode] = float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-12))))
    return values


def _bad_cluster_mask(
    original: HeteroGraph,
    assignment: Assignment,
    config: dict[str, Any] | None,
    *,
    groups: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    config = config or {}
    max_cluster_size = config.get("coarsening", {}).get("max_cluster_size")
    sizes = assignment.cluster_sizes()
    bad = np.zeros(assignment.num_supernodes, dtype=bool)
    if max_cluster_size is not None:
        bad |= sizes > int(max_cluster_size)
    order, starts, counts = groups if groups is not None else _assignment_groups(assignment)
    for supernode in range(assignment.num_supernodes):
        count = int(counts[supernode])
        if count == 0:
            bad[supernode] = True
            continue
        start = int(starts[supernode])
        members = order[start : start + count]
        if len(np.unique(original.node_type[members])) > 1:
            bad[supernode] = True
    return bad


def compute_cluster_quality_diagnostics(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    *,
    Z: np.ndarray | None = None,
    relation_profiles: np.ndarray | None = None,
    conv_response: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sizes = assignment.cluster_sizes()
    groups = _assignment_groups(assignment)
    sketch_spread = _cluster_spread_values(Z, assignment, mode="spread", groups=groups)
    relation_variance = _cluster_spread_values(relation_profiles, assignment, mode="variance", groups=groups)
    conv_spread = _cluster_spread_values(conv_response, assignment, mode="spread", groups=groups)
    label_entropy = _cluster_label_entropy_values(original.labels, assignment, groups=groups)
    bad = _bad_cluster_mask(original, assignment, config, groups=groups)
    return {
        "cluster_size_p50": _safe_percentile(sizes, 50),
        "cluster_size_histogram_by_type": _cluster_size_histogram_by_type(assignment),
        "node_reduction_by_type": _node_reduction_by_type(original, coarse),
        "cluster_sketch_spread_mean": float(sketch_spread.mean() if len(sketch_spread) else 0.0),
        "cluster_sketch_spread_p95": _safe_percentile(sketch_spread, 95),
        "cluster_relation_profile_variance_mean": float(
            relation_variance.mean() if len(relation_variance) else 0.0
        ),
        "cluster_relation_profile_variance_p95": _safe_percentile(relation_variance, 95),
        "cluster_conv_response_spread_mean": float(conv_spread.mean() if len(conv_spread) else 0.0),
        "cluster_conv_response_spread_p95": _safe_percentile(conv_spread, 95),
        "cluster_label_entropy_train_only_mean": float(
            label_entropy.mean() if len(label_entropy) else 0.0
        ),
        "cluster_label_entropy_train_only_p95": _safe_percentile(label_entropy, 95),
        "bad_cluster_count": int(np.sum(bad)),
        "bad_cluster_fraction": float(np.mean(bad) if len(bad) else 0.0),
    }


def _cluster_label_entropy(labels: np.ndarray | None, assignment: Assignment) -> float:
    if labels is None:
        return 0.0
    labels = np.asarray(labels).reshape(-1)
    order, starts, counts = _assignment_groups(assignment)
    total = 0.0
    observed = 0
    for supernode in np.flatnonzero(counts > 0):
        start = int(starts[int(supernode)])
        members = order[start : start + int(counts[int(supernode)])]
        cluster_labels = labels[members]
        cluster_labels = cluster_labels[cluster_labels >= 0]
        if len(cluster_labels) == 0:
            continue
        _values, label_counts = np.unique(cluster_labels, return_counts=True)
        probs = label_counts.astype(np.float64) / max(float(label_counts.sum()), 1.0)
        total += float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-12))))
        observed += 1
    return float(total / observed) if observed else 0.0


def _partition_imbalance(graph: HeteroGraph) -> dict[str, Any]:
    if graph.partitions is None:
        partitions = np.zeros(graph.num_nodes, dtype=np.int32)
    else:
        partitions = np.asarray(graph.partitions, dtype=np.int64).reshape(-1)
    if len(partitions) == 0:
        return {
            "partition_count": 0,
            "max_count": 0,
            "mean_count": 0.0,
            "max_to_mean": 0.0,
            "by_type": {},
        }
    _values, counts = np.unique(partitions, return_counts=True)
    mean_count = float(np.mean(counts)) if len(counts) else 0.0
    by_type: dict[str, dict[str, float | int]] = {}
    for type_id in sorted(np.unique(graph.node_type)):
        mask = graph.node_type == type_id
        _type_values, type_counts = np.unique(partitions[mask], return_counts=True)
        type_mean = float(np.mean(type_counts)) if len(type_counts) else 0.0
        by_type[str(int(type_id))] = {
            "partition_count": int(len(type_counts)),
            "max_count": int(type_counts.max(initial=0)),
            "mean_count": type_mean,
            "max_to_mean": float(type_counts.max(initial=0) / max(type_mean, 1.0e-12)),
        }
    return {
        "partition_count": int(len(counts)),
        "max_count": int(counts.max(initial=0)),
        "mean_count": mean_count,
        "max_to_mean": float(counts.max(initial=0) / max(mean_count, 1.0e-12)),
        "by_type": by_type,
    }


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
    Z: np.ndarray | None = None,
    relation_profiles: np.ndarray | None = None,
    conv_response: np.ndarray | None = None,
    candidate_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sizes = assignment.cluster_sizes()
    original_weights = _relation_weights(original)
    coarse_weights = _relation_weights(coarse)
    weight_error = {
        relation_id: abs(original_weights.get(relation_id, 0.0) - coarse_weights.get(relation_id, 0.0))
        for relation_id in sorted(set(original_weights) | set(coarse_weights))
    }
    candidate_generation = candidate_generation or {}
    candidate_substage_times = {
        key: float(value)
        for key, value in (candidate_generation.get("substage_times") or {}).items()
    }
    for key in (
        "onehop",
        "incident_index_build",
        "twohop_expansion",
        "simhash",
        "bucket_emit",
        "partition_ann",
        "fallback",
        "store_finalize",
    ):
        candidate_substage_times.setdefault(key, 0.0)
    candidate_generation_time = float(
        candidate_generation.get(
            "total_time",
            (runtime_by_stage or {}).get("candidates", 0.0),
        )
        or 0.0
    )
    retained_pair_count = int(
        candidate_generation.get("retained_pair_count", sum(int(value) for value in source_counts.values()))
        or 0
    )
    candidate_pairs_per_sec = float(
        candidate_generation.get(
            "candidate_pairs_per_sec",
            retained_pair_count / candidate_generation_time if candidate_generation_time > 0.0 else 0.0,
        )
        or 0.0
    )
    memory_by_candidate_buffers = dict(candidate_generation.get("memory_by_candidate_buffers") or {})
    if "estimated_total_bytes" not in memory_by_candidate_buffers:
        candidate_cfg = (config or {}).get("candidates", {})
        candidate_K = int(candidate_cfg.get("total_budget_K", 0) or 0)
        memory_by_candidate_buffers["estimated_total_bytes"] = int(
            original.num_nodes
            * max(candidate_K, 0)
            * (
                np.dtype(np.int64).itemsize
                + np.dtype(np.float32).itemsize
                + np.dtype(np.int16).itemsize
            )
            + original.num_nodes * np.dtype(np.int32).itemsize
        )
    candidate_source_coverage = {
        str(key): float(value)
        for key, value in (candidate_generation.get("source_node_coverage") or {}).items()
    }
    for source in ("onehop", "capped_twohop", "bucket", "fallback", "partition_ann"):
        candidate_source_coverage.setdefault(source, 0.0)
    for source in source_counts:
        candidate_source_coverage.setdefault(str(source), 0.0)
    generated_candidates_by_source = {
        str(source): int(float(stats.get("pairs_considered", 0) or 0))
        for source, stats in (candidate_generation.get("source_generation") or {}).items()
        if isinstance(stats, dict)
    }
    if not generated_candidates_by_source:
        generated_candidates_by_source = {str(key): int(value) for key, value in source_counts.items()}
    cluster_size_histogram = _cluster_size_histogram(sizes)
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
        "generated_candidates_by_source": generated_candidates_by_source,
        "candidate_generation_time": candidate_generation_time,
        "candidate_pairs_per_sec": candidate_pairs_per_sec,
        "candidate_retained_pair_count": retained_pair_count,
        "candidate_substage_times": candidate_substage_times,
        "candidate_source_generation": candidate_generation.get("source_generation", {}),
        "candidate_source_coverage": candidate_source_coverage,
        "bucket_coverage": float(candidate_source_coverage.get("bucket", 0.0)),
        "twohop_expansion_time": float(candidate_substage_times.get("twohop_expansion", 0.0)),
        "partition_imbalance": _partition_imbalance(original),
        "memory_by_candidate_buffers": memory_by_candidate_buffers,
        "matched_pairs": int(np.sum(sizes == 2)),
        "matched_merges": int(np.sum(np.maximum(sizes - 1, 0))),
        "matched_units": int(np.sum(np.maximum(sizes - 1, 0))),
        "cluster_count": int(assignment.num_supernodes),
        "node_reduction": int(original.num_nodes - coarse.num_nodes),
        "node_reduction_ratio": float(
            (original.num_nodes - coarse.num_nodes) / max(original.num_nodes, 1)
        ),
        "cluster_size_histogram": cluster_size_histogram,
        "cluster_size_hist": cluster_size_histogram,
        "cluster_size_mean": float(sizes.mean() if len(sizes) else 0.0),
        "cluster_size_p50": float(np.percentile(sizes, 50)) if len(sizes) else 0.0,
        "cluster_size_p95": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
        "cluster_size_p99": float(np.percentile(sizes, 99)) if len(sizes) else 0.0,
        "non_singleton_cluster_count": int(np.sum(sizes > 1)),
        "cluster_label_entropy": _cluster_label_entropy(original.labels, assignment),
        "singleton_ratio": float(np.sum(sizes == 1) / max(len(sizes), 1)),
        "max_cluster_size": int(sizes.max(initial=0)),
        "relation_weight_before": original_weights,
        "relation_weight_after": coarse_weights,
        "relation_weight_abs_error": weight_error,
        "runtime_by_stage": runtime_by_stage or {},
    }
    diagnostics.update(
        compute_cluster_quality_diagnostics(
            original,
            coarse,
            assignment,
            Z=Z,
            relation_profiles=relation_profiles,
            conv_response=conv_response,
            config=config,
        )
    )
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
