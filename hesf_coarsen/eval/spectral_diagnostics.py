from __future__ import annotations

from typing import Any

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.spectral import dirichlet_energy
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing


def _relative_error(before: float, after: float) -> float:
    denom = max(abs(float(before)), 1e-12)
    return float(abs(float(after) - float(before)) / denom)


def _relation_energy(graph: HeteroGraph, relation_id: int, signals: np.ndarray) -> float:
    rel = graph.relations[int(relation_id)]
    if rel.num_edges == 0:
        return 0.0
    diff = signals[rel.src] - signals[rel.dst]
    return float(np.sum(rel.weight.astype(np.float64) * np.sum(diff * diff, axis=1)))


def _aggregate_signals(signals: np.ndarray, assignment: Assignment) -> np.ndarray:
    coarse = np.zeros((assignment.num_supernodes, signals.shape[1]), dtype=np.float32)
    np.add.at(coarse, assignment.assignment, signals.astype(np.float32, copy=False))
    counts = assignment.cluster_sizes().astype(np.float32)
    coarse /= np.maximum(counts[:, None], 1.0)
    return coarse


def _smooth(
    graph: HeteroGraph,
    signals: np.ndarray,
    smoothing_steps: int,
    relation_weights: dict[int, float] | None,
) -> np.ndarray:
    smoothed = signals.astype(np.float32, copy=True)
    for _ in range(max(int(smoothing_steps), 0)):
        smoothed = apply_fused_smoothing(graph, smoothed, relation_weights=relation_weights)
    return smoothed


def _fused_energy(
    graph: HeteroGraph,
    signals: np.ndarray,
    relation_weights: dict[int, float] | None,
) -> float:
    sketch = apply_fused_smoothing(graph, signals, relation_weights=relation_weights)
    return float(np.sum(sketch.astype(np.float64) * sketch.astype(np.float64)))


def _normalized_relation_weights(
    graph: HeteroGraph,
    relation_weights: dict[int, float] | None,
) -> dict[int, float]:
    if relation_weights is None:
        if not graph.relations:
            return {}
        uniform = 1.0 / len(graph.relations)
        return {int(relation_id): uniform for relation_id in graph.relations}
    cleaned = {
        int(relation_id): max(float(weight), 0.0)
        for relation_id, weight in relation_weights.items()
        if int(relation_id) in graph.relations
    }
    total = float(sum(cleaned.values()))
    if total <= 0.0:
        if not graph.relations:
            return {}
        uniform = 1.0 / len(graph.relations)
        return {int(relation_id): uniform for relation_id in graph.relations}
    return {relation_id: weight / total for relation_id, weight in cleaned.items()}


def _relation_weighted_fused_energy(
    graph: HeteroGraph,
    signals: np.ndarray,
    relation_weights: dict[int, float] | None,
) -> float:
    weights = _normalized_relation_weights(graph, relation_weights)
    return float(
        sum(
            float(weights.get(int(relation_id), 0.0)) * _relation_energy(graph, int(relation_id), signals)
            for relation_id in graph.relations
        )
    )


def _inner_product_relative_error(Z: np.ndarray, Z_c: np.ndarray) -> float:
    q = min(Z.shape[1], Z_c.shape[1])
    if q == 0:
        return 0.0
    before = Z[:, :q].T @ Z[:, :q]
    after = Z_c[:, :q].T @ Z_c[:, :q]
    scale = max(float(np.linalg.norm(before, ord="fro")), 1e-12)
    return float(np.linalg.norm(before - after, ord="fro") / scale)


def _dense_smoothing_operator(
    graph: HeteroGraph,
    relation_weights: dict[int, float] | None,
) -> np.ndarray:
    eye = np.eye(graph.num_nodes, dtype=np.float32)
    return apply_fused_smoothing(graph, eye, relation_weights=relation_weights).astype(np.float64)


def _exact_eigenvalue_sanity(
    original: HeteroGraph,
    coarse: HeteroGraph,
    relation_weights: dict[int, float] | None,
    max_nodes: int | None,
    k: int = 8,
) -> dict[str, Any] | None:
    if max_nodes is None or max_nodes <= 0:
        return None
    if original.num_nodes > max_nodes or coarse.num_nodes > max_nodes:
        return {
            "status": "skipped",
            "reason": "node_count_exceeds_limit",
            "max_nodes": int(max_nodes),
            "original_nodes": int(original.num_nodes),
            "coarse_nodes": int(coarse.num_nodes),
        }
    original_operator = _dense_smoothing_operator(original, relation_weights)
    coarse_operator = _dense_smoothing_operator(coarse, relation_weights)
    original_laplacian = np.eye(original.num_nodes) - original_operator
    coarse_laplacian = np.eye(coarse.num_nodes) - coarse_operator
    original_laplacian = 0.5 * (original_laplacian + original_laplacian.T)
    coarse_laplacian = 0.5 * (coarse_laplacian + coarse_laplacian.T)
    original_values = np.linalg.eigvalsh(original_laplacian)
    coarse_values = np.linalg.eigvalsh(coarse_laplacian)
    q = min(int(k), len(original_values), len(coarse_values))
    before = original_values[:q]
    after = coarse_values[:q]
    denom = max(float(np.linalg.norm(before)), 1e-12)
    return {
        "status": "computed",
        "num_eigenvalues": int(q),
        "original_smallest_eigenvalues": [float(value) for value in before],
        "coarse_smallest_eigenvalues": [float(value) for value in after],
        "relative_error": float(np.linalg.norm(before - after) / denom),
    }


def _assignment_from_pairs(graph: HeteroGraph, pairs: list[tuple[int, int]], max_pairs: int) -> Assignment:
    assignment = np.full(graph.num_nodes, -1, dtype=np.int64)
    super_types: list[int] = []
    matched = 0
    for i, j in pairs:
        if matched >= max_pairs:
            break
        i = int(i)
        j = int(j)
        if i == j or assignment[i] >= 0 or assignment[j] >= 0:
            continue
        if graph.node_type[i] != graph.node_type[j]:
            continue
        super_id = len(super_types)
        assignment[i] = super_id
        assignment[j] = super_id
        super_types.append(int(graph.node_type[i]))
        matched += 1
    for node in range(graph.num_nodes):
        if assignment[node] >= 0:
            continue
        super_id = len(super_types)
        assignment[node] = super_id
        super_types.append(int(graph.node_type[node]))
    return Assignment(assignment, np.asarray(super_types, dtype=np.int32))


def _random_baseline_pairs(graph: HeteroGraph, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    pairs: list[tuple[int, int]] = []
    for type_id in sorted(np.unique(graph.node_type)):
        nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        rng.shuffle(nodes)
        pairs.extend((int(i), int(j)) for i, j in zip(nodes[::2], nodes[1::2]))
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def _heavy_edge_baseline_pairs(graph: HeteroGraph) -> list[tuple[int, int]]:
    scores: dict[tuple[int, int], float] = {}
    for rel in graph.relations.values():
        same_type = graph.node_type[rel.src] == graph.node_type[rel.dst]
        for src, dst, weight in zip(rel.src[same_type], rel.dst[same_type], rel.weight[same_type]):
            if int(src) == int(dst):
                continue
            key = (int(src), int(dst)) if int(src) < int(dst) else (int(dst), int(src))
            scores[key] = scores.get(key, 0.0) + float(weight)
    return [
        pair
        for pair, _score in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), int(item[0][0]), int(item[0][1])),
        )
    ]


def _degree_by_node(graph: HeteroGraph) -> np.ndarray:
    degree = np.zeros(graph.num_nodes, dtype=np.float64)
    for rel in graph.relations.values():
        np.add.at(degree, rel.src, rel.weight.astype(np.float64))
        np.add.at(degree, rel.dst, rel.weight.astype(np.float64))
    return degree


def _graphzoom_style_baseline_pairs(graph: HeteroGraph) -> list[tuple[int, int]]:
    degree = _degree_by_node(graph)
    pairs: list[tuple[int, int]] = []
    for type_id in sorted(np.unique(graph.node_type)):
        nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        order = np.lexsort((nodes, -degree[nodes]))
        ordered = nodes[order]
        pairs.extend((int(i), int(j)) for i, j in zip(ordered[::2], ordered[1::2]))
    return pairs


def _convmatch_style_baseline_pairs(
    graph: HeteroGraph,
    relation_weights: dict[int, float] | None,
    seed: int,
    dim: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    probe = rng.standard_normal((graph.num_nodes, max(int(dim), 1))).astype(np.float32)
    embedding = apply_fused_smoothing(graph, probe, relation_weights=relation_weights)
    pairs: list[tuple[int, int]] = []
    for type_id in sorted(np.unique(graph.node_type)):
        nodes = np.flatnonzero(graph.node_type == int(type_id)).astype(np.int64)
        if len(nodes) < 2:
            continue
        order = np.lexsort((nodes, embedding[nodes, 0]))
        ordered = nodes[order]
        pairs.extend((int(i), int(j)) for i, j in zip(ordered[::2], ordered[1::2]))
    return pairs


def _baseline_assignment(
    graph: HeteroGraph,
    method: str,
    max_pairs: int,
    seed: int,
    relation_weights: dict[int, float] | None,
    dim: int,
) -> Assignment:
    method = method.lower()
    if method == "random":
        pairs = _random_baseline_pairs(graph, max_pairs, seed)
    elif method == "heavy_edge":
        pairs = _heavy_edge_baseline_pairs(graph)
    elif method == "graphzoom_style":
        pairs = _graphzoom_style_baseline_pairs(graph)
    elif method == "convmatch_style":
        pairs = _convmatch_style_baseline_pairs(graph, relation_weights, seed, dim)
    else:
        raise ValueError(f"unsupported spectral baseline method: {method}")
    return _assignment_from_pairs(graph, pairs, max_pairs)


def _baseline_comparison(
    original: HeteroGraph,
    actual_assignment: Assignment,
    seed: int,
    relation_weights: dict[int, float] | None,
    Z: np.ndarray,
    smoothing_steps: int,
    baseline_methods: str | list[str] | tuple[str, ...] | None,
    baseline_max_nodes: int | None,
) -> dict[str, Any]:
    if isinstance(baseline_methods, str):
        methods = [method.strip() for method in baseline_methods.split(",") if method.strip()]
    else:
        methods = list(baseline_methods or [])
    if not methods:
        return {}
    if baseline_max_nodes is not None and baseline_max_nodes > 0 and original.num_nodes > baseline_max_nodes:
        return {
            method: {
                "status": "skipped",
                "reason": "node_count_exceeds_limit",
                "max_nodes": int(baseline_max_nodes),
                "original_nodes": int(original.num_nodes),
            }
            for method in methods
        }
    max_pairs = int(np.sum(actual_assignment.cluster_sizes() == 2))
    comparison: dict[str, Any] = {}
    for offset, method in enumerate(methods):
        baseline_assignment = _baseline_assignment(
            original,
            method,
            max_pairs=max_pairs,
            seed=int(seed) + 104729 * offset,
            relation_weights=relation_weights,
            dim=Z.shape[1],
        )
        baseline_coarse = coarsen_graph(original, baseline_assignment)
        baseline_matched_pairs = int(np.sum(baseline_assignment.cluster_sizes() == 2))
        baseline_metrics = compute_spectral_diagnostics(
            original,
            baseline_coarse,
            baseline_assignment,
            seed=seed,
            num_signals=Z.shape[1],
            smoothing_steps=smoothing_steps,
            relation_weights=relation_weights,
            Z=Z,
            baseline_methods=None,
            exact_eigenvalue_max_nodes=None,
        )
        comparison[method] = {
            "status": "computed",
            "coarse_nodes": int(baseline_coarse.num_nodes),
            "matched_pairs": baseline_matched_pairs,
            "dirichlet_energy_relative_error": baseline_metrics[
                "dirichlet_energy_relative_error"
            ],
            "relation_weighted_fused_energy_relative_error": baseline_metrics[
                "relation_weighted_fused_energy_relative_error"
            ],
            "chebheat_sketch_inner_product_relative_error": baseline_metrics[
                "chebheat_sketch_inner_product_relative_error"
            ],
        }
    return comparison


def compute_spectral_diagnostics(
    original: HeteroGraph,
    coarse: HeteroGraph,
    assignment: Assignment,
    seed: int = 12345,
    num_signals: int = 4,
    smoothing_steps: int = 1,
    relation_weights: dict[int, float] | None = None,
    Z: np.ndarray | None = None,
    Z_c: np.ndarray | None = None,
    exact_eigenvalue_max_nodes: int | None = None,
    baseline_methods: str | list[str] | tuple[str, ...] | None = None,
    baseline_max_nodes: int | None = None,
) -> dict[str, Any]:
    """Compute sparse, sketch-based spectral diagnostics for one coarsening level."""

    if Z is None:
        rng = np.random.default_rng(int(seed))
        Z = rng.standard_normal((original.num_nodes, int(num_signals))).astype(np.float32)
    else:
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z[:, None]
    if Z.shape[0] != original.num_nodes:
        raise ValueError("Z must have one row per original node")
    if Z.shape[1] > int(num_signals):
        Z = Z[:, : max(int(num_signals), 1)]

    original_signals = _smooth(original, Z, smoothing_steps, relation_weights)
    if Z_c is None:
        coarse_seed = _aggregate_signals(Z, assignment)
    else:
        coarse_seed = np.asarray(Z_c, dtype=np.float32)
        if coarse_seed.ndim == 1:
            coarse_seed = coarse_seed[:, None]
        if coarse_seed.shape[0] != coarse.num_nodes:
            raise ValueError("Z_c must have one row per coarse node")
    if coarse_seed.shape[1] > Z.shape[1]:
        coarse_seed = coarse_seed[:, : Z.shape[1]]
    coarse_signals = _smooth(coarse, coarse_seed, smoothing_steps, relation_weights)

    original_relation_energy: dict[str, float] = {}
    coarse_relation_energy: dict[str, float] = {}
    relation_relative_errors: dict[str, float] = {}
    for relation_id in sorted(set(original.relations) | set(coarse.relations)):
        before = (
            _relation_energy(original, relation_id, original_signals)
            if relation_id in original.relations
            else 0.0
        )
        after = (
            _relation_energy(coarse, relation_id, coarse_signals)
            if relation_id in coarse.relations
            else 0.0
        )
        original_relation_energy[str(relation_id)] = before
        coarse_relation_energy[str(relation_id)] = after
        relation_relative_errors[str(relation_id)] = _relative_error(before, after)

    energy_before = dirichlet_energy(original, original_signals)
    energy_after = dirichlet_energy(coarse, coarse_signals)
    fused_before = _fused_energy(original, original_signals, relation_weights)
    fused_after = _fused_energy(coarse, coarse_signals, relation_weights)
    weighted_fused_before = _relation_weighted_fused_energy(
        original,
        original_signals,
        relation_weights,
    )
    weighted_fused_after = _relation_weighted_fused_energy(
        coarse,
        coarse_signals,
        relation_weights,
    )

    diagnostics: dict[str, Any] = {
        "num_signals": int(original_signals.shape[1]),
        "smoothing_steps": int(max(smoothing_steps, 0)),
        "dirichlet_energy_before": float(energy_before),
        "dirichlet_energy_after": float(energy_after),
        "dirichlet_energy_relative_error": _relative_error(energy_before, energy_after),
        "sketch_dirichlet_energy_before": float(energy_before),
        "sketch_dirichlet_energy_after": float(energy_after),
        "sketch_dirichlet_energy_relative_error": _relative_error(energy_before, energy_after),
        "relation_energy_before": original_relation_energy,
        "relation_energy_after": coarse_relation_energy,
        "relation_energy_relative_error": relation_relative_errors,
        "relation_energy_relative_error_max": float(max(relation_relative_errors.values(), default=0.0)),
        "fused_sketch_energy_before": float(fused_before),
        "fused_sketch_energy_after": float(fused_after),
        "fused_sketch_energy_relative_error": _relative_error(fused_before, fused_after),
        "relation_weighted_fused_energy_before": float(weighted_fused_before),
        "relation_weighted_fused_energy_after": float(weighted_fused_after),
        "relation_weighted_fused_energy_relative_error": _relative_error(
            weighted_fused_before,
            weighted_fused_after,
        ),
        "chebheat_sketch_inner_product_relative_error": _inner_product_relative_error(
            original_signals,
            coarse_signals,
        ),
    }
    diagnostics["sketch_inner_product_relative_error"] = diagnostics[
        "chebheat_sketch_inner_product_relative_error"
    ]
    eigen_sanity = _exact_eigenvalue_sanity(
        original,
        coarse,
        relation_weights,
        exact_eigenvalue_max_nodes,
    )
    if eigen_sanity is not None:
        diagnostics["exact_eigenvalue_sanity"] = eigen_sanity
    diagnostics["baseline_comparison"] = _baseline_comparison(
        original,
        assignment,
        seed,
        relation_weights,
        Z,
        smoothing_steps,
        baseline_methods,
        baseline_max_nodes,
    )
    return diagnostics
