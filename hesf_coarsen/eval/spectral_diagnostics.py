from __future__ import annotations

from typing import Any

import numpy as np

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


def _inner_product_relative_error(Z: np.ndarray, Z_c: np.ndarray) -> float:
    q = min(Z.shape[1], Z_c.shape[1])
    if q == 0:
        return 0.0
    before = Z[:, :q].T @ Z[:, :q]
    after = Z_c[:, :q].T @ Z_c[:, :q]
    scale = max(float(np.linalg.norm(before, ord="fro")), 1e-12)
    return float(np.linalg.norm(before - after, ord="fro") / scale)


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

    original_signals = _smooth(original, Z, smoothing_steps, relation_weights)
    if Z_c is None:
        coarse_seed = _aggregate_signals(original_signals, assignment)
    else:
        coarse_seed = np.asarray(Z_c, dtype=np.float32)
        if coarse_seed.ndim == 1:
            coarse_seed = coarse_seed[:, None]
        if coarse_seed.shape[0] != coarse.num_nodes:
            raise ValueError("Z_c must have one row per coarse node")
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

    diagnostics: dict[str, Any] = {
        "num_signals": int(original_signals.shape[1]),
        "smoothing_steps": int(max(smoothing_steps, 0)),
        "dirichlet_energy_before": float(energy_before),
        "dirichlet_energy_after": float(energy_after),
        "dirichlet_energy_relative_error": _relative_error(energy_before, energy_after),
        "relation_energy_before": original_relation_energy,
        "relation_energy_after": coarse_relation_energy,
        "relation_energy_relative_error": relation_relative_errors,
        "relation_energy_relative_error_max": float(max(relation_relative_errors.values(), default=0.0)),
        "fused_sketch_energy_before": float(fused_before),
        "fused_sketch_energy_after": float(fused_after),
        "fused_sketch_energy_relative_error": _relative_error(fused_before, fused_after),
    }
    if Z_c is not None:
        diagnostics["sketch_inner_product_relative_error"] = _inner_product_relative_error(
            original_signals,
            coarse_signals,
        )
    return diagnostics
