from __future__ import annotations

import json
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph


def _smooth_bidirectional(graph: HeteroGraph, signals: np.ndarray, steps: int = 3) -> np.ndarray:
    values = signals.astype(np.float32, copy=True)
    for _ in range(max(int(steps), 1)):
        accum = values.copy()
        counts = np.ones((graph.num_nodes, 1), dtype=np.float32)
        for rel in graph.relations.values():
            row_sum = np.bincount(rel.src, weights=rel.weight.astype(np.float64), minlength=graph.num_nodes).astype(np.float32)
            w = rel.weight.astype(np.float32, copy=False) / np.maximum(row_sum[rel.src], 1.0e-12)
            np.add.at(accum, rel.dst, values[rel.src] * w[:, None])
            np.add.at(counts, rel.dst, 1.0)
            np.add.at(accum, rel.src, values[rel.dst] * w[:, None])
            np.add.at(counts, rel.src, 1.0)
        values = accum / np.maximum(counts, 1.0)
    return values.astype(np.float32)


def _aggregate(values: np.ndarray, assignment: Assignment) -> np.ndarray:
    out = np.zeros((assignment.num_supernodes, values.shape[1]), dtype=np.float32)
    np.add.at(out, assignment.assignment, values.astype(np.float32, copy=False))
    counts = assignment.cluster_sizes().astype(np.float32)
    out /= np.maximum(counts[:, None], 1.0)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    flat_a = a.reshape(-1).astype(np.float64)
    flat_b = b.reshape(-1).astype(np.float64)
    return float(np.dot(flat_a, flat_b) / max(float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b)), 1.0e-12))


def evaluate_lowpass_signal_reconstruction(
    original_graph: HeteroGraph,
    coarse_graph: HeteroGraph,
    assignment: Assignment,
    seed: int = 12345,
    num_signals: int = 8,
    smoothing_steps: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((original_graph.num_nodes, int(num_signals))).astype(np.float32)
    target = _smooth_bidirectional(original_graph, noise, steps=int(smoothing_steps))
    coarse_seed = _aggregate(noise, assignment)
    coarse_signal = _smooth_bidirectional(coarse_graph, coarse_seed, steps=int(smoothing_steps))
    lifted = coarse_signal[assignment.assignment]
    diff = target - lifted
    per_type = {}
    for type_id in sorted(np.unique(original_graph.node_type)):
        nodes = np.flatnonzero(original_graph.node_type == int(type_id))
        per_type[str(int(type_id))] = float(np.mean(diff[nodes] * diff[nodes])) if len(nodes) else 0.0
    corr = 0.0
    if target.size and np.std(target.reshape(-1)) > 1.0e-12 and np.std(lifted.reshape(-1)) > 1.0e-12:
        corr = float(np.corrcoef(target.reshape(-1), lifted.reshape(-1))[0, 1])
    return {
        "task": "lowpass_signal_reconstruction",
        "signal_mse": float(np.mean(diff * diff)),
        "signal_mae": float(np.mean(np.abs(diff))),
        "signal_correlation": corr,
        "signal_cosine": _cosine(target, lifted),
        "per_type_signal_mse": json.dumps(per_type, sort_keys=True),
    }


def _accuracy(labels: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(labels).reshape(-1) == np.asarray(pred).reshape(-1)))


def evaluate_feature_free_label_propagation(
    original_graph: HeteroGraph,
    coarse_graph: HeteroGraph,
    assignment: Assignment,
    seed: int = 12345,
    num_classes: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    logits = _smooth_bidirectional(
        original_graph,
        rng.standard_normal((original_graph.num_nodes, int(num_classes))).astype(np.float32),
        steps=4,
    )
    labels = np.argmax(logits, axis=1).astype(np.int64)
    coarse_scores = _aggregate(np.eye(int(num_classes), dtype=np.float32)[labels], assignment)
    lifted_scores = coarse_scores[assignment.assignment]
    curves = {}
    refined = lifted_scores
    for epoch in (0, 1, 3, 5):
        while len(curves) == 0 and epoch == 0:
            curves[epoch] = _accuracy(labels, np.argmax(refined, axis=1))
            break
        while max(curves, default=0) < epoch:
            refined = _smooth_bidirectional(original_graph, refined, steps=1)
            curves[max(curves, default=0) + 1] = _accuracy(labels, np.argmax(refined, axis=1))
        curves[epoch] = _accuracy(labels, np.argmax(refined, axis=1))
    values = [curves[e] for e in (0, 1, 3, 5)]
    return {
        "task": "feature_free_label_propagation",
        "projected": float(curves[0]),
        "refined@0": float(curves[0]),
        "refined@1": float(curves[1]),
        "refined@3": float(curves[3]),
        "refined@5": float(curves[5]),
        "best": float(max(values)),
        "AUC": float(np.mean(values)),
    }
