from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.sketch.relation_weights import compute_relation_weights


def relation_volume(graph: HeteroGraph, relation_id: int) -> float:
    result = compute_relation_weights(
        graph,
        {"fusion": {"relation_weighting": {"method": "volume", "eta": 1.0}}},
    )
    return float(result.volume_estimates[int(relation_id)])


def relation_energy_estimate(
    graph: HeteroGraph,
    relation_id: int,
    signals: np.ndarray,
) -> float:
    result = compute_relation_weights(
        graph,
        {"fusion": {"relation_weighting": {"method": "inverse_energy", "eta": 0.0}}},
        basis=signals,
    )
    return float(result.energy_estimates[int(relation_id)])


def compute_relation_fusion_weights(
    graph: HeteroGraph,
    signals: np.ndarray,
    config: dict | None = None,
) -> dict[int, float]:
    """Backward-compatible wrapper for older callers."""

    return compute_relation_weights(graph, config or {}, basis=signals).weights
