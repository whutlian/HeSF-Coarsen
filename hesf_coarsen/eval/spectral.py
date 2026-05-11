from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph


def dirichlet_energy(graph: HeteroGraph, Z: np.ndarray) -> float:
    """Approximate relation-wise Dirichlet energy without dense matrices."""

    Z = Z.astype(np.float32, copy=False)
    energy = 0.0
    for rel in graph.relations.values():
        diff = Z[rel.src] - Z[rel.dst]
        energy += float(np.sum(rel.weight * np.sum(diff * diff, axis=1)))
    return energy
