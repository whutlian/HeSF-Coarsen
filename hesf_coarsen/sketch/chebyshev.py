from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.progress import progress_iter
from hesf_coarsen.sketch.operators import apply_fused_operator


def chebyshev_heat_coefficients(
    heat_time: float,
    order: int,
    quadrature_points: int | None = None,
) -> np.ndarray:
    order = max(int(order), 0)
    M = max(int(quadrature_points or 0), 128, 8 * max(order, 1))
    j = np.arange(M, dtype=np.float64)
    theta = np.pi * (j + 0.5) / M
    x = np.cos(theta)
    values = np.exp(-float(heat_time) * (x + 1.0))
    coeffs = np.empty(order + 1, dtype=np.float64)
    coeffs[0] = np.sum(values) / M
    for k in range(1, order + 1):
        coeffs[k] = (2.0 / M) * np.sum(values * np.cos(k * theta))
    return coeffs.astype(np.float32)


def chebyshev_heat_filter(
    graph: HeteroGraph,
    basis: np.ndarray,
    relation_weights: dict[int, float],
    *,
    heat_time: float,
    order: int,
    quadrature_points: int | None = None,
    symmetric_relation_operator: bool = True,
    reverse_relation_policy: str = "include_all",
    progress_config: dict | None = None,
    progress_desc: str = "chebyshev recurrence",
) -> np.ndarray:
    """Approximate exp(-t L_F) basis by Chebyshev recurrence."""

    B = np.asarray(basis, dtype=np.float32)
    if B.ndim == 1:
        B = B[:, None]
    if B.shape[0] != graph.num_nodes:
        raise ValueError("basis must have one row per graph node")

    coeffs = chebyshev_heat_coefficients(heat_time, order, quadrature_points)

    def apply_scaled(H: np.ndarray) -> np.ndarray:
        return -apply_fused_operator(
            graph,
            H,
            relation_weights,
            symmetric_relation_operator=symmetric_relation_operator,
            reverse_relation_policy=reverse_relation_policy,
        )

    T_prev = B.astype(np.float32, copy=False)
    out = coeffs[0] * T_prev
    T_curr: np.ndarray | None = None
    recurrence_steps = max(len(coeffs) - 1, 0)
    for coefficient_index in progress_iter(
        range(1, len(coeffs)),
        total=recurrence_steps,
        desc=progress_desc,
        config=progress_config,
        unit="step",
    ):
        if coefficient_index == 1:
            T_curr = apply_scaled(T_prev)
            out = out + coeffs[coefficient_index] * T_curr
        else:
            if T_curr is None:
                raise RuntimeError("Chebyshev recurrence state was not initialized")
            T_next = 2.0 * apply_scaled(T_curr) - T_prev
            out = out + coeffs[coefficient_index] * T_next
            T_prev, T_curr = T_curr, T_next
    out = out.astype(np.float32, copy=False)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Chebyshev heat filter produced NaN or Inf")
    return out
