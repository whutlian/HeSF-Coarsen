from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.sketch.operators import apply_fused_operator


def compute_conv_response_sketch(
    graph: HeteroGraph,
    H: np.ndarray,
    relation_weights: dict[int, float] | None = None,
    *,
    operator: str = "fused_operator",
    relation_operator_mode: str = "relationwise",
) -> np.ndarray:
    """Compute a ConvMatch-style relation convolution response sketch.

    The default explicit path applies C = sum_r alpha_r S_r H through the
    sketch fused-operator apply-function. ``lazy_smoothing`` keeps the previous
    low-pass baseline with a self term for comparisons.
    """

    H = H.astype(np.float32, copy=False)
    operator = str(operator)
    if operator == "fused_operator":
        return apply_fused_operator(
            graph,
            H,
            relation_weights,
            relation_operator_mode=relation_operator_mode,
        )
    if operator == "lazy_smoothing":
        return apply_fused_smoothing(
            graph,
            H,
            relation_weights,
            relation_operator_mode=relation_operator_mode,
        )
    raise ValueError(f"unsupported conv response operator: {operator}")
