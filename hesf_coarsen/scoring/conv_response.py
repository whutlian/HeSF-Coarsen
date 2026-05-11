from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing


def compute_conv_response_sketch(
    graph: HeteroGraph,
    H: np.ndarray,
    relation_weights: dict[int, float] | None = None,
) -> np.ndarray:
    return apply_fused_smoothing(graph, H.astype(np.float32), relation_weights)
