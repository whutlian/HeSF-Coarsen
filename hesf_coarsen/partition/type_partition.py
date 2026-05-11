from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph


def default_partition(graph: HeteroGraph) -> np.ndarray:
    if graph.partitions is not None:
        return graph.partitions.astype(np.int32, copy=False)
    return np.zeros(graph.num_nodes, dtype=np.int32)
