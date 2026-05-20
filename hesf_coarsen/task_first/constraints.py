from __future__ import annotations

from typing import Any

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.support_purity import merge_is_purity_allowed


def allow_task_first_merge(
    graph: HeteroGraph,
    u: int,
    v: int,
    assignment: Any,
    state,
    cfg: TaskFirstConfig,
) -> bool:
    del assignment
    u = int(u)
    v = int(v)
    if u == v:
        return False
    if graph.node_type[u] == int(cfg.target_node_type) or graph.node_type[v] == int(cfg.target_node_type):
        return False
    if cfg.same_type_only and graph.node_type[u] != graph.node_type[v]:
        return False
    if cfg.same_partition_only and graph.partitions is not None:
        partitions = np.asarray(graph.partitions)
        if partitions[u] != partitions[v]:
            return False
    return merge_is_purity_allowed(u, v, state, cfg)
