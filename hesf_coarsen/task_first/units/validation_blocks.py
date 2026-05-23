from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.validation_selector import build_support_block_keys, group_support_by_block
from hesf_coarsen.task_first.units.base import SupportUnit, build_unit_structure_index, unit_structure_from_index


def extract_validation_block_units(
    graph: HeteroGraph,
    support_features: dict[str, Any],
    block_key_mode: str = "class_anchor_relation",
    target_type: int | None = None,
    labels: np.ndarray | None = None,
    splits: Mapping[str, np.ndarray] | None = None,
) -> list[SupportUnit]:
    if target_type is None:
        target_type = int(support_features.get("target_node_type", 0))
    support_nodes = np.asarray(support_features["support_nodes"], dtype=np.int64)
    block_keys = build_support_block_keys(support_features, mode=str(block_key_mode))
    groups = group_support_by_block(block_keys)
    index = build_unit_structure_index(graph, target_type=int(target_type), labels=labels, splits=splits)
    units: list[SupportUnit] = []
    for key, local_indices in sorted(groups.items(), key=lambda item: repr(item[0])):
        members = support_nodes[np.asarray(local_indices, dtype=np.int64)]
        units.append(
            unit_structure_from_index(
                graph,
                members,
                source="validation_block",
                unit_id=repr(tuple(int(value) for value in key)),
                index=index,
                metadata={"block_key": repr(tuple(int(value) for value in key)), "unit_family": "validation_block"},
            )
        )
    return units
