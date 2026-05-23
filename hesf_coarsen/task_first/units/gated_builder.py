from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.condensation import build_selected_support_graph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig
from hesf_coarsen.task_first.units.base import SupportUnit


def select_units_under_budget(
    units: list[SupportUnit],
    *,
    support_count: int,
    requested_support_ratio: float,
    allow_underfill: bool = True,
    min_score: float | None = None,
) -> list[SupportUnit]:
    budget = max(0, int(np.ceil(int(support_count) * float(requested_support_ratio) - 1.0e-12)))
    selected: list[SupportUnit] = []
    used: set[int] = set()
    for unit in sorted(units, key=lambda item: (-float(item.metadata.get("score", 0.0)), -int(item.member_count), str(item.source), str(item.unit_id))):
        score = float(unit.metadata.get("score", 0.0) or 0.0)
        if min_score is not None and score < float(min_score):
            continue
        new_nodes = [int(node) for node in unit.member_nodes if int(node) not in used]
        if not new_nodes:
            continue
        if len(used) + len(new_nodes) > budget:
            if bool(allow_underfill):
                continue
            break
        selected.append(unit)
        used.update(new_nodes)
        if len(used) >= budget:
            break
    return selected


def selected_member_nodes(units: list[SupportUnit]) -> np.ndarray:
    return np.asarray(sorted({int(node) for unit in units for node in unit.member_nodes}), dtype=np.int64)


def build_graph_from_units(
    graph: HeteroGraph,
    units: list[SupportUnit],
    *,
    target_type: int,
    selector_config: SupportSelectorConfig | None = None,
    support_features: dict[str, Any] | None = None,
) -> tuple[HeteroGraph, np.ndarray, dict[str, Any]]:
    cfg = selector_config or SupportSelectorConfig(
        selector="teacher_topk",
        background_strategy="drop",
        allow_background_bucket=False,
        residual_prototype_mode="none",
        force_raw_bridge_nodes=False,
        force_raw_keep_high_degree_bridges=False,
        allow_proxy_fill=False,
    )
    cfg = replace(cfg, background_strategy="drop", allow_background_bucket=False, residual_prototype_mode="none")
    selected_nodes = selected_member_nodes(units)
    coarse, assignment_obj, diagnostics = build_selected_support_graph(
        graph,
        selected_nodes,
        cfg,
        target_node_type=int(target_type),
        support_features=support_features,
    )
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "selected_unit_count": int(len(units)),
            "selected_member_node_count": int(len(selected_nodes)),
            "selected_unit_sources": sorted({str(unit.source) for unit in units}),
        }
    )
    return coarse, np.asarray(assignment_obj.assignment, dtype=np.int64), diagnostics
