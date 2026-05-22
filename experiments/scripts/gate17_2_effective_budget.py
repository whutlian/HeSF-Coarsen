from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _ratio(count: int, total: int) -> float:
    return float(int(count) / max(int(total), 1))


def compute_effective_budget_fields(
    *,
    original_support_nodes: int,
    requested_support_ratio: float,
    selected_support_count: int,
    graph_diagnostics: Mapping[str, Any] | None = None,
    candidate_allclose_to_full: bool = False,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    diagnostics = dict(graph_diagnostics or {})
    original_support = max(0, _int(original_support_nodes))
    requested_ratio = float(requested_support_ratio)
    requested_count = int(np.ceil(original_support * requested_ratio - 1.0e-12)) if requested_ratio > 0.0 else 0
    selected_budget_count = max(0, _int(selected_support_count))
    forced_raw_count = max(
        0,
        _int(
            diagnostics.get(
                "forced_raw_support_count",
                diagnostics.get("forced_raw_bridge_count", 0),
            )
        ),
    )
    prototype_background_count = max(0, _int(diagnostics.get("prototype_background_count", 0)))
    prototype_member_count_sum = max(0, _int(diagnostics.get("prototype_member_count_sum", 0)))
    if prototype_member_count_sum == 0 and _int(diagnostics.get("unselected_support_count", 0)) > 0:
        prototype_member_count_sum = max(0, _int(diagnostics.get("unselected_support_count", 0)))

    effective_raw_support_count = int(selected_budget_count + forced_raw_count)
    effective_support_node_count = int(effective_raw_support_count + prototype_background_count)
    represented_support_context_count = int(effective_raw_support_count + prototype_member_count_sum)
    effective_support_node_ratio = _ratio(effective_support_node_count, original_support)
    represented_support_context_ratio = _ratio(represented_support_context_count, original_support)
    effective_raw_support_ratio = _ratio(effective_raw_support_count, original_support)
    budget_leak_ratio = float(round(effective_support_node_ratio - requested_ratio, 12))
    represented_context_leak_ratio = float(round(represented_support_context_ratio - requested_ratio, 12))
    forced_raw_budget_leak_ratio = _ratio(forced_raw_count, original_support)
    prototype_node_budget_leak_ratio = _ratio(prototype_background_count, original_support)
    exact = abs(float(budget_leak_ratio)) <= float(tolerance)

    return {
        "original_support_nodes": int(original_support),
        "requested_support_ratio": requested_ratio,
        "requested_support_count": int(requested_count),
        "selected_budget_support_count": int(selected_budget_count),
        "selected_support_count": int(selected_budget_count),
        "forced_raw_support_count": int(forced_raw_count),
        "prototype_background_count": int(prototype_background_count),
        "prototype_member_count_sum": int(prototype_member_count_sum),
        "effective_raw_support_count": int(effective_raw_support_count),
        "effective_raw_support_ratio": float(effective_raw_support_ratio),
        "effective_support_node_count": int(effective_support_node_count),
        "effective_support_node_ratio": float(effective_support_node_ratio),
        "represented_support_context_count": int(represented_support_context_count),
        "represented_support_context_ratio": float(represented_support_context_ratio),
        "budget_leak_ratio": float(budget_leak_ratio),
        "represented_context_leak_ratio": float(represented_context_leak_ratio),
        "forced_raw_budget_leak_ratio": float(forced_raw_budget_leak_ratio),
        "prototype_node_budget_leak_ratio": float(prototype_node_budget_leak_ratio),
        "effective_budget_exact_match": bool(exact),
        "candidate_allclose_to_full": bool(candidate_allclose_to_full),
    }
