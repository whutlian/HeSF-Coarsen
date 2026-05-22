from __future__ import annotations

from typing import Any

import numpy as np


DIAGNOSTIC_ONLY_METHODS = {
    "HeSF-SS-real-validation-no-fallback",
    "HeSF-SS-full-residual-prototype-upperbound",
}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _ratio(count: int, total: int) -> float:
    return float(int(count) / max(int(total), 1))


def compute_gate17_3_budget_fields(
    *,
    original_support_nodes: int,
    requested_support_ratio: float,
    selected_raw_support_count: int,
    forced_raw_support_count: int = 0,
    prototype_background_count: int = 0,
    prototype_member_count_sum: int = 0,
    prototype_member_budget_total: int | None = None,
    prototype_budget_fraction: float = 0.10,
    represented_context_slack: float = 0.10,
    full_residual_upperbound: bool = False,
    method: str = "",
    no_test_leakage: bool = True,
) -> dict[str, Any]:
    original_support = max(0, _int(original_support_nodes))
    requested_ratio = float(requested_support_ratio)
    selected_raw = max(0, _int(selected_raw_support_count))
    forced_raw = max(0, _int(forced_raw_support_count))
    prototype_count = max(0, _int(prototype_background_count))
    prototype_members = max(0, _int(prototype_member_count_sum))
    if prototype_member_budget_total is None:
        member_budget_total = max(0, int(np.floor(float(prototype_budget_fraction) * original_support)))
    else:
        member_budget_total = max(0, _int(prototype_member_budget_total))
    node_budget_count = int(selected_raw + forced_raw + prototype_count)
    represented_context_count = int(selected_raw + forced_raw + prototype_members)
    node_budget_ratio = _ratio(node_budget_count, original_support)
    represented_context_ratio = _ratio(represented_context_count, original_support)
    node_budget_exact_match = abs(float(node_budget_ratio) - requested_ratio) <= 0.02
    represented_context_bound = requested_ratio + float(represented_context_slack)
    represented_bounded = bool(represented_context_ratio <= represented_context_bound + 1.0e-12)
    diagnostic_only = (
        bool(full_residual_upperbound)
        or "upperbound" in str(method).lower()
        or "full-residual" in str(method).lower()
        or str(method) in DIAGNOSTIC_ONLY_METHODS
    )
    eligible = bool(
        str(method).startswith("HeSF-SS")
        and not diagnostic_only
        and node_budget_exact_match
        and represented_bounded
        and bool(no_test_leakage)
    )
    return {
        "node_budget_count": int(node_budget_count),
        "node_budget_ratio": float(node_budget_ratio),
        "node_budget_exact_match": bool(node_budget_exact_match),
        "represented_context_count": int(represented_context_count),
        "represented_context_ratio": float(represented_context_ratio),
        "represented_context_exact_or_bounded": bool(represented_bounded),
        "represented_context_leak_ratio": float(round(represented_context_ratio - requested_ratio, 12)),
        "prototype_member_budget_total": int(member_budget_total),
        "prototype_member_budget_used": int(prototype_members),
        "prototype_budget_fraction": float(prototype_budget_fraction),
        "full_residual_upperbound": bool(full_residual_upperbound),
        "eligible_for_main_decision": bool(eligible),
    }
