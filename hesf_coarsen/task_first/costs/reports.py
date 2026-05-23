from __future__ import annotations

from typing import Any, Sequence

from hesf_coarsen.task_first.costs.accounting import CompressionCost, cost_to_row


def cost_rows(costs: Sequence[CompressionCost], *, cost_axis_used: str = "total_storage_ratio_vs_full_stc") -> list[dict[str, Any]]:
    return [cost_to_row(cost, cost_axis_used=cost_axis_used) for cost in costs]
