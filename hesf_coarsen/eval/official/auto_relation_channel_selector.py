from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def build_relation_channel_keep_plan(
    *,
    dataset: str,
    target_type: str,
    relation_edge_counts: Mapping[str, int],
    mode: str = "coverage_greedy",
) -> dict[str, Any]:
    channels = []
    total = max(1, sum(int(v) for v in relation_edge_counts.values()))
    for relation, count in sorted(relation_edge_counts.items()):
        cost = float(int(count) / total)
        keep_ratio = 1.0 if cost <= 0.20 else 0.5
        channels.append(
            {
                "channel_name": str(relation),
                "relations": [str(relation)],
                "keep_ratio_by_direction": {str(relation): keep_ratio},
                "estimated_utility": float(1.0 - cost),
                "estimated_cost": cost,
                "decision_reason": f"{mode}: keep_ratio={keep_ratio} from relation cost share {cost:.4f}",
            }
        )
    return {
        "dataset": str(dataset).upper(),
        "target_type": str(target_type),
        "channels": channels,
        "selected_by": str(mode),
        "used_test_data": False,
    }


def plan_to_jsonl_row(plan: Mapping[str, Any]) -> str:
    return json.dumps(dict(plan), sort_keys=True)
