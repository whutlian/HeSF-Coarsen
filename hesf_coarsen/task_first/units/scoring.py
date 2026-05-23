from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from hesf_coarsen.task_first.units.base import SupportUnit


def _entropy_score(counts: Iterable[float]) -> float:
    arr = np.asarray([float(value) for value in counts if float(value) > 0.0], dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    probs = arr / max(float(np.sum(arr)), 1.0e-12)
    return float(-np.sum(probs * np.log2(probs)) / max(math.log2(int(arr.size)), 1.0))


def score_units(
    units: list[SupportUnit],
    *,
    validation_accuracy_gain_by_unit: dict[str, float] | None = None,
    validation_macro_gain_by_unit: dict[str, float] | None = None,
    lambda_acc: float = 1.0,
    lambda_macro: float = 0.5,
    lambda_edge: float = 0.1,
    lambda_anchor: float = 0.1,
    lambda_relation: float = 0.1,
    lambda_class_balance: float = 0.1,
) -> list[SupportUnit]:
    acc_gain = validation_accuracy_gain_by_unit or {}
    macro_gain = validation_macro_gain_by_unit or {}
    max_edge = max([float(unit.edge_mass) for unit in units] + [1.0])
    scored: list[SupportUnit] = []
    for unit in units:
        normalized_edge_mass = float(unit.edge_mass) / max(max_edge, 1.0e-12)
        relation_channel_diversity = _entropy_score(unit.relation_profile.values())
        class_balance_score = _entropy_score(unit.class_footprint.values())
        validation_accuracy_gain = float(acc_gain.get(unit.unit_id, unit.metadata.get("validation_accuracy_gain", 0.0)) or 0.0)
        validation_macro_gain = float(macro_gain.get(unit.unit_id, unit.metadata.get("validation_macro_gain", 0.0)) or 0.0)
        score = (
            float(lambda_acc) * validation_accuracy_gain
            + float(lambda_macro) * validation_macro_gain
            + float(lambda_edge) * normalized_edge_mass
            + float(lambda_anchor) * float(unit.target_anchor_coverage)
            + float(lambda_relation) * relation_channel_diversity
            + float(lambda_class_balance) * class_balance_score
        )
        meta: dict[str, Any] = dict(unit.metadata)
        meta.update(
            {
                "validation_accuracy_gain": validation_accuracy_gain,
                "validation_macro_gain": validation_macro_gain,
                "normalized_edge_mass": normalized_edge_mass,
                "relation_channel_diversity": relation_channel_diversity,
                "class_balance_score": class_balance_score,
                "score": float(score),
                "scoring_formula": "1.0*validation_accuracy_gain + 0.5*validation_macro_gain + 0.1*normalized_edge_mass + 0.1*target_anchor_coverage + 0.1*relation_channel_diversity + 0.1*class_balance_score",
            }
        )
        scored.append(replace(unit, metadata=meta))
    return sorted(scored, key=lambda item: (-float(item.metadata.get("score", 0.0)), -int(item.member_count), str(item.source), str(item.unit_id)))
