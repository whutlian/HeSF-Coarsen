from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence


SURVIVOR = "KEEP_ACCURACY_BRANCH_MINIMAL"
DROP_HYBRID = "DROP_HYBRID_B_KEEP_A1_A2_EXPLORATORY"
DROP_ALL = "DROP_ENTIRE_ACCURACY_BRANCH"
SURVIVOR_METHODS = {"A1_target_preserve", "A2_hybridA_keepall"}
COMPARATOR_METHODS = {"flatten-sum_keep_target", "H6_keep_target", "TypedHash-ChebHeat_keep_target"}
FAITHFUL = {"official", "faithful_reproduction"}


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("method", row.get("variant", "")))


def _eligible(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("eval_mode")) == "real_full_target_inference"
        and str(row.get("model_fidelity")) in FAITHFUL
        and _as_float(row.get("macro_f1")) is not None
    )


def decide_accuracy_branch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if _eligible(row)]
    if not eligible:
        return {
            "decision": DROP_ALL,
            "reason": "No official or faithful real full-target inference rows are available.",
            "wins_vs_internal_comparator": 0,
        }

    by_dataset_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in eligible:
        value = _as_float(row.get("macro_f1"))
        if value is None:
            continue
        by_dataset_method[(str(row.get("dataset")), _method(row))].append(value)

    wins = 0
    datasets = sorted({dataset for dataset, _method_name in by_dataset_method})
    for dataset in datasets:
        survivor_scores = [
            mean(values)
            for (row_dataset, method), values in by_dataset_method.items()
            if row_dataset == dataset and method in SURVIVOR_METHODS and values
        ]
        comparator_scores = [
            mean(values)
            for (row_dataset, method), values in by_dataset_method.items()
            if row_dataset == dataset and method in COMPARATOR_METHODS and values
        ]
        if survivor_scores and comparator_scores and max(survivor_scores) >= max(comparator_scores):
            wins += 1

    survivor_values = [
        value
        for (dataset, method), values in by_dataset_method.items()
        for value in values
        if method in SURVIVOR_METHODS
    ]
    comparator_values = [
        value
        for (dataset, method), values in by_dataset_method.items()
        for value in values
        if method in COMPARATOR_METHODS
    ]
    survivor_mean = mean(survivor_values) if survivor_values else 0.0
    comparator_mean = mean(comparator_values) if comparator_values else 0.0
    if wins >= 2 and survivor_mean >= comparator_mean - 1.0e-12:
        decision = SURVIVOR
        reason = "A1/A2 match or beat the strongest keep-target comparator on at least 2 of 3 datasets."
    elif wins > 0:
        decision = DROP_HYBRID
        reason = "A1/A2 have partial signal, but not enough stable wins for the accuracy branch mainline."
    else:
        decision = DROP_ALL
        reason = "A1/A2 do not beat keep-target comparators under eligible faithful rows."
    return {
        "decision": decision,
        "reason": reason,
        "wins_vs_internal_comparator": int(wins),
        "survivor_macro_f1_mean": float(survivor_mean),
        "comparator_macro_f1_mean": float(comparator_mean),
        "eligible_rows": int(len(eligible)),
    }
