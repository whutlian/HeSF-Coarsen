from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RelationStats:
    relation_id: int | str
    relation_name: str
    relation_pair_name: str | None
    src_type: str
    dst_type: str
    full_edge_count: int
    candidate_edge_count: int
    min_edges: int


@dataclass(frozen=True)
class RelationBudgetAllocation:
    relation_id: int | str
    relation_name: str
    relation_pair_name: str | None
    requested_edges: int
    actual_edges: int
    requested_ratio_vs_candidate: float
    min_edges_constraint_active: bool


@dataclass(frozen=True)
class RelationPairRetentionSpec:
    pair_name: str
    forward_retention: float
    reverse_retention: float
    sampling_strategy: str
    min_edges: int = 1
    max_edges: int | None = None


@dataclass(frozen=True)
class ParsedRelationChannelSpec:
    raw_spec: str
    retention_by_relation: dict[str, float]
    pair_specs: tuple[RelationPairRetentionSpec, ...]
    sampling_strategy: str = "random"


RELATION_PAIR_DIRECTIONS = {
    "AP_PA": ("AP", "PA"),
    "PT_TP": ("PT", "TP"),
    "PV_VP": ("PV", "VP"),
}

PAIR_TOKEN_TO_PAIR = {
    "APPA": "AP_PA",
    "PTTP": "PT_TP",
    "PVVP": "PV_VP",
}


def _ratio_from_percent(value: str) -> float:
    return max(0.0, min(1.0, float(int(value)) / 100.0))


def parse_relation_channel_spec(spec: str, *, sampling_strategy: str = "random") -> ParsedRelationChannelSpec:
    raw = str(spec).strip()
    if raw.startswith("H6-relgrid-"):
        raw = raw[len("H6-relgrid-") :]
    if raw.startswith("H6-dir-"):
        raw = raw[len("H6-dir-") :]
    retention = {name: 0.0 for pair in RELATION_PAIR_DIRECTIONS.values() for name in pair}
    for token in [part for part in raw.split("-") if part]:
        pair_match = re.fullmatch(r"(APPA|PTTP|PVVP)(\d{1,3})", token)
        if pair_match:
            pair = PAIR_TOKEN_TO_PAIR[pair_match.group(1)]
            forward, reverse = RELATION_PAIR_DIRECTIONS[pair]
            ratio = _ratio_from_percent(pair_match.group(2))
            retention[forward] = ratio
            retention[reverse] = ratio
            continue
        direction_match = re.fullmatch(r"(AP|PA|PT|TP|PV|VP)(\d{1,3})", token)
        if direction_match:
            retention[direction_match.group(1)] = _ratio_from_percent(direction_match.group(2))
            continue
        raise ValueError(f"unsupported relation-channel token {token!r} in {spec!r}")
    pair_specs = []
    for pair, (forward, reverse) in RELATION_PAIR_DIRECTIONS.items():
        pair_specs.append(
            RelationPairRetentionSpec(
                pair_name=pair,
                forward_retention=float(retention[forward]),
                reverse_retention=float(retention[reverse]),
                sampling_strategy=str(sampling_strategy),
            )
        )
    return ParsedRelationChannelSpec(raw_spec=raw, retention_by_relation=retention, pair_specs=tuple(pair_specs), sampling_strategy=str(sampling_strategy))


def allocate_relation_channel_spec(
    relation_stats: list[RelationStats],
    parsed_spec: ParsedRelationChannelSpec,
    *,
    min_edges_per_relation: int = 1,
) -> list[RelationBudgetAllocation]:
    rows: list[RelationBudgetAllocation] = []
    for stat in relation_stats:
        candidate = max(0, int(stat.candidate_edge_count))
        requested = int(round(candidate * float(parsed_spec.retention_by_relation.get(str(stat.relation_name), 0.0))))
        minimum = min(candidate, max(int(min_edges_per_relation), int(stat.min_edges)))
        actual = min(candidate, max(minimum, requested))
        rows.append(
            RelationBudgetAllocation(
                relation_id=stat.relation_id,
                relation_name=str(stat.relation_name),
                relation_pair_name=stat.relation_pair_name,
                requested_edges=int(requested),
                actual_edges=int(actual),
                requested_ratio_vs_candidate=float(requested / max(candidate, 1)),
                min_edges_constraint_active=bool(actual != requested and candidate > 0),
            )
        )
    return rows


class RelationBudgetAllocator:
    def allocate(
        self,
        *,
        relation_stats: list[RelationStats],
        total_edge_budget: int,
        strategy: str,
        relation_pair_weights: dict[str, float] | None = None,
        min_edges_per_relation: int = 1,
        seed: int = 1,
        validation_feedback: dict[str, float] | None = None,
    ) -> list[RelationBudgetAllocation]:
        stats = list(relation_stats)
        if not stats:
            return []
        capacities = {stat.relation_id: max(0, int(stat.candidate_edge_count)) for stat in stats}
        mins = {
            stat.relation_id: min(
                capacities[stat.relation_id],
                max(int(min_edges_per_relation), int(stat.min_edges)),
            )
            for stat in stats
        }
        min_sum = int(sum(mins.values()))
        if int(total_edge_budget) < min_sum:
            raise ValueError(f"total_edge_budget={total_edge_budget} is below required relation minimum {min_sum}")
        budgets = dict(mins)
        remaining = min(int(total_edge_budget), int(sum(capacities.values()))) - min_sum
        weights = self._weights(
            stats=stats,
            strategy=strategy,
            relation_pair_weights=relation_pair_weights,
            seed=int(seed),
            validation_feedback=validation_feedback or {},
        )
        self._distribute_remaining(stats, budgets, capacities, weights, remaining)
        rows: list[RelationBudgetAllocation] = []
        for stat in stats:
            actual = int(min(capacities[stat.relation_id], budgets.get(stat.relation_id, 0)))
            rows.append(
                RelationBudgetAllocation(
                    relation_id=stat.relation_id,
                    relation_name=str(stat.relation_name),
                    relation_pair_name=stat.relation_pair_name,
                    requested_edges=actual,
                    actual_edges=actual,
                    requested_ratio_vs_candidate=float(actual / max(int(stat.candidate_edge_count), 1)),
                    min_edges_constraint_active=bool(actual <= mins[stat.relation_id] and capacities[stat.relation_id] > 0),
                )
            )
        return rows

    def _weights(
        self,
        *,
        stats: list[RelationStats],
        strategy: str,
        relation_pair_weights: Mapping[str, float] | None,
        seed: int,
        validation_feedback: Mapping[str, float],
    ) -> dict[int | str, float]:
        strategy = str(strategy)
        if strategy in {"manual_pair", "pair_grid"}:
            pair_weights = relation_pair_weights or {"AP_PA": 0.50, "PT_TP": 0.30, "PV_VP": 0.20}
            out: dict[int | str, float] = {}
            pair_capacity: dict[str, float] = {}
            for stat in stats:
                pair = str(stat.relation_pair_name or stat.relation_name)
                pair_capacity[pair] = pair_capacity.get(pair, 0.0) + max(float(stat.candidate_edge_count), 0.0)
            for stat in stats:
                pair = str(stat.relation_pair_name or stat.relation_name)
                pair_weight = float(pair_weights.get(pair, 0.0))
                within_pair = float(stat.candidate_edge_count) / max(pair_capacity.get(pair, 0.0), 1.0)
                out[stat.relation_id] = max(0.0, pair_weight * within_pair)
            return out
        if strategy == "random_relationwise":
            rng = np.random.default_rng(int(seed))
            return {stat.relation_id: float(rng.random() + 1e-6) for stat in stats}
        if strategy == "validation_greedy_chunks":
            return {
                stat.relation_id: max(0.0, float(validation_feedback.get(str(stat.relation_pair_name or stat.relation_name), 0.0)))
                for stat in stats
            }
        if strategy in {"degree_topk_relationwise", "current_heuristic", "path_aware", "proportional"}:
            return {stat.relation_id: max(0.0, float(stat.candidate_edge_count)) for stat in stats}
        raise ValueError(f"unsupported relation budget strategy: {strategy}")

    @staticmethod
    def _distribute_remaining(
        stats: list[RelationStats],
        budgets: dict[int | str, int],
        capacities: dict[int | str, int],
        weights: dict[int | str, float],
        remaining: int,
    ) -> None:
        if remaining <= 0:
            return
        active = [stat for stat in stats if capacities[stat.relation_id] > budgets.get(stat.relation_id, 0)]
        if not active:
            return
        total_weight = sum(max(0.0, float(weights.get(stat.relation_id, 0.0))) for stat in active)
        if total_weight <= 0.0:
            weights = {stat.relation_id: float(stat.candidate_edge_count) for stat in active}
            total_weight = sum(float(weights[stat.relation_id]) for stat in active) or 1.0
        fractional: list[tuple[float, str, int | str]] = []
        for stat in active:
            capacity = capacities[stat.relation_id] - budgets.get(stat.relation_id, 0)
            raw = float(remaining) * max(0.0, float(weights.get(stat.relation_id, 0.0))) / total_weight
            add = min(capacity, int(np.floor(raw)))
            budgets[stat.relation_id] = budgets.get(stat.relation_id, 0) + max(0, add)
            fractional.append((raw - np.floor(raw), str(stat.relation_id), stat.relation_id))
        leftover = max(0, int(remaining) - sum(max(0, budgets[stat.relation_id] - max(0, int(stat.min_edges))) for stat in active))
        while leftover > 0:
            progressed = False
            for _frac, _name, relation_id in sorted(fractional, reverse=True):
                if leftover <= 0:
                    break
                if budgets.get(relation_id, 0) >= capacities.get(relation_id, 0):
                    continue
                budgets[relation_id] = budgets.get(relation_id, 0) + 1
                leftover -= 1
                progressed = True
            if not progressed:
                break
