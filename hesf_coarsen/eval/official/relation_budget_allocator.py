from __future__ import annotations

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
