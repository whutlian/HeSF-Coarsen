from __future__ import annotations

from typing import Any, Sequence

from hesf_coarsen.eval.official.relation_budget_allocator import (
    RelationBudgetAllocation,
    RelationStats,
    allocate_relation_channel_spec,
    parse_relation_channel_spec,
)
from hesf_coarsen.eval.official.directed_relation_skeleton import canonicalize_directed_method, is_directed_skeleton_method


APV_SKELETON_SPEC = "APPA100-PVVP100-PTTP00"
APV_SKELETON_METHOD = "H6-APV-skeleton"
APV_SKELETON_CANONICAL_METHOD = f"H6-relgrid-{APV_SKELETON_SPEC}"
APV_SKELETON_EDGE_SCORE_STRATEGY = "keep_full_APPA_PVVP_min_schema_PTTP"

DIRECTIONALITY_SPECS = [
    "AP100-PA00-PTTP30-PVVP100",
    "AP00-PA100-PTTP30-PVVP100",
    "AP100-PA50-PTTP30-PVVP100",
    "AP50-PA100-PTTP30-PVVP100",
    "APPA100-PT100-TP00-PVVP100",
    "APPA100-PT00-TP100-PVVP100",
    "APPA100-PT50-TP20-PVVP100",
    "APPA100-PT20-TP50-PVVP100",
    "APPA100-PTTP30-PV100-VP00",
    "APPA100-PTTP30-PV00-VP100",
    "APPA100-PTTP30-PV100-VP50",
    "APPA100-PTTP30-PV50-VP100",
]


def canonicalize_gate21_4_method(method: str) -> dict[str, Any]:
    raw = str(method).strip()
    token = raw
    if token in {"APV-skeleton", "H6-APV-skeleton", APV_SKELETON_CANONICAL_METHOD, APV_SKELETON_SPEC}:
        return {
            "method": APV_SKELETON_METHOD,
            "canonical_method": APV_SKELETON_CANONICAL_METHOD,
            "relation_channel_spec": APV_SKELETON_SPEC,
            "method_family": "schema_compatible_subgraph",
            "budget_strategy": "relation_channel_skeleton",
            "edge_score_strategy": APV_SKELETON_EDGE_SCORE_STRATEGY,
            "is_directionality_ablation": False,
            "official_sehgnn_unmodified": True,
            "eligible_for_main_decision": True,
        }
    if token.upper().startswith("PTTP"):
        suffix = token.upper().replace("PTTP", "", 1)
        if not suffix.isdigit():
            raise ValueError(f"unsupported Gate21.4 PTTP method: {method!r}")
        spec = f"APPA100-PVVP100-PTTP{suffix.zfill(2)}"
        return {
            "method": f"H6-relgrid-{spec}",
            "canonical_method": f"H6-relgrid-{spec}",
            "relation_channel_spec": spec,
            "method_family": "schema_compatible_subgraph",
            "budget_strategy": "relation_channel_grid",
            "edge_score_strategy": "random_edge_within_relation",
            "is_directionality_ablation": False,
            "official_sehgnn_unmodified": True,
            "eligible_for_main_decision": True,
        }
    if token == "directionality":
        return {
            "method": "directionality",
            "canonical_method": "directionality",
            "relation_channel_spec": "",
            "method_family": "schema_compatible_subgraph",
            "budget_strategy": "relation_channel_directionality",
            "edge_score_strategy": "random_edge_within_relation",
            "is_directionality_ablation": True,
            "official_sehgnn_unmodified": True,
            "eligible_for_main_decision": True,
        }
    if token.startswith("H6-dirskel-") or token.startswith("dirskel-"):
        return {**canonicalize_directed_method(token), "is_directionality_ablation": False}
    if token in set(DIRECTIONALITY_SPECS) or _looks_like_directionality_spec(token):
        return {
            "method": f"H6-dir-{token}",
            "canonical_method": f"H6-dir-{token}",
            "relation_channel_spec": token,
            "method_family": "schema_compatible_subgraph",
            "budget_strategy": "relation_channel_directionality",
            "edge_score_strategy": "random_edge_within_relation",
            "is_directionality_ablation": True,
            "official_sehgnn_unmodified": True,
            "eligible_for_main_decision": True,
        }
    if token in {"full-native", "full-native-SeHGNN"}:
        return _reference_method("full-native-SeHGNN")
    if token in {"export-full", "export-full-SeHGNN"}:
        return _reference_method("export-full-SeHGNN")
    if token == "H6-node30":
        return {
            "method": "H6-node30",
            "canonical_method": "H6-node30",
            "relation_channel_spec": "",
            "method_family": "schema_compatible_subgraph",
            "budget_strategy": "node30_support",
            "edge_score_strategy": "none",
            "is_directionality_ablation": False,
            "official_sehgnn_unmodified": True,
            "eligible_for_main_decision": True,
        }
    if token == "H6-struct40-random-relwise":
        return _structural_method(token, "random_edge_within_relation", 0.40)
    if token == "H6-struct30-proportional-current":
        return _structural_method(token, "current_heuristic", 0.30)
    raise ValueError(f"unsupported Gate21.4 method: {method!r}")


def expand_gate21_4_methods(methods: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        if str(method).strip() == "directionality":
            rows.extend(canonicalize_gate21_4_method(spec) for spec in DIRECTIONALITY_SPECS)
        else:
            rows.append(canonicalize_gate21_4_method(str(method)))
    return rows


def allocate_skeleton_budget(
    relation_stats: list[RelationStats],
    relation_channel_spec: str = APV_SKELETON_SPEC,
    *,
    min_edges_per_relation: int = 1,
) -> list[RelationBudgetAllocation]:
    parsed = parse_relation_channel_spec(relation_channel_spec, sampling_strategy="random_edge_within_relation")
    return allocate_relation_channel_spec(relation_stats, parsed, min_edges_per_relation=min_edges_per_relation)


def _reference_method(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "canonical_method": method,
        "relation_channel_spec": "",
        "method_family": "reference_full",
        "budget_strategy": "reference",
        "edge_score_strategy": "reference",
        "is_directionality_ablation": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": False,
    }


def _structural_method(method: str, edge_score_strategy: str, budget: float) -> dict[str, Any]:
    return {
        "method": method,
        "canonical_method": method,
        "relation_channel_spec": "",
        "method_family": "schema_compatible_subgraph",
        "budget_strategy": "proportional",
        "edge_score_strategy": edge_score_strategy,
        "storage_budget": float(budget),
        "is_directionality_ablation": False,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": True,
    }


def _looks_like_directionality_spec(token: str) -> bool:
    try:
        parsed = parse_relation_channel_spec(token)
    except ValueError:
        return False
    values = parsed.retention_by_relation
    return any(values[forward] != values[reverse] for forward, reverse in [("AP", "PA"), ("PT", "TP"), ("PV", "VP")])
