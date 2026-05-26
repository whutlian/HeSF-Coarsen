from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS, supported_sehgnn_hgb_dataset
from hesf_coarsen.io.schema import HeteroGraph


DBLP_RECIPROCAL = {
    "AP": "PA",
    "PA": "AP",
    "PT": "TP",
    "TP": "PT",
    "PV": "VP",
    "VP": "PV",
}

RELATION_MAPPING_FIELDS = [
    "dataset",
    "method",
    "seed",
    "source_relation_id",
    "source_relation_name",
    "official_relation_id",
    "official_relation_name",
    "source_src_type",
    "source_dst_type",
    "official_src_type",
    "official_dst_type",
    "source_edge_count",
    "candidate_edge_count_after_node_pruning",
    "retained_edge_count",
    "original_full_edge_count",
    "retention_vs_candidate",
    "retention_vs_full",
    "reciprocal_relation_id",
    "reciprocal_relation_name",
    "reciprocal_count_consistent",
    "min_edges_constraint_active",
    "relation_dropped_flag",
]

RELATION_RETENTION_FIELDS = [
    "dataset",
    "seed",
    "method",
    "budget_strategy",
    "edge_score_strategy",
    "source_relation_id",
    "source_relation_name",
    "official_relation_id",
    "official_relation_name",
    "relation_pair_name",
    "original_full_edge_count",
    "candidate_edge_count_after_node_pruning",
    "retained_edge_count",
    "retention_vs_candidate",
    "retention_vs_full",
    "requested_relation_budget",
    "actual_relation_budget",
    "min_edges_constraint_active",
    "relation_dropped_flag",
]


@dataclass(frozen=True)
class RelationMappingRow:
    dataset: str
    method: str
    seed: int | None
    source_relation_id: int | str | None
    source_relation_name: str | None
    official_relation_id: int | str | None
    official_relation_name: str | None
    source_src_type: str | None
    source_dst_type: str | None
    official_src_type: str | None
    official_dst_type: str | None
    source_edge_count: int | None
    candidate_edge_count_after_node_pruning: int | None
    retained_edge_count: int | None
    original_full_edge_count: int | None
    retention_vs_candidate: float | None
    retention_vs_full: float | None
    reciprocal_relation_id: int | str | None
    reciprocal_relation_name: str | None
    reciprocal_count_consistent: bool | None
    min_edges_constraint_active: bool
    relation_dropped_flag: bool

    def to_row(self) -> dict[str, Any]:
        data = asdict(self)
        return {field: data.get(field, "") for field in RELATION_MAPPING_FIELDS}

    def to_retention_row(
        self,
        *,
        budget_strategy: str,
        edge_score_strategy: str,
        requested_relation_budget: int | None = None,
        actual_relation_budget: int | None = None,
    ) -> dict[str, Any]:
        pair = relation_pair_name(str(self.official_relation_name or self.source_relation_name or ""))
        data = {
            "dataset": self.dataset,
            "seed": self.seed,
            "method": self.method,
            "budget_strategy": budget_strategy,
            "edge_score_strategy": edge_score_strategy,
            "source_relation_id": self.source_relation_id,
            "source_relation_name": self.source_relation_name,
            "official_relation_id": self.official_relation_id,
            "official_relation_name": self.official_relation_name,
            "relation_pair_name": pair,
            "original_full_edge_count": self.original_full_edge_count,
            "candidate_edge_count_after_node_pruning": self.candidate_edge_count_after_node_pruning,
            "retained_edge_count": self.retained_edge_count,
            "retention_vs_candidate": self.retention_vs_candidate,
            "retention_vs_full": self.retention_vs_full,
            "requested_relation_budget": requested_relation_budget,
            "actual_relation_budget": actual_relation_budget if actual_relation_budget is not None else self.retained_edge_count,
            "min_edges_constraint_active": self.min_edges_constraint_active,
            "relation_dropped_flag": self.relation_dropped_flag,
        }
        return {field: data.get(field, "") for field in RELATION_RETENTION_FIELDS}


def relation_pair_name(relation_name: str) -> str:
    name = str(relation_name)
    if name in {"AP", "PA"}:
        return "AP_PA"
    if name in {"PT", "TP"}:
        return "PT_TP"
    if name in {"PV", "VP"}:
        return "PV_VP"
    return name


def _official_endpoints(dataset: str, relation_name: str) -> tuple[str | None, str | None]:
    schema = SEHGNN_HGB_SCHEMAS[supported_sehgnn_hgb_dataset(dataset)]
    token = str(relation_name).replace("_r", "")
    if len(token) < 2:
        return None, None
    return token[0], token[1]


def _retention(retained: int | None, denom: int | None) -> float | None:
    if retained is None or denom in {None, 0}:
        return None
    return float(int(retained) / max(int(denom), 1))


def audit_relation_mapping(
    *,
    graph: HeteroGraph,
    dataset: str,
    method: str,
    seed: int | None,
    candidate_edge_counts: Mapping[int | str, int] | None = None,
    retained_edge_counts: Mapping[int | str, int] | None = None,
    original_full_edge_counts: Mapping[int | str, int] | None = None,
    min_edges_constraint_active: Mapping[int | str, bool] | None = None,
) -> list[RelationMappingRow]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    official_by_name = {str(name): int(rid) for name, rid in schema["relation_id_order"].items()}
    name_by_official = {int(rid): str(name) for name, rid in schema["relation_id_order"].items()}
    source_by_name = {str(spec.name): int(rel_id) for rel_id, spec in graph.relation_specs.items()}
    rows: list[RelationMappingRow] = []
    candidate_edge_counts = candidate_edge_counts or {}
    retained_edge_counts = retained_edge_counts or {}
    original_full_edge_counts = original_full_edge_counts or {}
    min_edges_constraint_active = min_edges_constraint_active or {}
    for official_id in sorted(name_by_official):
        official_name = name_by_official[int(official_id)]
        source_id = source_by_name.get(official_name)
        rel = None if source_id is None else graph.relations.get(int(source_id))
        spec = None if source_id is None else graph.relation_specs.get(int(source_id))
        source_edges = None if rel is None else int(rel.num_edges)
        candidate = candidate_edge_counts.get(source_id, candidate_edge_counts.get(official_name, source_edges))
        retained = retained_edge_counts.get(source_id, retained_edge_counts.get(official_name, source_edges))
        full = original_full_edge_counts.get(source_id, original_full_edge_counts.get(official_name, source_edges))
        reciprocal_name = DBLP_RECIPROCAL.get(official_name) if dataset_name == "DBLP" else None
        reciprocal_id = official_by_name.get(reciprocal_name) if reciprocal_name is not None else None
        reciprocal_retained = None
        if reciprocal_name is not None:
            reciprocal_source = source_by_name.get(reciprocal_name)
            reciprocal_retained = retained_edge_counts.get(reciprocal_source, retained_edge_counts.get(reciprocal_name))
        reciprocal_consistent = None
        if reciprocal_name is not None and retained is not None and reciprocal_retained is not None:
            reciprocal_consistent = int(retained) == int(reciprocal_retained)
        official_src, official_dst = _official_endpoints(dataset_name, official_name)
        rows.append(
            RelationMappingRow(
                dataset=dataset_name,
                method=str(method),
                seed=None if seed is None else int(seed),
                source_relation_id=source_id,
                source_relation_name=None if spec is None else str(spec.name),
                official_relation_id=int(official_id),
                official_relation_name=official_name,
                source_src_type=None if spec is None else str(spec.src_type),
                source_dst_type=None if spec is None else str(spec.dst_type),
                official_src_type=official_src,
                official_dst_type=official_dst,
                source_edge_count=source_edges,
                candidate_edge_count_after_node_pruning=None if candidate is None else int(candidate),
                retained_edge_count=None if retained is None else int(retained),
                original_full_edge_count=None if full is None else int(full),
                retention_vs_candidate=_retention(None if retained is None else int(retained), None if candidate is None else int(candidate)),
                retention_vs_full=_retention(None if retained is None else int(retained), None if full is None else int(full)),
                reciprocal_relation_id=reciprocal_id,
                reciprocal_relation_name=reciprocal_name,
                reciprocal_count_consistent=reciprocal_consistent,
                min_edges_constraint_active=bool(min_edges_constraint_active.get(source_id, min_edges_constraint_active.get(official_name, False))),
                relation_dropped_flag=bool(retained is not None and int(retained) == 0),
            )
        )
    return rows
