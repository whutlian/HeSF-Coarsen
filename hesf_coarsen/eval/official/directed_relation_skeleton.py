from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from hesf_coarsen.eval.official.relation_budget_allocator import RelationStats, parse_relation_channel_spec


DIRECTED_PRIMARY_SPECS = [
    "AP100-PA00-PV100-VP00-PTTP00",
    "AP100-PA00-PV100-VP00-PTTP05",
    "AP100-PA00-PV100-VP00-PTTP10",
    "AP100-PA00-PV100-VP00-PTTP30",
]

DIRECTED_FULL_SPECS = [
    *DIRECTED_PRIMARY_SPECS,
    "AP100-PA50-PV100-VP00-PTTP00",
    "AP100-PA00-PV100-VP50-PTTP00",
    "AP100-PA50-PV100-VP50-PTTP00",
    "AP100-PA00-PV50-VP00-PTTP00",
    "AP50-PA00-PV100-VP00-PTTP00",
    "AP75-PA00-PV100-VP00-PTTP00",
    "AP100-PA00-PV75-VP00-PTTP00",
]

DIRECTED_CONTROLS = [
    "H6-APV-skeleton",
    "H6-node30",
    "export-full-SeHGNN",
    "full-native-SeHGNN",
]


@dataclass(frozen=True)
class DirectedRelationSpec:
    ap: float
    pa: float
    pt: float
    tp: float
    pv: float
    vp: float
    schema_min_edges: bool = True
    min_edges_per_relation: int = 1
    sampling_strategy: str = "random_edge_within_relation"
    deterministic: bool = True

    @property
    def retention_by_relation(self) -> dict[str, float]:
        return {
            "AP": float(self.ap),
            "PA": float(self.pa),
            "PT": float(self.pt),
            "TP": float(self.tp),
            "PV": float(self.pv),
            "VP": float(self.vp),
        }

    @property
    def canonical_spec(self) -> str:
        parts = [_token("AP", self.ap), _token("PA", self.pa), _token("PV", self.pv), _token("VP", self.vp)]
        if abs(float(self.pt) - float(self.tp)) <= 1e-12:
            parts.append(_token("PTTP", self.pt))
        else:
            parts.extend([_token("PT", self.pt), _token("TP", self.tp)])
        return "-".join(parts)

    @property
    def canonical_method(self) -> str:
        return f"H6-dirskel-{self.canonical_spec}"


def parse_directed_relation_spec(spec: str, *, schema_min_edges: bool = True, min_edges_per_relation: int = 1, sampling_strategy: str = "random_edge_within_relation") -> DirectedRelationSpec:
    raw = str(spec).strip()
    for prefix in ("H6-dirskel-", "dirskel-"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    parsed = parse_relation_channel_spec(raw, sampling_strategy=sampling_strategy)
    values = parsed.retention_by_relation
    return DirectedRelationSpec(
        ap=float(values["AP"]),
        pa=float(values["PA"]),
        pt=float(values["PT"]),
        tp=float(values["TP"]),
        pv=float(values["PV"]),
        vp=float(values["VP"]),
        schema_min_edges=bool(schema_min_edges),
        min_edges_per_relation=int(min_edges_per_relation),
        sampling_strategy=str(sampling_strategy),
        deterministic=True,
    )


def is_directed_skeleton_method(method: str) -> bool:
    token = str(method).strip()
    if token.startswith("H6-dirskel-") or token.startswith("dirskel-"):
        return True
    try:
        parse_directed_relation_spec(token)
    except ValueError:
        return False
    return bool(re.search(r"(AP|PA|PV|VP)\d", token))


def canonicalize_directed_method(method: str) -> dict[str, Any]:
    spec = parse_directed_relation_spec(method)
    return {
        "method": spec.canonical_method,
        "canonical_method": spec.canonical_method,
        "relation_channel_spec": spec.canonical_spec,
        "method_family": "schema_compatible_directed_skeleton",
        "budget_strategy": "directed_relation_channel_skeleton",
        "edge_score_strategy": str(spec.sampling_strategy),
        "deterministic_graph_method": bool(spec.deterministic),
        "graph_seed_independence_required": False,
        "graph_seed_independence_status": "not_applicable_deterministic",
        "num_effective_graph_variants": 1,
        "official_sehgnn_unmodified": True,
        "eligible_for_main_decision": True,
    }


def expand_directed_methods(mode: str, custom_methods: str | Sequence[str] | None = None) -> list[str]:
    if str(mode) == "core":
        return [f"H6-dirskel-{spec}" for spec in DIRECTED_PRIMARY_SPECS] + list(DIRECTED_CONTROLS)
    if str(mode) == "full":
        return [f"H6-dirskel-{spec}" for spec in DIRECTED_FULL_SPECS] + list(DIRECTED_CONTROLS)
    if str(mode) == "custom":
        if custom_methods is None:
            raise ValueError("--custom-methods is required for --methods custom")
        if isinstance(custom_methods, str):
            items = [item.strip() for item in custom_methods.split(",") if item.strip()]
        else:
            items = [str(item).strip() for item in custom_methods if str(item).strip()]
        return [canonicalize_directed_method(item)["method"] if is_directed_skeleton_method(item) else item for item in items]
    raise ValueError(f"unsupported Gate21.5 method mode: {mode!r}")


def allocate_directed_relation_budget(
    relation_stats: list[RelationStats],
    relation_channel_spec: str,
    *,
    schema_min_edges: bool = True,
    min_edges_per_relation: int = 1,
) -> list[dict[str, Any]]:
    spec = parse_directed_relation_spec(
        relation_channel_spec,
        schema_min_edges=schema_min_edges,
        min_edges_per_relation=min_edges_per_relation,
    )
    rows: list[dict[str, Any]] = []
    for stat in relation_stats:
        candidate = max(0, int(stat.candidate_edge_count))
        requested = int(round(candidate * spec.retention_by_relation.get(str(stat.relation_name), 0.0)))
        if schema_min_edges:
            minimum = min(candidate, max(int(min_edges_per_relation), int(stat.min_edges)))
            actual = min(candidate, max(minimum, requested))
        else:
            actual = min(candidate, requested)
        rows.append(
            {
                "relation_id": stat.relation_id,
                "relation_name": str(stat.relation_name),
                "relation_pair_name": stat.relation_pair_name,
                "candidate_edge_count": int(candidate),
                "requested_edge_count": int(requested),
                "retained_edge_count": int(actual),
                "requested_ratio_vs_candidate": float(requested / max(candidate, 1)),
                "min_edges_constraint_active": bool(schema_min_edges and actual != requested and candidate > 0),
                "relation_dropped_flag": bool(actual == 0),
                "eligible_for_main_decision": bool(schema_min_edges and actual > 0),
            }
        )
    return rows


def _token(name: str, ratio: float) -> str:
    pct = int(round(float(ratio) * 100.0))
    return f"{name}{pct:02d}"
