from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS, supported_sehgnn_hgb_dataset
from hesf_coarsen.io.schema import HeteroGraph


@dataclass(frozen=True, order=True)
class RelationKey:
    dataset: str
    source_relation_id: int | None
    source_relation_name: str | None
    source_src_type: str | None
    source_dst_type: str | None
    official_relation_id: int
    official_relation_name: str
    official_src_type: str
    official_dst_type: str
    relation_pair_name: str
    reciprocal_official_relation_id: int | None
    reciprocal_official_relation_name: str | None


DBLP_RELATION_PAIRS: dict[str, tuple[str, str]] = {
    "AP_PA": ("AP", "PA"),
    "PT_TP": ("PT", "TP"),
    "PV_VP": ("PV", "VP"),
}

DBLP_RECIPROCAL: dict[str, str] = {
    "AP": "PA",
    "PA": "AP",
    "PT": "TP",
    "TP": "PT",
    "PV": "VP",
    "VP": "PV",
}


def relation_pair_name(relation_name: str) -> str:
    name = str(relation_name)
    for pair, members in DBLP_RELATION_PAIRS.items():
        if name in members:
            return pair
    return name


def official_endpoints(dataset: str, relation_name: str) -> tuple[str, str]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    token = str(relation_name).replace("_r", "")
    if len(token) < 2:
        raise ValueError(f"cannot infer official endpoints for relation {relation_name!r}")
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    for node_name in (token[0], token[1]):
        if node_name not in schema["node_type_order"]:
            raise ValueError(f"relation {relation_name!r} endpoint {node_name!r} is not in {dataset_name} schema")
    return token[0], token[1]


def official_relation_name_for_source(dataset: str, source_src_type: int | str, source_dst_type: int | str) -> str:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    type_name_by_id = {int(type_id): str(name) for name, type_id in schema["node_type_order"].items()}
    if str(source_src_type).isdigit():
        src_name = type_name_by_id[int(source_src_type)]
    else:
        src_name = str(source_src_type)
    if str(source_dst_type).isdigit():
        dst_name = type_name_by_id[int(source_dst_type)]
    else:
        dst_name = str(source_dst_type)
    relation_name = f"{src_name}{dst_name}"
    if relation_name not in schema["relation_id_order"]:
        raise ValueError(f"no official {dataset_name} relation for {src_name}->{dst_name}")
    return relation_name


def _canonical_relation_key(dataset: str, relation_name: str, relation_id: int) -> RelationKey:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    src, dst = official_endpoints(dataset_name, relation_name)
    reciprocal_name = DBLP_RECIPROCAL.get(str(relation_name)) if dataset_name == "DBLP" else None
    reciprocal_id = None
    if reciprocal_name is not None:
        reciprocal_id = int(SEHGNN_HGB_SCHEMAS[dataset_name]["relation_id_order"][reciprocal_name])
    return RelationKey(
        dataset=dataset_name,
        source_relation_id=-1,
        source_relation_name=str(relation_name),
        source_src_type=src,
        source_dst_type=dst,
        official_relation_id=int(relation_id),
        official_relation_name=str(relation_name),
        official_src_type=src,
        official_dst_type=dst,
        relation_pair_name=relation_pair_name(str(relation_name)),
        reciprocal_official_relation_id=reciprocal_id,
        reciprocal_official_relation_name=reciprocal_name,
    )


DBLP_RELATION_KEYS: tuple[RelationKey, ...] = tuple(
    _canonical_relation_key("DBLP", name, relation_id)
    for name, relation_id in sorted(SEHGNN_HGB_SCHEMAS["DBLP"]["relation_id_order"].items(), key=lambda item: int(item[1]))
)


def build_relation_keys(graph: HeteroGraph, *, dataset: str) -> list[RelationKey]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    source_by_official_name: dict[str, tuple[int, Any]] = {}
    for relation_id, spec in graph.relation_specs.items():
        try:
            official_name = official_relation_name_for_source(dataset_name, spec.src_type, spec.dst_type)
        except ValueError:
            official_name = str(spec.name)
        source_by_official_name[official_name] = (int(relation_id), spec)
    official_by_name = {str(name): int(rid) for name, rid in schema["relation_id_order"].items()}
    rows: list[RelationKey] = []
    type_name_by_id = {int(type_id): str(name) for name, type_id in schema["node_type_order"].items()}
    for official_name, official_id in sorted(official_by_name.items(), key=lambda item: int(item[1])):
        official_src, official_dst = official_endpoints(dataset_name, official_name)
        source_id, source_spec = source_by_official_name.get(str(official_name), (-1, None))
        if source_spec is None:
            source_name = official_name
            source_src = official_src
            source_dst = official_dst
        else:
            source_name = str(source_spec.name)
            source_src = type_name_by_id.get(int(source_spec.src_type), str(source_spec.src_type))
            source_dst = type_name_by_id.get(int(source_spec.dst_type), str(source_spec.dst_type))
        reciprocal_name = DBLP_RECIPROCAL.get(official_name) if dataset_name == "DBLP" else None
        reciprocal_id = official_by_name.get(reciprocal_name) if reciprocal_name is not None else None
        rows.append(
            RelationKey(
                dataset=dataset_name,
                source_relation_id=int(source_id),
                source_relation_name=source_name,
                source_src_type=source_src,
                source_dst_type=source_dst,
                official_relation_id=int(official_id),
                official_relation_name=official_name,
                official_src_type=official_src,
                official_dst_type=official_dst,
                relation_pair_name=relation_pair_name(official_name),
                reciprocal_official_relation_id=None if reciprocal_id is None else int(reciprocal_id),
                reciprocal_official_relation_name=reciprocal_name,
            )
        )
    return rows


def assert_relation_key_reciprocals(graph: HeteroGraph, *, dataset: str) -> None:
    keys = build_relation_keys(graph, dataset=dataset)
    by_name = {key.official_relation_name: key for key in keys}
    for key in keys:
        if key.reciprocal_official_relation_name is None:
            continue
        reciprocal = by_name[key.reciprocal_official_relation_name]
        if reciprocal.reciprocal_official_relation_name != key.official_relation_name:
            raise AssertionError(f"reciprocal mismatch for {key.official_relation_name}")
        if reciprocal.official_relation_id != key.reciprocal_official_relation_id:
            raise AssertionError(f"reciprocal id mismatch for {key.official_relation_name}")


def parse_link_relation_counts(dataset_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    link_path = Path(dataset_dir) / "link.dat"
    if not link_path.exists():
        return {}
    with link_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2] != "":
                counts[str(int(parts[2]))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def parse_node_type_counts(dataset_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    node_path = Path(dataset_dir) / "node.dat"
    if not node_path.exists():
        return {}
    with node_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2] != "":
                counts[str(int(parts[2]))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _normalise_counts(counts: Mapping[str | int, Any] | str | None) -> dict[str, int]:
    if counts is None:
        return {}
    if isinstance(counts, str):
        if not counts.strip():
            return {}
        counts = json.loads(counts)
    return {str(int(key)): int(value) for key, value in dict(counts).items()}


def validate_hgb_relation_order(
    *,
    dataset: str,
    dataset_dir: Path,
    hgb_export_edge_counts: Mapping[str | int, Any] | str | None,
) -> dict[str, Any]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    relation_ids = {str(int(value)) for value in schema["relation_id_order"].values()}
    node_type_ids = {str(int(value)) for value in schema["node_type_order"].values()}
    link_counts = parse_link_relation_counts(Path(dataset_dir))
    node_counts = parse_node_type_counts(Path(dataset_dir))
    export_counts = _normalise_counts(hgb_export_edge_counts)
    return {
        "dataset": dataset_name,
        "dataset_dir": str(Path(dataset_dir)),
        "relation_order_matches_official": set(link_counts).issubset(relation_ids) and relation_ids.issubset(set(export_counts) | set(link_counts)),
        "node_type_order_matches_official": set(node_counts).issubset(node_type_ids),
        "link_dat_relation_counts_match_export_audit": link_counts == export_counts if export_counts else False,
        "edge_count_by_relation": link_counts,
        "hgb_export_edge_count_by_relation": export_counts,
    }
