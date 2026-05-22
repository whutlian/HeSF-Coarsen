from __future__ import annotations

from typing import Any
from collections import Counter, defaultdict

import numpy as np

from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.task_first.selection.config import SupportSelectorConfig


def _argmax_or_unknown(matrix: np.ndarray, node: int) -> int:
    if matrix.size == 0 or int(node) >= matrix.shape[0] or matrix.shape[1] == 0:
        return -1
    row = np.asarray(matrix[int(node)], dtype=np.float32)
    if float(np.sum(row)) <= 1.0e-12:
        return -1
    return int(np.argmax(row))


def _prototype_key(
    node: int,
    type_id: int,
    support_features: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    if not support_features:
        return (int(type_id), -1, -1, -1)
    all_components = support_features.get("all_node_component_matrices", {})
    class_id = _argmax_or_unknown(np.asarray(all_components.get("class_footprint", np.empty((0, 0)))), int(node))
    anchor_id = _argmax_or_unknown(np.asarray(all_components.get("anchor_distribution", np.empty((0, 0)))), int(node))
    relation_bucket = _argmax_or_unknown(np.asarray(all_components.get("relation_profile", np.empty((0, 0)))), int(node))
    return (int(type_id), int(class_id), int(anchor_id), int(relation_bucket))


def _node_component_row(
    support_features: dict[str, Any] | None,
    name: str,
    node: int,
) -> np.ndarray:
    if not support_features:
        return np.empty(0, dtype=np.float32)
    all_components = support_features.get("all_node_component_matrices", {})
    matrix = np.asarray(all_components.get(name, np.empty((0, 0))), dtype=np.float32)
    if matrix.size == 0 or int(node) >= matrix.shape[0]:
        return np.empty(0, dtype=np.float32)
    return np.asarray(matrix[int(node)], dtype=np.float32).reshape(-1)


def _degree_bucket(
    node: int,
    graph: HeteroGraph,
    support_features: dict[str, Any] | None,
) -> int:
    degree_row = _node_component_row(support_features, "degree_profile", int(node))
    if degree_row.size:
        degree = float(np.sum(degree_row))
    else:
        degree = 0.0
        for rel in graph.relations.values():
            degree += float(np.count_nonzero(rel.src == int(node)))
            degree += float(np.count_nonzero(rel.dst == int(node)))
    if degree <= 0.0:
        return 0
    if degree <= 1.0:
        return 1
    if degree <= 3.0:
        return 2
    if degree <= 10.0:
        return 3
    return 4


def _bridge_flag(
    node: int,
    degree_bucket: int,
    support_features: dict[str, Any] | None,
) -> int:
    relation = _node_component_row(support_features, "relation_profile", int(node))
    anchor = _node_component_row(support_features, "anchor_distribution", int(node))
    class_fp = _node_component_row(support_features, "class_footprint", int(node))
    return int(
        int(degree_bucket) >= 4
        or int(np.count_nonzero(relation > 1.0e-12)) > 1
        or int(np.count_nonzero(anchor > 1.0e-12)) > 1
        or int(np.count_nonzero(class_fp > 1.0e-12)) > 1
    )


def _dblp_aware_prototype_key(
    node: int,
    type_id: int,
    graph: HeteroGraph,
    support_features: dict[str, Any] | None,
) -> tuple[int, int, int, int, int, int]:
    base = _prototype_key(int(node), int(type_id), support_features)
    degree_bucket = _degree_bucket(int(node), graph, support_features)
    bridge = _bridge_flag(int(node), degree_bucket, support_features)
    # DBLP-aware order: support type, relation channel, anchor group, class bucket, degree bucket, bridge flag.
    return (int(type_id), int(base[3]), int(base[2]), int(base[1]), int(degree_bucket), int(bridge))


def _relation_channel(node: int, support_features: dict[str, Any] | None) -> int:
    relation = _node_component_row(support_features, "relation_profile", int(node))
    if relation.size == 0 or float(np.sum(relation)) <= 1.0e-12:
        return -1
    return int(np.argmax(relation))


def _prototype_count_fields(key: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(key) >= 6:
        return int(key[0]), int(key[3]), int(key[2]), int(key[1])
    return int(key[0]), int(key[1]), int(key[2]), int(key[3])


def build_selected_support_graph(
    original_graph: HeteroGraph,
    selected_support_nodes: np.ndarray,
    cfg: SupportSelectorConfig,
    *,
    target_node_type: int,
    support_features: dict[str, Any] | None = None,
) -> tuple[HeteroGraph, Assignment, dict[str, Any]]:
    target_type = int(target_node_type)
    selected = {int(node) for node in np.asarray(selected_support_nodes, dtype=np.int64).reshape(-1)}
    target_nodes = np.flatnonzero(original_graph.node_type == target_type).astype(np.int64)
    support_nodes = np.flatnonzero(original_graph.node_type != target_type).astype(np.int64)
    strategy = str(cfg.background_strategy)
    base_prototype_key_by_node: dict[int, tuple[int, ...]] = {}
    prototype_key_by_node: dict[int, tuple[int, ...]] = {}
    large_prototype_count = 0
    large_prototype_split_count = 0
    forced_raw_nodes: set[int] = set()
    raw_bridge_by_type: Counter[str] = Counter()
    raw_bridge_by_relation_channel: Counter[str] = Counter()
    force_raw_bridges = bool(
        getattr(cfg, "force_raw_bridge_nodes", False)
        or getattr(cfg, "force_raw_keep_high_degree_bridges", False)
    )
    if strategy in {"class_anchor_relation_prototype", "dblp_aware_prototype"} and force_raw_bridges:
        for node in support_nodes:
            node = int(node)
            if node in selected:
                continue
            type_id = int(original_graph.node_type[node])
            degree_bucket = _degree_bucket(node, original_graph, support_features)
            if _bridge_flag(node, degree_bucket, support_features):
                forced_raw_nodes.add(node)
                raw_bridge_by_type[str(type_id)] += 1
                raw_bridge_by_relation_channel[str(_relation_channel(node, support_features))] += 1
    if strategy in {"class_anchor_relation_prototype", "dblp_aware_prototype"}:
        grouped_nodes: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for node in support_nodes:
            node = int(node)
            if node in selected or node in forced_raw_nodes:
                continue
            type_id = int(original_graph.node_type[node])
            if strategy == "dblp_aware_prototype":
                key = _dblp_aware_prototype_key(node, type_id, original_graph, support_features)
            else:
                key = _prototype_key(node, type_id, support_features)
            base_prototype_key_by_node[node] = tuple(int(value) for value in key)
            grouped_nodes[base_prototype_key_by_node[node]].append(node)
        cap = max(1, int(cfg.max_members_per_prototype))
        for key, members in grouped_nodes.items():
            ordered = sorted(
                members,
                key=lambda item: (
                    -_degree_bucket(int(item), original_graph, support_features),
                    int(item),
                ),
            )
            if len(ordered) > cap:
                large_prototype_count += 1
            for chunk_id, start in enumerate(range(0, len(ordered), cap)):
                chunk = ordered[start : start + cap]
                split_key = tuple(key) + ((int(chunk_id) if len(ordered) > cap else 0),)
                if len(ordered) > cap:
                    large_prototype_split_count += 1
                for node in chunk:
                    prototype_key_by_node[int(node)] = split_key
    base_class_member_counts = Counter(
        (int(_prototype_count_fields(tuple(key))[0]), int(_prototype_count_fields(tuple(key))[1]))
        for key in base_prototype_key_by_node.values()
    )
    assignment = np.empty(original_graph.num_nodes, dtype=np.int64)
    super_types: list[int] = []
    background_by_type: dict[int, int] = {}
    prototype_by_key: dict[tuple[int, ...], int] = {}
    prototype_type_counts: dict[int, int] = {}
    prototype_members: dict[int, list[int]] = defaultdict(list)
    selected_by_type: dict[int, int] = {}
    prototype_budget_conflict_count = 0
    prototype_fallback_member_count = 0
    rare_class_fallback_count = 0
    fallback_keys: set[tuple[int, ...]] = set()

    for node in target_nodes:
        assignment[int(node)] = len(super_types)
        super_types.append(int(original_graph.node_type[int(node)]))
    for node in support_nodes:
        node = int(node)
        type_id = int(original_graph.node_type[node])
        if node in selected:
            assignment[node] = len(super_types)
            super_types.append(type_id)
            selected_by_type[type_id] = selected_by_type.get(type_id, 0) + 1
            continue
        if node in forced_raw_nodes:
            assignment[node] = len(super_types)
            super_types.append(type_id)
            continue
        if not cfg.allow_background_bucket:
            assignment[node] = len(super_types)
            super_types.append(type_id)
            continue
        if strategy in {"class_anchor_relation_prototype", "dblp_aware_prototype"}:
            key = prototype_key_by_node.get(node, base_prototype_key_by_node.get(node, _prototype_key(node, type_id, support_features)))
            type_count = int(prototype_type_counts.get(type_id, 0))
            if key not in prototype_by_key and type_count >= int(cfg.max_prototypes_per_type):
                parsed = _prototype_count_fields(tuple(key))
                class_member_count = int(base_class_member_counts.get((int(parsed[0]), int(parsed[1])), 0))
                rare_class = class_member_count <= max(
                    1,
                    int(getattr(cfg, "min_prototype_per_class", getattr(cfg, "rare_class_min_prototypes", 1))),
                )
                if not (bool(getattr(cfg, "rare_class_never_fallback", False)) and rare_class):
                    prototype_budget_conflict_count += 1
                    prototype_fallback_member_count += 1
                    if rare_class:
                        rare_class_fallback_count += 1
                    key = (type_id, -1, -1, -1, 0)
                    fallback_keys.add(tuple(key))
            if key not in prototype_by_key:
                prototype_by_key[key] = len(super_types)
                super_types.append(type_id)
                prototype_type_counts[type_id] = int(prototype_type_counts.get(type_id, 0)) + 1
            assignment[node] = prototype_by_key[key]
            prototype_members[prototype_by_key[key]].append(node)
            continue
        if type_id not in background_by_type:
            background_by_type[type_id] = len(super_types)
            super_types.append(type_id)
        assignment[node] = background_by_type[type_id]

    assignment_obj = Assignment(assignment, np.asarray(super_types, dtype=np.int32))
    coarse = coarsen_graph(original_graph, assignment_obj)
    background_nodes = set(background_by_type.values()) | set(prototype_by_key.values())
    background_edges_by_relation = {}
    edge_retention_by_relation = {}
    for relation_id, rel in coarse.relations.items():
        incident = np.isin(rel.src, list(background_nodes)) | np.isin(rel.dst, list(background_nodes))
        background_edges_by_relation[str(int(relation_id))] = int(np.count_nonzero(incident))
        original_edges = int(original_graph.relations[int(relation_id)].num_edges)
        edge_retention_by_relation[str(int(relation_id))] = float(rel.num_edges / max(original_edges, 1))
    parsed_keys = [_prototype_count_fields(tuple(key)) for key in prototype_by_key]
    prototype_count_by_type = Counter(str(item[0]) for item in parsed_keys)
    prototype_count_by_class = Counter(str(item[1]) for item in parsed_keys)
    prototype_count_by_anchor = Counter(str(item[2]) for item in parsed_keys)
    prototype_count_by_relation = Counter(str(item[3]) for item in parsed_keys)
    member_counts = [len(value) for value in prototype_members.values()]
    cap = max(1, int(cfg.max_members_per_prototype))
    saturated = [count for count in member_counts if int(count) >= cap]
    rare_class_min = max(1, int(getattr(cfg, "min_prototype_per_class", getattr(cfg, "rare_class_min_prototypes", 1))))
    relation_min = max(1, int(getattr(cfg, "min_prototype_per_relation_channel", getattr(cfg, "per_relation_min_prototypes", 1))))
    diagnostics = {
        "background_node_count": int(len(background_by_type)),
        "prototype_background_count": int(len(prototype_by_key)),
        "prototype_count_by_type": dict(prototype_count_by_type),
        "prototype_count_by_class": dict(prototype_count_by_class),
        "prototype_count_by_anchor": dict(prototype_count_by_anchor),
        "prototype_count_by_relation_bucket": dict(prototype_count_by_relation),
        "prototype_member_count_mean": float(np.mean(member_counts)) if member_counts else 0.0,
        "prototype_member_count_p50": float(np.percentile(member_counts, 50)) if member_counts else 0.0,
        "prototype_member_count_p90": float(np.percentile(member_counts, 90)) if member_counts else 0.0,
        "prototype_member_count_p99": float(np.percentile(member_counts, 99)) if member_counts else 0.0,
        "prototype_member_count_max": int(max(member_counts)) if member_counts else 0,
        "large_prototype_count": int(large_prototype_count),
        "large_prototype_split_count": int(large_prototype_split_count),
        "forced_raw_bridge_count": int(len(forced_raw_nodes)),
        "raw_bridge_by_type": dict(raw_bridge_by_type),
        "raw_bridge_by_relation_channel": dict(raw_bridge_by_relation_channel),
        "rare_class_prototype_count": int(sum(1 for value in prototype_count_by_class.values() if value >= rare_class_min)),
        "rare_class_fallback_count": int(rare_class_fallback_count),
        "relation_channel_prototype_count": int(sum(1 for value in prototype_count_by_relation.values() if value >= relation_min)),
        "prototype_budget_conflict_count": int(prototype_budget_conflict_count),
        "prototype_fallback_member_count": int(prototype_fallback_member_count),
        "fallback_key_count": int(len(fallback_keys)),
        "prototype_saturation_rate": float(len(saturated) / max(len(member_counts), 1)) if member_counts else 0.0,
        "prototype_key_mode": "dblp_aware" if strategy == "dblp_aware_prototype" else "class_anchor_relation",
        "meta_path_channel_source": "relation_bucket_fallback" if strategy == "dblp_aware_prototype" else "class_anchor_relation",
        "background_edges_by_relation": background_edges_by_relation,
        "dropped_support_count": int(len(support_nodes) - len(selected) - len(forced_raw_nodes)),
        "selected_support_count": int(len(selected)),
        "selected_raw_support_count": int(len(selected) + len(forced_raw_nodes)),
        "unselected_support_count": int(len(support_nodes) - len(selected) - len(forced_raw_nodes)),
        "selected_by_type": {str(key): int(value) for key, value in selected_by_type.items()},
        "edge_retention_by_relation": edge_retention_by_relation,
        "support_context_collision_after_condensation": 0.0,
        "coarse_nodes": int(coarse.num_nodes),
        "coarse_edges": int(sum(rel.num_edges for rel in coarse.relations.values())),
        "background_strategy": strategy,
    }
    return coarse, assignment_obj, diagnostics
