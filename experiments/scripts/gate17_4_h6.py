from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.scripts._common import write_csv
from experiments.scripts.run_gate17_1_support_sensitivity import _aligned_tree_tensor, _hash_tensor
from hesf_coarsen.eval.hettree_task import _feature_width, _type_ids, build_semantic_tree_features, enumerate_target_paths
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj


def _round(value: float, digits: int = 12) -> float:
    return float(round(float(value), int(digits)))


def compute_h6_equivalence_fields(
    *,
    mode: str,
    h6_macro_f1: float,
    control_macro_f1: float,
    h6_accuracy: float,
    control_accuracy: float,
    h6_validation_macro_f1: float,
    control_validation_macro_f1: float,
    tree_l2_delta_vs_h6: float,
    tree_cosine_delta_vs_h6: float,
    tree_hash_equal_to_h6: bool,
    coarse_graph_hash_equal_to_h6: bool,
    edge_mass_l1_delta_vs_h6: float,
    edge_mass_linf_delta_vs_h6: float,
    feature_mean_l2_delta_vs_h6: float,
    assignment_equivalent_to_h6: bool,
    selected_jaccard_with_H6: float,
    selected_recall_of_H6: float,
    selected_precision_vs_H6: float,
) -> dict[str, Any]:
    macro_gap = _round(float(control_macro_f1) - float(h6_macro_f1))
    accuracy_gap = _round(float(control_accuracy) - float(h6_accuracy))
    validation_gap = _round(float(control_validation_macro_f1) - float(h6_validation_macro_f1))
    task_close = abs(macro_gap) <= 0.005 and abs(accuracy_gap) <= 0.005
    tree_close = bool(tree_hash_equal_to_h6) or float(tree_l2_delta_vs_h6) <= 1.0e-6
    mass_close = float(edge_mass_l1_delta_vs_h6) <= 1.0e-6 and float(edge_mass_linf_delta_vs_h6) <= 1.0e-6
    feature_close = float(feature_mean_l2_delta_vs_h6) <= 1.0e-6
    assignment_equal = bool(assignment_equivalent_to_h6)
    construction_pass = bool(task_close and tree_close and mass_close and feature_close and assignment_equal)
    return {
        "mode": str(mode),
        "h6_macro_f1": float(h6_macro_f1),
        "control_macro_f1": float(control_macro_f1),
        "macro_gap_vs_h6": macro_gap,
        "h6_accuracy": float(h6_accuracy),
        "control_accuracy": float(control_accuracy),
        "accuracy_gap_vs_h6": accuracy_gap,
        "h6_validation_macro_f1": float(h6_validation_macro_f1),
        "control_validation_macro_f1": float(control_validation_macro_f1),
        "validation_gap_vs_h6": validation_gap,
        "tree_l2_delta_vs_h6": float(tree_l2_delta_vs_h6),
        "tree_cosine_delta_vs_h6": float(tree_cosine_delta_vs_h6),
        "tree_hash_equal_to_h6": bool(tree_hash_equal_to_h6),
        "coarse_graph_hash_equal_to_h6": bool(coarse_graph_hash_equal_to_h6),
        "edge_mass_l1_delta_vs_h6": float(edge_mass_l1_delta_vs_h6),
        "edge_mass_linf_delta_vs_h6": float(edge_mass_linf_delta_vs_h6),
        "feature_mean_l2_delta_vs_h6": float(feature_mean_l2_delta_vs_h6),
        "assignment_equivalent_to_h6": bool(assignment_equivalent_to_h6),
        "selected_jaccard_with_H6": float(selected_jaccard_with_H6),
        "selected_recall_of_H6": float(selected_recall_of_H6),
        "selected_precision_vs_H6": float(selected_precision_vs_H6),
        "construction_equivalence_pass": construction_pass,
        "h6_construction_gap_detected": bool(not construction_pass),
    }


def relation_edge_mass(graph: HeteroGraph) -> dict[str, float]:
    out: dict[str, float] = {}
    for relation_id, rel in sorted(graph.relations.items()):
        if rel.weight is None:
            mass = float(rel.num_edges)
        else:
            mass = float(np.sum(np.asarray(rel.weight, dtype=np.float64)))
        out[str(int(relation_id))] = mass
    return out


def type_feature_means(graph: HeteroGraph) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for type_id, feature in sorted(graph.features.items()):
        arr = np.asarray(feature, dtype=np.float32)
        if arr.size == 0:
            out[str(int(type_id))] = []
        else:
            out[str(int(type_id))] = np.mean(arr, axis=0).astype(float).tolist()
    return out


def edge_mass_delta(control: HeteroGraph, reference: HeteroGraph) -> dict[str, Any]:
    control_mass = relation_edge_mass(control)
    reference_mass = relation_edge_mass(reference)
    keys = sorted(set(control_mass) | set(reference_mass))
    deltas = [float(control_mass.get(key, 0.0) - reference_mass.get(key, 0.0)) for key in keys]
    return {
        "edge_mass_l1_delta_vs_h6": float(np.sum(np.abs(deltas))) if deltas else 0.0,
        "edge_mass_linf_delta_vs_h6": float(np.max(np.abs(deltas))) if deltas else 0.0,
        "edge_mass_by_relation": json.dumps(control_mass, sort_keys=True),
    }


def feature_mean_delta(control: HeteroGraph, reference: HeteroGraph) -> dict[str, Any]:
    control_means = type_feature_means(control)
    reference_means = type_feature_means(reference)
    parts: list[np.ndarray] = []
    for key in sorted(set(control_means) | set(reference_means)):
        left = np.asarray(control_means.get(key, []), dtype=np.float32)
        right = np.asarray(reference_means.get(key, []), dtype=np.float32)
        width = max(left.size, right.size)
        if width == 0:
            continue
        padded_left = np.zeros(width, dtype=np.float32)
        padded_right = np.zeros(width, dtype=np.float32)
        padded_left[: left.size] = left
        padded_right[: right.size] = right
        parts.append(padded_left - padded_right)
    delta = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return {
        "feature_mean_l2_delta_vs_h6": float(np.linalg.norm(delta.reshape(-1))) if delta.size else 0.0,
        "feature_mean_by_type": json.dumps(control_means, sort_keys=True),
    }


def coarse_graph_hash(graph: HeteroGraph) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(graph.node_type.astype(np.int32)).tobytes())
    for type_id, feature in sorted(graph.features.items()):
        digest.update(str(int(type_id)).encode("ascii"))
        digest.update(np.ascontiguousarray(np.asarray(feature, dtype=np.float32)).tobytes())
    for relation_id, rel in sorted(graph.relations.items()):
        digest.update(str(int(relation_id)).encode("ascii"))
        digest.update(np.ascontiguousarray(np.asarray(rel.src, dtype=np.int64)).tobytes())
        digest.update(np.ascontiguousarray(np.asarray(rel.dst, dtype=np.int64)).tobytes())
        if rel.weight is not None:
            digest.update(np.ascontiguousarray(np.asarray(rel.weight, dtype=np.float32)).tobytes())
    return digest.hexdigest()[:16]


def semantic_delta_vs_h6(
    *,
    original: HeteroGraph,
    control: HeteroGraph,
    control_assignment: np.ndarray,
    h6: HeteroGraph,
    h6_assignment: np.ndarray,
    target_type: int,
    max_paths: int,
) -> dict[str, Any]:
    paths = enumerate_target_paths(original, target_type=int(target_type), max_paths=int(max_paths))
    width = _feature_width([original, control, h6])
    ids = _type_ids([original, control, h6])
    control_tree = build_semantic_tree_features(control, target_type=int(target_type), paths=paths, feature_width=width, type_ids=ids)
    h6_tree = build_semantic_tree_features(h6, target_type=int(target_type), paths=paths, feature_width=width, type_ids=ids)
    original_targets = np.flatnonzero(original.node_type == int(target_type)).astype(np.int64)
    control_aligned = _aligned_tree_tensor(control_tree, original_targets, np.asarray(control_assignment, dtype=np.int64))
    h6_aligned = _aligned_tree_tensor(h6_tree, original_targets, np.asarray(h6_assignment, dtype=np.int64))
    delta = control_aligned - h6_aligned
    flat_control = control_aligned.reshape(-1)
    flat_h6 = h6_aligned.reshape(-1)
    denom = max(float(np.linalg.norm(flat_control) * np.linalg.norm(flat_h6)), 1.0e-12)
    return {
        "control_tree_hash": _hash_tensor(control_aligned),
        "h6_tree_hash": _hash_tensor(h6_aligned),
        "tree_l2_delta_vs_h6": float(np.linalg.norm(delta.reshape(-1))),
        "tree_cosine_delta_vs_h6": float(1.0 - float(np.dot(flat_control, flat_h6)) / denom),
        "tree_hash_equal_to_h6": bool(_hash_tensor(control_aligned) == _hash_tensor(h6_aligned)),
    }


def assignment_cluster_members(assignment: np.ndarray, graph: HeteroGraph) -> list[dict[str, Any]]:
    arr = np.asarray(assignment, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for supernode in sorted(int(value) for value in np.unique(arr)):
        members = np.flatnonzero(arr == int(supernode)).astype(np.int64)
        rows.append(
            {
                "supernode": int(supernode),
                "node_type": int(graph.node_type[int(members[0])]) if len(members) else -1,
                "member_count": int(len(members)),
                "members": json.dumps([int(node) for node in members.tolist()], separators=(",", ":")),
            }
        )
    return rows


def selected_support_representatives_from_assignment(graph: HeteroGraph, assignment: np.ndarray, target_type: int) -> np.ndarray:
    support_nodes = np.flatnonzero(graph.node_type != int(target_type)).astype(np.int64)
    groups: dict[int, list[int]] = {}
    arr = np.asarray(assignment, dtype=np.int64)
    for node in support_nodes:
        groups.setdefault(int(arr[int(node)]), []).append(int(node))
    reps = [min(nodes) for nodes in groups.values() if nodes]
    return np.asarray(sorted(reps), dtype=np.int64)


def export_h6_artifacts(
    *,
    output_dir: Path,
    dataset: str,
    seed: int,
    ratio: float,
    original: HeteroGraph,
    h6: HeteroGraph,
    h6_assignment: np.ndarray,
    target_type: int,
    max_paths: int,
) -> dict[str, Any]:
    token = f"{dataset}_seed{int(seed)}_ratio{float(ratio):.2f}".replace(".", "p")
    artifact_dir = output_dir / token
    artifact_dir.mkdir(parents=True, exist_ok=True)
    assignment = np.asarray(h6_assignment, dtype=np.int64)
    np.save(artifact_dir / "h6_assignment.npy", assignment)
    selected = selected_support_representatives_from_assignment(original, assignment, int(target_type))
    np.save(artifact_dir / "h6_selected_support_nodes.npy", selected)
    write_csv(artifact_dir / "h6_cluster_members.csv", assignment_cluster_members(assignment, original))
    write_csv(
        artifact_dir / "h6_coarse_node_map.csv",
        [
            {"coarse_node": int(node), "node_type": int(h6.node_type[int(node)])}
            for node in range(int(h6.num_nodes))
        ],
    )
    edge_rows: list[dict[str, Any]] = []
    for relation_id, rel in sorted(h6.relations.items()):
        weights = np.ones(rel.num_edges, dtype=np.float32) if rel.weight is None else np.asarray(rel.weight, dtype=np.float32)
        for src, dst, weight in zip(rel.src, rel.dst, weights):
            edge_rows.append({"relation_id": int(relation_id), "src": int(src), "dst": int(dst), "weight": float(weight)})
    write_csv(artifact_dir / "h6_coarse_graph_edges.csv", edge_rows)
    write_csv(
        artifact_dir / "h6_relation_edge_mass_by_relation.csv",
        [{"relation_id": key, "edge_mass": value} for key, value in relation_edge_mass(h6).items()],
    )
    write_csv(
        artifact_dir / "h6_type_feature_mean_by_type.csv",
        [{"type_id": key, "feature_mean": json.dumps(value)} for key, value in type_feature_means(h6).items()],
    )
    paths = enumerate_target_paths(original, target_type=int(target_type), max_paths=int(max_paths))
    width = _feature_width([original, h6])
    ids = _type_ids([original, h6])
    h6_tree = build_semantic_tree_features(h6, target_type=int(target_type), paths=paths, feature_width=width, type_ids=ids)
    original_targets = np.flatnonzero(original.node_type == int(target_type)).astype(np.int64)
    aligned = _aligned_tree_tensor(h6_tree, original_targets, assignment)
    tree_hash = _hash_tensor(aligned)
    (artifact_dir / "h6_semantic_tree_hash.txt").write_text(tree_hash + "\n", encoding="utf-8")
    checksum = {
        "semantic_tree_hash": tree_hash,
        "semantic_tree_shape": list(aligned.shape),
        "semantic_tree_sum": float(np.sum(aligned, dtype=np.float64)),
        "semantic_tree_l2": float(np.linalg.norm(aligned.reshape(-1))),
    }
    (artifact_dir / "h6_semantic_tree_checksum.json").write_text(json.dumps(checksum, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "h6_selected_support_count": int(len(selected)),
        "h6_semantic_tree_hash": tree_hash,
        "h6_coarse_graph_hash": coarse_graph_hash(h6),
    }


def write_artifact_limitations(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Gate17.4 H6 Artifact Limitations",
                "",
                "- `h6_selected_support_nodes.npy` stores one representative original support node per H6 support cluster; it is not the full H6 construction.",
                "- H6 relation-channel summaries are exported as relation edge-mass tables because the baseline candidate store does not retain a semantic channel label per retained cluster.",
                "- Task metric equality uses deterministic seeding but HETTREE training is still treated with a stochastic tolerance in the Gate17.4 report.",
                "- H6 construction equivalence is therefore judged primarily by assignment, tree hash/delta, edge mass, and feature mean equality.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def induced_coarse_graph(
    coarse: HeteroGraph,
    original_assignment: np.ndarray,
    keep_coarse_nodes: np.ndarray,
) -> tuple[HeteroGraph, np.ndarray]:
    kept = np.asarray(sorted(int(node) for node in np.asarray(keep_coarse_nodes, dtype=np.int64).reshape(-1)), dtype=np.int64)
    local_of = {int(node): idx for idx, node in enumerate(kept.tolist())}
    relations: dict[int, RelationAdj] = {}
    for relation_id, rel in coarse.relations.items():
        mask = np.asarray([int(src) in local_of and int(dst) in local_of for src, dst in zip(rel.src, rel.dst)], dtype=bool)
        src = np.asarray([local_of[int(node)] for node in np.asarray(rel.src)[mask]], dtype=np.int64)
        dst = np.asarray([local_of[int(node)] for node in np.asarray(rel.dst)[mask]], dtype=np.int64)
        weight = np.asarray(rel.weight, dtype=np.float32)[mask] if rel.weight is not None else None
        relations[int(relation_id)] = RelationAdj(
            src=src,
            dst=dst,
            weight=weight,
            src_type=int(rel.src_type),
            dst_type=int(rel.dst_type),
            relation_id=int(relation_id),
        )
    features: dict[int, np.ndarray] = {}
    for type_id, feature in coarse.features.items():
        type_nodes = np.flatnonzero(coarse.node_type == int(type_id)).astype(np.int64)
        local_lookup = {int(node): idx for idx, node in enumerate(type_nodes.tolist())}
        type_kept = [int(node) for node in kept.tolist() if int(coarse.node_type[int(node)]) == int(type_id)]
        indices = [local_lookup[int(node)] for node in type_kept]
        features[int(type_id)] = np.asarray(feature, dtype=np.float32)[indices].astype(np.float32, copy=False)
    graph = HeteroGraph(
        num_nodes=int(len(kept)),
        node_type=coarse.node_type[kept].astype(np.int32, copy=False),
        relations=relations,
        relation_specs=dict(coarse.relation_specs),
        features=features,
        labels=None if coarse.labels is None else np.asarray(coarse.labels)[kept],
        partitions=None if coarse.partitions is None else np.asarray(coarse.partitions)[kept],
    )
    original_assignment = np.asarray(original_assignment, dtype=np.int64)
    mapped = np.zeros_like(original_assignment, dtype=np.int64)
    for original_node, supernode in enumerate(original_assignment.tolist()):
        if int(supernode) in local_of:
            mapped[int(original_node)] = int(local_of[int(supernode)])
    return graph, mapped
