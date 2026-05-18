from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.sketch.chebyshev import chebyshev_heat_filter


def stable_probe_seed(dataset: str, seed: int, namespace: str = "holdout_operator") -> int:
    payload = f"{dataset}:{int(seed)}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def make_holdout_probe_matrix(
    graph: HeteroGraph,
    *,
    dataset: str,
    seed: int,
    probe_dim: int,
    namespace: str = "holdout_operator",
) -> np.ndarray:
    rng = np.random.default_rng(stable_probe_seed(dataset, int(seed), namespace))
    probes = np.zeros((graph.num_nodes, int(probe_dim)), dtype=np.float32)
    for type_id in sorted(np.unique(graph.node_type).astype(int).tolist()):
        nodes = nodes_of_type(graph, int(type_id))
        if len(nodes) == 0:
            continue
        typed = rng.standard_normal((len(nodes), int(probe_dim))).astype(np.float32)
        typed -= typed.mean(axis=0, keepdims=True)
        denom = np.maximum(typed.std(axis=0, keepdims=True), 1.0e-6)
        probes[nodes] = typed / denom
    return probes


def apply_fused_operator_response(
    graph: HeteroGraph,
    probes: np.ndarray,
    *,
    cheb_order: int = 5,
    heat_time: float = 1.0,
) -> np.ndarray:
    """Apply a bounded low-pass fused relation operator by sparse relation matvecs."""

    relation_ids = sorted(graph.relations)
    uniform = 1.0 / max(len(relation_ids), 1)
    relation_weights = {int(relation_id): uniform for relation_id in relation_ids}
    return chebyshev_heat_filter(
        graph,
        np.asarray(probes, dtype=np.float32),
        relation_weights,
        heat_time=float(heat_time),
        order=int(cheb_order),
        symmetric_relation_operator=True,
        symmetric_relation_scale=0.5,
        reverse_relation_policy="include_all",
        relation_operator_mode="relationwise",
        progress_config={"enabled": False},
        progress_desc="held-out operator probe",
    )


def project_to_coarse(values: np.ndarray, assignment: Assignment, reduce: str = "mean") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros((assignment.num_supernodes, values.shape[1]), dtype=np.float32)
    np.add.at(out, assignment.assignment, values)
    if reduce == "mean":
        out /= np.maximum(assignment.cluster_sizes().astype(np.float32)[:, None], 1.0)
    elif reduce != "sum":
        raise ValueError("reduce must be 'mean' or 'sum'")
    return out


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    denom = max(float(np.linalg.norm(np.asarray(a, dtype=np.float64))), 1.0e-12)
    return float(np.linalg.norm(diff) / denom)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    flat_a = np.asarray(a, dtype=np.float64).reshape(-1)
    flat_b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(flat_a, flat_b) / max(float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b)), 1.0e-12))


def _energy_error(a: np.ndarray, b: np.ndarray) -> float:
    ea = float(np.sum(np.asarray(a, dtype=np.float64) ** 2))
    eb = float(np.sum(np.asarray(b, dtype=np.float64) ** 2))
    return float(abs(eb - ea) / max(abs(ea), 1.0e-12))


def _dirichlet_energy(graph: HeteroGraph, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = 0.0
    for rel in graph.relations.values():
        if rel.num_edges == 0:
            continue
        diff = values[rel.src] - values[rel.dst]
        total += float(np.sum(rel.weight.astype(np.float64, copy=False) * np.sum(diff * diff, axis=1)))
    return total


def _relation_energy(graph: HeteroGraph, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    out: dict[str, float] = {}
    for relation_id, rel in graph.relations.items():
        if rel.num_edges == 0:
            out[str(int(relation_id))] = 0.0
            continue
        diff = values[rel.src] - values[rel.dst]
        out[str(int(relation_id))] = float(np.sum(rel.weight.astype(np.float64, copy=False) * np.sum(diff * diff, axis=1)))
    return out


def _json_relative_by_type(original_graph: HeteroGraph, original: np.ndarray, lifted: np.ndarray) -> str:
    out: dict[str, float] = {}
    for type_id in sorted(np.unique(original_graph.node_type).astype(int).tolist()):
        nodes = nodes_of_type(original_graph, int(type_id))
        out[str(type_id)] = _relative_error(original[nodes], lifted[nodes]) if len(nodes) else 0.0
    return json.dumps(out, sort_keys=True)


def _assignment_hash(assignment: Assignment) -> str:
    digest = hashlib.sha256()
    digest.update(str(assignment.assignment.shape).encode("ascii"))
    digest.update(assignment.assignment.astype(np.int64, copy=False).tobytes(order="C"))
    digest.update(assignment.supernode_type.astype(np.int32, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def evaluate_holdout_operator_probe(
    original_graph: HeteroGraph,
    coarse_graph: HeteroGraph,
    assignment: Assignment,
    *,
    dataset: str,
    seed: int,
    probe_dim: int = 32,
    cheb_order: int = 5,
    heat_time: float = 1.0,
    probe_namespace: str = "holdout_operator",
) -> dict[str, Any]:
    probes = make_holdout_probe_matrix(
        original_graph,
        dataset=dataset,
        seed=int(seed),
        probe_dim=int(probe_dim),
        namespace=probe_namespace,
    )
    original_response = apply_fused_operator_response(
        original_graph,
        probes,
        cheb_order=int(cheb_order),
        heat_time=float(heat_time),
    )
    coarse_probes = project_to_coarse(probes, assignment)
    coarse_response = apply_fused_operator_response(
        coarse_graph,
        coarse_probes,
        cheb_order=int(cheb_order),
        heat_time=float(heat_time),
    )
    projected_original = project_to_coarse(original_response, assignment)
    lifted = coarse_response[assignment.assignment]
    coarse_rel_energy = _relation_energy(coarse_graph, coarse_response)
    projected_rel_energy = _relation_energy(coarse_graph, projected_original)
    relation_errors = []
    for key, value in projected_rel_energy.items():
        relation_errors.append(abs(coarse_rel_energy.get(key, 0.0) - value) / max(abs(value), 1.0e-12))
    orig_dirichlet = _dirichlet_energy(original_graph, original_response)
    lifted_dirichlet = _dirichlet_energy(original_graph, lifted)
    config_hash = hashlib.sha256(f"{probe_namespace}:{probe_dim}:{cheb_order}:{heat_time}".encode("ascii")).hexdigest()
    return {
        "holdout_operator_relative_error": _relative_error(projected_original, coarse_response),
        "holdout_operator_lifted_relative_error": _relative_error(original_response, lifted),
        "holdout_operator_cosine_similarity": _cosine(projected_original, coarse_response),
        "holdout_operator_energy_error": _energy_error(projected_original, coarse_response),
        "holdout_operator_dirichlet_error": float(abs(lifted_dirichlet - orig_dirichlet) / max(abs(orig_dirichlet), 1.0e-12)),
        "holdout_operator_typewise_relative_error": _json_relative_by_type(original_graph, original_response, lifted),
        "holdout_operator_relation_energy_error": float(np.mean(relation_errors)) if relation_errors else 0.0,
        "num_holdout_probes": int(probe_dim),
        "cheb_order": int(cheb_order),
        "heat_time": float(heat_time),
        "probe_seed": int(stable_probe_seed(dataset, int(seed), probe_namespace)),
        "probe_namespace": probe_namespace,
        "assignment_hash": _assignment_hash(assignment),
        "config_hash": config_hash,
    }
