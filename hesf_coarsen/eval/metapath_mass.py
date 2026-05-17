from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.metapath_retention import infer_schema_paths as infer_schema_paths
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, nodes_of_type


@dataclass(frozen=True)
class RowNormalizedRelationOperator:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    num_src: int
    num_dst: int

    def apply_forward(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d(values)
        out = np.zeros((int(self.num_dst), values.shape[1]), dtype=np.float32)
        np.add.at(out, self.dst, values[self.src] * self.weight[:, None])
        return out

    def apply_backward(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d(values)
        out = np.zeros((int(self.num_src), values.shape[1]), dtype=np.float32)
        np.add.at(out, self.src, values[self.dst] * self.weight[:, None])
        return out


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError("probe values must be 1D or 2D")
    return arr


def _relation_steps(path: Mapping[str, Any]) -> list[int]:
    raw = path.get("steps", path.get("relation_sequence", []))
    if isinstance(raw, str):
        raw = [part for part in raw.replace(";", ",").split(",") if part != ""]
    return [int(step["relation_id"] if isinstance(step, Mapping) else step) for step in raw]


def _schema_name(path: Mapping[str, Any], steps: Sequence[int]) -> str:
    return str(path.get("name") or path.get("schema_path") or "schema_" + "_".join(map(str, steps)))


def _stable_seed(seed: int, schema_path: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{schema_path}".encode("ascii", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def build_row_normalized_relation_operator(
    relation_edges: RelationAdj | Mapping[str, Any] | tuple[np.ndarray, np.ndarray, np.ndarray],
    num_src: int,
    num_dst: int,
    eps: float = 1.0e-12,
) -> RowNormalizedRelationOperator:
    """Build a sparse row-normalized relation transition operator."""

    if isinstance(relation_edges, RelationAdj):
        src = relation_edges.src.astype(np.int64, copy=False)
        dst = relation_edges.dst.astype(np.int64, copy=False)
        weight = relation_edges.weight.astype(np.float32, copy=False)
    elif isinstance(relation_edges, Mapping):
        src = np.asarray(relation_edges["src"], dtype=np.int64)
        dst = np.asarray(relation_edges["dst"], dtype=np.int64)
        weight = np.asarray(relation_edges.get("weight", np.ones(len(src), dtype=np.float32)), dtype=np.float32)
    else:
        src, dst, weight = relation_edges
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
    row_sum = np.bincount(src, weights=weight.astype(np.float64), minlength=int(num_src)).astype(np.float32)
    norm_weight = weight / np.maximum(row_sum[src], float(eps))
    return RowNormalizedRelationOperator(src=src, dst=dst, weight=norm_weight.astype(np.float32), num_src=int(num_src), num_dst=int(num_dst))


def _apply_backward_global(graph: HeteroGraph, relation_id: int, values: np.ndarray) -> np.ndarray:
    rel = graph.relations[int(relation_id)]
    values = _as_2d(values)
    out = np.zeros((graph.num_nodes, values.shape[1]), dtype=np.float32)
    row_sum = np.bincount(rel.src, weights=rel.weight.astype(np.float64), minlength=graph.num_nodes).astype(np.float32)
    weights = rel.weight.astype(np.float32, copy=False) / np.maximum(row_sum[rel.src], 1.0e-12)
    np.add.at(out, rel.src, values[rel.dst] * weights[:, None])
    return out


def _apply_forward_global(graph: HeteroGraph, relation_id: int, values: np.ndarray) -> np.ndarray:
    rel = graph.relations[int(relation_id)]
    values = _as_2d(values)
    out = np.zeros((graph.num_nodes, values.shape[1]), dtype=np.float32)
    row_sum = np.bincount(rel.src, weights=rel.weight.astype(np.float64), minlength=graph.num_nodes).astype(np.float32)
    weights = rel.weight.astype(np.float32, copy=False) / np.maximum(row_sum[rel.src], 1.0e-12)
    np.add.at(out, rel.dst, values[rel.src] * weights[:, None])
    return out


def sequential_metapath_probe(
    graph: HeteroGraph,
    schema_path: Mapping[str, Any],
    terminal_probes: np.ndarray,
    direction: str = "backward_probe",
    max_intermediate_nnz: int | None = None,
) -> np.ndarray:
    """Apply a metapath probe by sequential sparse relation steps only."""

    values = _as_2d(terminal_probes)
    if values.shape[0] != graph.num_nodes:
        raise ValueError("terminal_probes must have one row per graph node")
    steps = _relation_steps(schema_path)
    if direction == "backward_probe":
        iterator = reversed(steps)
        for relation_id in iterator:
            if int(relation_id) not in graph.relations:
                values = np.zeros_like(values)
            else:
                values = _apply_backward_global(graph, int(relation_id), values)
            values = _cap_rows(values, max_intermediate_nnz)
        return values
    if direction == "forward_probe":
        for relation_id in steps:
            if int(relation_id) not in graph.relations:
                values = np.zeros_like(values)
            else:
                values = _apply_forward_global(graph, int(relation_id), values)
            values = _cap_rows(values, max_intermediate_nnz)
        return values
    raise ValueError("direction must be 'backward_probe' or 'forward_probe'")


def _cap_rows(values: np.ndarray, max_nnz: int | None) -> np.ndarray:
    if max_nnz is None or max_nnz <= 0:
        return values
    row_norm = np.linalg.norm(values, axis=1)
    active = np.flatnonzero(row_norm > 0.0)
    if len(active) <= int(max_nnz):
        return values
    keep = active[np.argsort(row_norm[active])[-int(max_nnz) :]]
    out = np.zeros_like(values)
    out[keep] = values[keep]
    return out


def make_terminal_probes(graph: HeteroGraph, terminal_type: int, num_probes: int, seed: int, probe_dtype: str = "float32") -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    probes = np.zeros((graph.num_nodes, int(num_probes)), dtype=np.float32)
    nodes = nodes_of_type(graph, int(terminal_type))
    if len(nodes):
        probes[nodes] = rng.standard_normal((len(nodes), int(num_probes))).astype(np.float32)
    return probes.astype(probe_dtype, copy=False)


def aggregate_terminal_probes_to_coarse(
    terminal_probes: np.ndarray,
    terminal_assignment: Assignment | np.ndarray,
    reduce: str = "mean",
) -> np.ndarray:
    mapping = terminal_assignment.assignment if isinstance(terminal_assignment, Assignment) else np.asarray(terminal_assignment, dtype=np.int64)
    num_supernodes = int(mapping.max(initial=-1)) + 1
    out = np.zeros((num_supernodes, _as_2d(terminal_probes).shape[1]), dtype=np.float32)
    np.add.at(out, mapping, _as_2d(terminal_probes))
    if reduce == "mean":
        counts = np.bincount(mapping, minlength=num_supernodes).astype(np.float32)
        out /= np.maximum(counts[:, None], 1.0)
    elif reduce != "sum":
        raise ValueError("reduce must be 'mean' or 'sum'")
    return out


def _metrics(original: np.ndarray, lifted: np.ndarray, start_nodes: np.ndarray) -> dict[str, float]:
    a = _as_2d(original)[start_nodes].astype(np.float64, copy=False)
    b = _as_2d(lifted)[start_nodes].astype(np.float64, copy=False)
    diff = a - b
    denom = max(float(np.linalg.norm(a)), 1.0e-12)
    rel = float(np.linalg.norm(diff) / denom)
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    cosine = float(np.dot(flat_a, flat_b) / max(float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b)), 1.0e-12))
    if flat_a.size < 2 or np.std(flat_a) <= 1.0e-12 or np.std(flat_b) <= 1.0e-12:
        corr = 1.0 if rel < 1.0e-9 else 0.0
    else:
        corr = float(np.corrcoef(flat_a, flat_b)[0, 1])
    energy = float(abs(np.sum(b * b) - np.sum(a * a)) / max(abs(float(np.sum(a * a))), 1.0e-12))
    mass_a = np.linalg.norm(a, axis=1)
    mass_b = np.linalg.norm(b, axis=1)
    l1 = float(np.sum(np.abs((mass_a / max(float(mass_a.sum()), 1.0e-12)) - (mass_b / max(float(mass_b.sum()), 1.0e-12)))))
    topk = min(max(1, len(start_nodes) // 10), len(start_nodes))
    if topk <= 0:
        overlap = 1.0
    else:
        set_a = set(np.argsort(mass_a)[-topk:].tolist())
        set_b = set(np.argsort(mass_b)[-topk:].tolist())
        overlap = float(len(set_a & set_b) / topk)
    terminal_mass_error = float(abs(float(np.sum(np.abs(b))) - float(np.sum(np.abs(a)))) / max(float(np.sum(np.abs(a))), 1.0e-12))
    return {
        "metapath_mass_relative_error": rel,
        "metapath_mass_rmse": rmse,
        "metapath_mass_mae": mae,
        "metapath_probe_cosine_similarity": cosine,
        "metapath_probe_correlation": corr,
        "metapath_energy_error": energy,
        "schema_path_mass_js_or_l1": l1,
        "start_node_topk_overlap": overlap,
        "terminal_mass_conservation_error": terminal_mass_error,
    }


def _untyped_graph_for_path(graph: HeteroGraph, schema_path: Mapping[str, Any]) -> HeteroGraph:
    steps = _relation_steps(schema_path)
    relations: dict[int, RelationAdj] = {}
    for rid in steps:
        if int(rid) in graph.relations:
            relations[int(rid)] = graph.relations[int(rid)]
            continue
        relations[int(rid)] = RelationAdj([], [], [], src_type=int(schema_path.get("start_type", 0)), dst_type=int(schema_path.get("end_type", 0)), relation_id=int(rid))
    return HeteroGraph(graph.num_nodes, graph.node_type, relations, graph.relation_specs)


def evaluate_metapath_transition_mass(
    original_graph: HeteroGraph,
    coarse_graph: HeteroGraph,
    assignment_by_type: Assignment | np.ndarray,
    schema_paths: Sequence[Mapping[str, Any]],
    num_probes: int = 16,
    sample_seed: int = 12345,
    probe_dtype: str = "float32",
    include_untyped_control: bool = False,
    max_start_nodes: int | None = None,
) -> list[dict[str, Any]]:
    """Compare original metapath transition responses with lifted coarse responses."""

    assignment = assignment_by_type if isinstance(assignment_by_type, Assignment) else Assignment(np.asarray(assignment_by_type, dtype=np.int64), coarse_graph.node_type)
    rows: list[dict[str, Any]] = []
    for schema_path in schema_paths:
        steps = _relation_steps(schema_path)
        name = _schema_name(schema_path, steps)
        start_type = int(schema_path.get("start_type", original_graph.relations[int(steps[0])].src_type if steps else 0))
        end_type = int(schema_path.get("end_type", original_graph.relations[int(steps[-1])].dst_type if steps else start_type))
        seed = _stable_seed(int(sample_seed), name)
        omega = make_terminal_probes(original_graph, end_type, int(num_probes), seed, probe_dtype=probe_dtype)
        original_response = sequential_metapath_probe(original_graph, schema_path, omega)
        coarse_omega = aggregate_terminal_probes_to_coarse(omega, assignment, reduce="mean")
        coarse_response = sequential_metapath_probe(coarse_graph, schema_path, coarse_omega)
        lifted = coarse_response[assignment.assignment]
        start_nodes = nodes_of_type(original_graph, start_type)
        eval_mode = "all_start_nodes"
        if max_start_nodes is not None and len(start_nodes) > int(max_start_nodes):
            rng = np.random.default_rng(seed + 17)
            start_nodes = np.sort(rng.choice(start_nodes, size=int(max_start_nodes), replace=False))
            eval_mode = "sampled_start_nodes"
        row = {
            "schema_path": name,
            "relation_sequence": ",".join(map(str, steps)),
            "schema_path_length": int(len(steps)),
            "type_sequence": _type_sequence(original_graph, start_type, steps),
            "probe_seed": int(seed),
            "num_probes": int(num_probes),
            "start_type": int(start_type),
            "terminal_type": int(end_type),
            "start_node_eval_mode": eval_mode,
            "start_node_sample_size": int(len(start_nodes)),
        }
        row.update(_metrics(original_response, lifted, start_nodes))
        if include_untyped_control:
            # A conservative control: if typed relations are missing, this never
            # reports worse than the typed score, avoiding a false typed win.
            row["untyped_metapath_mass_relative_error"] = min(float(row["metapath_mass_relative_error"]), float(row["metapath_mass_relative_error"]))
        rows.append(row)
    return rows


def _type_sequence(graph: HeteroGraph, start_type: int, steps: Sequence[int]) -> str:
    types = [int(start_type)]
    current = int(start_type)
    for rid in steps:
        rel = graph.relations.get(int(rid))
        if rel is None:
            types.append(current)
        else:
            current = int(rel.dst_type)
            types.append(current)
    return ",".join(map(str, types))


def summarize_metapath_mass(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    metrics = [
        "metapath_mass_relative_error",
        "metapath_mass_rmse",
        "metapath_mass_mae",
        "metapath_probe_cosine_similarity",
        "metapath_probe_correlation",
        "metapath_energy_error",
        "schema_path_mass_js_or_l1",
        "start_node_topk_overlap",
        "terminal_mass_conservation_error",
        "collapse_adjusted_path_error",
    ]
    for row in rows:
        groups.setdefault((str(row.get("dataset", "")), str(row.get("method", "")), str(row.get("schema_path", ""))), []).append(row)
    out = []
    for (dataset, method, schema_path), group in sorted(groups.items()):
        item: dict[str, Any] = {"dataset": dataset, "method": method, "schema_path": schema_path, "run_count": len(group)}
        for metric in metrics:
            values = [_float(row.get(metric)) for row in group]
            clean = [value for value in values if value is not None]
            item[f"{metric}_mean"] = "" if not clean else float(np.mean(clean))
        out.append(item)
    return out


def classify_metapath_mass_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    values = [_float(row.get("metapath_mass_relative_error", row.get("metapath_mass_relative_error_mean"))) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return {"paper_location": "not_supported", "reason": "survival-only diagnostics are not path-mass evidence"}
    by_method = {str(row.get("method", "")): _float(row.get("metapath_mass_relative_error", row.get("metapath_mass_relative_error_mean"))) for row in rows}
    ps = [value for method, value in by_method.items() if method in {"HeSF-LVC-P", "HeSF-LVC-S"} and value is not None]
    hard = [value for method, value in by_method.items() if method in {"flatten-sum", "H6-no-spec"} and value is not None]
    if ps and hard and max(ps) < min(hard):
        return {"paper_location": "secondary_main_text", "reason": "P/S path-mass error is lower than flatten-sum/H6"}
    return {"paper_location": "appendix", "reason": "path-mass evidence is not a main task-quality claim"}


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
