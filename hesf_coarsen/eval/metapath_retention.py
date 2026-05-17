from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, nodes_of_type


def _relation_steps(path: Mapping[str, Any]) -> list[int]:
    raw = path.get("steps", path.get("relation_sequence", []))
    if isinstance(raw, str):
        if raw.strip().startswith("["):
            raw = json.loads(raw)
        else:
            raw = [part for part in raw.replace(";", ",").split(",") if part != ""]
    steps: list[int] = []
    for step in raw:
        if isinstance(step, Mapping):
            steps.append(int(step["relation_id"]))
        else:
            steps.append(int(step))
    return steps


def _schema_name(path: Mapping[str, Any], steps: Sequence[int]) -> str:
    return str(path.get("name") or path.get("schema_path") or "r" + "_".join(map(str, steps)))


def _node_path(value: Any) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return [int(v) for v in json.loads(text)]
        return [int(part) for part in text.replace(";", ",").split(",") if part != ""]
    return [int(v) for v in value]


def _sequence_text(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _outgoing_by_relation(graph: HeteroGraph) -> dict[int, dict[int, list[tuple[int, float]]]]:
    outgoing: dict[int, dict[int, list[tuple[int, float]]]] = {}
    for relation_id, rel in graph.relations.items():
        buckets: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for src, dst, weight in zip(rel.src, rel.dst, rel.weight):
            buckets[int(src)].append((int(dst), float(weight)))
        outgoing[int(relation_id)] = {node: sorted(values) for node, values in buckets.items()}
    return outgoing


def _edge_lookup_by_relation(graph: HeteroGraph) -> dict[int, dict[tuple[int, int], float]]:
    lookup: dict[int, dict[tuple[int, int], float]] = {}
    for relation_id, rel in graph.relations.items():
        pairs: dict[tuple[int, int], float] = defaultdict(float)
        for src, dst, weight in zip(rel.src, rel.dst, rel.weight):
            pairs[(int(src), int(dst))] += float(weight)
        lookup[int(relation_id)] = dict(pairs)
    return lookup


def _untyped_lookup(graph: HeteroGraph) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for rel in graph.relations.values():
        pairs.update((int(src), int(dst)) for src, dst in zip(rel.src, rel.dst))
    return pairs


def _schema_is_valid(graph: HeteroGraph, path: Mapping[str, Any], steps: Sequence[int]) -> bool:
    current = path.get("start_type", None)
    if current in {None, ""}:
        if not steps or int(steps[0]) not in graph.relations:
            return False
        current = graph.relations[int(steps[0])].src_type
    current_type = int(current)
    for relation_id in steps:
        rel = graph.relations.get(int(relation_id))
        if rel is None or int(rel.src_type) != current_type:
            return False
        current_type = int(rel.dst_type)
    end_type = path.get("end_type", None)
    return end_type in {None, ""} or int(end_type) == current_type


def infer_schema_paths(
    schema: HeteroGraph | Mapping[str, Any],
    target_node_type: int | str | None = None,
    lengths: Sequence[int] = (2, 3),
    max_paths: int = 12,
) -> list[dict[str, Any]]:
    """Infer bounded relation-compatible schema paths.

    The implementation walks the relation schema only; it never constructs
    relation products or dense adjacency. Cyclic paths around the target type
    are emitted first when a target type is supplied.
    """

    graph = schema if isinstance(schema, HeteroGraph) else None
    if graph is None:
        raise TypeError("infer_schema_paths currently expects a HeteroGraph schema")
    target = None if target_node_type in {None, ""} else int(target_node_type)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for relation_id, rel in graph.relations.items():
        outgoing[int(rel.src_type)].append((int(relation_id), int(rel.dst_type)))
    for value in outgoing.values():
        value.sort()

    candidates: list[dict[str, Any]] = []

    def walk(start_type: int, current_type: int, remaining: int, steps: list[int]) -> None:
        if len(candidates) >= max_paths * 4:
            return
        if remaining == 0:
            candidates.append(
                {
                    "name": "schema_" + "_".join(map(str, steps)),
                    "steps": list(steps),
                    "start_type": int(start_type),
                    "end_type": int(current_type),
                }
            )
            return
        for relation_id, next_type in outgoing.get(int(current_type), []):
            walk(start_type, next_type, remaining - 1, [*steps, relation_id])

    starts = sorted(outgoing)
    if target is not None and target in starts:
        starts = [target, *[value for value in starts if value != target]]
    for length in lengths:
        for start_type in starts:
            walk(int(start_type), int(start_type), int(length), [])

    if target is not None:
        candidates.sort(key=lambda row: (int(row["start_type"]) != target or int(row["end_type"]) != target, row["name"]))
    else:
        candidates.sort(key=lambda row: row["name"])
    return candidates[: int(max_paths)]


def sample_typed_paths(
    graph: HeteroGraph,
    schema_paths: Sequence[Mapping[str, Any]],
    sample_seed: int,
    max_samples_per_schema: int = 2000,
    max_trials_per_schema: int = 20000,
    max_frontier_per_step: int = 128,
    return_status_rows: bool = False,
) -> list[dict[str, Any]]:
    """Sample original typed paths with bounded random frontier expansion."""

    rng = np.random.default_rng(int(sample_seed))
    outgoing = _outgoing_by_relation(graph)
    rows: list[dict[str, Any]] = []
    for schema_index, path in enumerate(schema_paths):
        steps = _relation_steps(path)
        name = _schema_name(path, steps)
        if not steps or not _schema_is_valid(graph, path, steps):
            if return_status_rows:
                rows.append(
                    {
                        "schema_path": name,
                        "relation_sequence": _sequence_text(steps),
                        "sample_status": "invalid_schema_path",
                        "sample_count": 0,
                    }
                )
            continue
        start_type = int(path.get("start_type", graph.relations[int(steps[0])].src_type))
        starts = nodes_of_type(graph, start_type)
        if len(starts) == 0:
            if return_status_rows:
                rows.append(
                    {
                        "schema_path": name,
                        "relation_sequence": _sequence_text(steps),
                        "sample_status": "no_start_nodes",
                        "sample_count": 0,
                    }
                )
            continue
        seen: set[tuple[int, ...]] = set()
        trials = 0
        while len(seen) < int(max_samples_per_schema) and trials < int(max_trials_per_schema):
            trials += 1
            current = int(starts[int(rng.integers(0, len(starts)))])
            node_path = [current]
            ok = True
            for relation_id in steps:
                neighbors = outgoing.get(int(relation_id), {}).get(current, [])
                if not neighbors:
                    ok = False
                    break
                if len(neighbors) > int(max_frontier_per_step):
                    pick = int(rng.integers(0, int(max_frontier_per_step)))
                    next_node = neighbors[pick][0]
                else:
                    next_node = neighbors[int(rng.integers(0, len(neighbors)))][0]
                node_path.append(int(next_node))
                current = int(next_node)
            if ok:
                seen.add(tuple(node_path))
        for sample_index, node_path in enumerate(sorted(seen)):
            rows.append(
                {
                    "sample_id": f"{name}:{schema_index}:{sample_index}",
                    "schema_path": name,
                    "schema_path_length": int(len(steps)),
                    "relation_sequence": _sequence_text(steps),
                    "node_path": _sequence_text(node_path),
                    "path_length": int(len(steps)),
                    "sample_seed": int(sample_seed),
                    "sample_status": "ok",
                }
            )
        if return_status_rows and not seen:
            rows.append(
                {
                    "schema_path": name,
                    "relation_sequence": _sequence_text(steps),
                    "sample_status": "zero_valid_samples",
                    "sample_count": 0,
                }
            )
    return rows


def _ordered_unique_after_consecutive(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or int(out[-1]) != int(value):
            out.append(int(value))
    return out


def _bounded_count(
    graph: HeteroGraph,
    start: int,
    end: int,
    steps: Sequence[int],
    *,
    max_frontier_per_step: int,
    max_count: int,
    outgoing: dict[int, dict[int, list[tuple[int, float]]]] | None = None,
) -> tuple[int, bool]:
    outgoing = outgoing if outgoing is not None else _outgoing_by_relation(graph)
    frontier: dict[int, int] = {int(start): 1}
    capped = False
    for relation_id in steps:
        next_frontier: dict[int, int] = defaultdict(int)
        total_next = 0
        for node, count in frontier.items():
            neighbors = outgoing.get(int(relation_id), {}).get(int(node), [])
            if len(neighbors) > int(max_frontier_per_step):
                neighbors = neighbors[: int(max_frontier_per_step)]
                capped = True
            for dst, _weight in neighbors:
                add = min(int(count), int(max_count) - total_next)
                if add <= 0:
                    capped = True
                    break
                next_frontier[int(dst)] += add
                total_next += add
                if total_next >= int(max_count):
                    capped = True
                    break
            if total_next >= int(max_count):
                break
        frontier = dict(sorted(next_frontier.items())[: int(max_frontier_per_step)])
        if len(next_frontier) > int(max_frontier_per_step):
            capped = True
        if not frontier:
            break
    return min(int(frontier.get(int(end), 0)), int(max_count)), capped


def evaluate_path_retention(
    samples: Sequence[Mapping[str, Any]],
    assignment: Assignment | np.ndarray,
    coarse_graph: HeteroGraph,
    original_graph: HeteroGraph | None = None,
    relation_mode: str = "typed_exact",
    max_count_frontier_per_step: int = 512,
    max_count_per_endpoint_schema: int = 4096,
) -> list[dict[str, Any]]:
    """Evaluate method-specific coarse survival for sampled typed paths."""

    mapping = assignment.assignment if isinstance(assignment, Assignment) else np.asarray(assignment, dtype=np.int64)
    typed_lookup = _edge_lookup_by_relation(coarse_graph)
    untyped = _untyped_lookup(coarse_graph)
    original_outgoing = _outgoing_by_relation(original_graph) if original_graph is not None else None
    coarse_outgoing = _outgoing_by_relation(coarse_graph) if original_graph is not None else None
    original_count_cache: dict[tuple[int, int, tuple[int, ...]], tuple[int, bool]] = {}
    coarse_count_cache: dict[tuple[int, int, tuple[int, ...]], tuple[int, bool]] = {}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if str(sample.get("sample_status", "ok")) != "ok":
            continue
        steps = _relation_steps(sample)
        nodes = _node_path(sample.get("node_path", []))
        if len(nodes) != len(steps) + 1:
            continue
        clusters = [int(mapping[int(node)]) for node in nodes]
        typed_survived: list[bool] = []
        untyped_survived: list[bool] = []
        log_weights: list[float] = []
        missing_weight_steps = 0
        for index, relation_id in enumerate(steps):
            edge = (clusters[index], clusters[index + 1])
            weight = typed_lookup.get(int(relation_id), {}).get(edge)
            typed_ok = weight is not None and float(weight) > 0.0
            untyped_ok = edge in untyped
            typed_survived.append(bool(typed_ok))
            untyped_survived.append(bool(untyped_ok))
            if typed_ok and weight is not None:
                log_weights.append(math.log1p(max(float(weight), 0.0)))
            else:
                missing_weight_steps += 1
        unique_clusters = len(set(clusters))
        collapsed = _ordered_unique_after_consecutive(clusters)
        original_count = ""
        coarse_count = ""
        path_count_ratio = ""
        log_error = ""
        count_capped = False
        if original_graph is not None:
            step_key = tuple(int(step) for step in steps)
            original_key = (int(nodes[0]), int(nodes[-1]), step_key)
            coarse_key = (int(clusters[0]), int(clusters[-1]), step_key)
            if original_key not in original_count_cache:
                original_count_cache[original_key] = _bounded_count(
                    original_graph,
                    nodes[0],
                    nodes[-1],
                    steps,
                    max_frontier_per_step=max_count_frontier_per_step,
                    max_count=max_count_per_endpoint_schema,
                    outgoing=original_outgoing,
                )
            if coarse_key not in coarse_count_cache:
                coarse_count_cache[coarse_key] = _bounded_count(
                    coarse_graph,
                    clusters[0],
                    clusters[-1],
                    steps,
                    max_frontier_per_step=max_count_frontier_per_step,
                    max_count=max_count_per_endpoint_schema,
                    outgoing=coarse_outgoing,
                )
            original_count, original_capped = original_count_cache[original_key]
            coarse_count, coarse_capped = coarse_count_cache[coarse_key]
            count_capped = bool(original_capped or coarse_capped)
            path_count_ratio = float(coarse_count / max(int(original_count), 1))
            log_error = float(abs(math.log1p(float(coarse_count)) - math.log1p(float(original_count))))
        typed_rate = float(sum(typed_survived) / max(len(steps), 1))
        untyped_rate = float(sum(untyped_survived) / max(len(steps), 1))
        rows.append(
            {
                **{key: sample.get(key, "") for key in ("dataset", "seed", "method")},
                "sample_id": sample.get("sample_id", ""),
                "schema_path": sample.get("schema_path", _schema_name(sample, steps)),
                "relation_sequence": _sequence_text(steps),
                "node_path": _sequence_text(nodes),
                "cluster_path": _sequence_text(clusters),
                "path_length": int(len(steps)),
                "relation_mode": str(relation_mode),
                "typed_exact_step_survival_rate": typed_rate,
                "all_steps_survived": bool(all(typed_survived)),
                "num_survived_steps": int(sum(typed_survived)),
                "untyped_step_survival_rate": untyped_rate,
                "untyped_all_steps_survived": bool(all(untyped_survived)),
                "schema_path_survival_gap": float(untyped_rate - typed_rate),
                "endpoint_pair_collapse_rate": float(clusters[0] == clusters[-1]),
                "any_consecutive_collapse_rate": float(any(clusters[i] == clusters[i + 1] for i in range(len(clusters) - 1))),
                "unique_cluster_ratio": float(unique_clusters / max(len(clusters), 1)),
                "path_cluster_length_after_collapse": int(len(collapsed)),
                "original_count_bounded": original_count,
                "coarse_count_bounded": coarse_count,
                "path_count_ratio": path_count_ratio,
                "log_path_count_error": log_error,
                "count_capped": bool(count_capped),
                "mean_log_step_weight": float(mean(log_weights)) if log_weights else float("nan"),
                "sum_log_step_weight": float(sum(log_weights)) if log_weights else float("nan"),
                "missing_weight_steps": int(missing_weight_steps),
                "path_weight_log_mean": float(mean(log_weights)) if log_weights else float("nan"),
                "path_weight_missing_step_rate": float(missing_weight_steps / max(len(steps), 1)),
                "weight_status": "available" if log_weights else "unavailable",
            }
        )
    return rows


def _numeric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | str:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in {None, ""}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return "" if not values else float(mean(values))


def summarize_metapath_retention(per_sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = ("dataset", "seed", "method", "schema_path", "path_length")
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "typed_exact_step_survival_rate",
        "untyped_step_survival_rate",
        "schema_path_survival_gap",
        "endpoint_pair_collapse_rate",
        "any_consecutive_collapse_rate",
        "unique_cluster_ratio",
        "path_cluster_length_after_collapse",
        "log_path_count_error",
        "path_weight_missing_step_rate",
    ]
    for key_values, rows in sorted(groups.items()):
        row = {key: key_values[index] for index, key in enumerate(keys)}
        row["schema_path_sample_count"] = int(len(rows))
        row["schema_path_length"] = int(float(row.get("path_length") or 0))
        for metric in metrics:
            row[f"{metric}_mean"] = _numeric_mean(rows, metric)
        row["schema_path_typed_survival_mean"] = row["typed_exact_step_survival_rate_mean"]
        row["schema_path_untyped_survival_mean"] = row["untyped_step_survival_rate_mean"]
        out.append(row)
    return out
