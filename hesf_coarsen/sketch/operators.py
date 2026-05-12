from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.relation_ops import apply_relation, apply_relation_transpose


def _relations_are_exact_reverses(graph: HeteroGraph, left_id: int, right_id: int) -> bool:
    left = graph.relations[int(left_id)]
    right = graph.relations[int(right_id)]
    if left.src_type != right.dst_type or left.dst_type != right.src_type:
        return False
    if left.num_edges != right.num_edges:
        return False
    if left.num_edges == 0:
        return True
    left_table = np.empty(
        left.num_edges,
        dtype=[("src", np.int64), ("dst", np.int64), ("weight", np.float32)],
    )
    right_table = np.empty(
        right.num_edges,
        dtype=[("src", np.int64), ("dst", np.int64), ("weight", np.float32)],
    )
    left_table["src"] = left.src
    left_table["dst"] = left.dst
    left_table["weight"] = left.weight
    right_table["src"] = right.dst
    right_table["dst"] = right.src
    right_table["weight"] = right.weight
    left_order = np.argsort(left_table, order=("src", "dst", "weight"))
    right_order = np.argsort(right_table, order=("src", "dst", "weight"))
    return bool(np.array_equal(left_table[left_order], right_table[right_order]))


def _detected_reverse_relation_ids(graph: HeteroGraph) -> set[int]:
    dropped: set[int] = set()
    relation_ids = sorted(graph.relations)
    for index, left_id in enumerate(relation_ids):
        if left_id in dropped:
            continue
        for right_id in relation_ids[index + 1 :]:
            if right_id in dropped:
                continue
            if _relations_are_exact_reverses(graph, left_id, right_id):
                dropped.add(int(right_id))
                break
    return dropped


def _effective_relation_weights(
    graph: HeteroGraph,
    relation_weights: dict[int, float],
    reverse_relation_policy: str,
    symmetric_relation_operator: bool,
) -> dict[int, float]:
    if reverse_relation_policy == "include_all":
        return dict(relation_weights)
    if reverse_relation_policy == "auto" and not symmetric_relation_operator:
        return dict(relation_weights)
    dropped = _detected_reverse_relation_ids(graph)
    kept = {
        relation_id: weight
        for relation_id, weight in relation_weights.items()
        if int(relation_id) not in dropped
    }
    total = float(sum(max(float(value), 0.0) for value in kept.values()))
    if total <= 0.0:
        uniform = 1.0 / max(len(kept), 1)
        return {int(relation_id): uniform for relation_id in kept}
    return {
        int(relation_id): max(float(weight), 0.0) / total
        for relation_id, weight in kept.items()
    }


def apply_relation_operator(
    graph: HeteroGraph,
    H: np.ndarray,
    relation_id: int,
    *,
    direction: str = "symmetric",
    weights_cache: dict[int, np.ndarray] | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Apply one normalized relation operator without materializing adjacency."""

    if backend != "numpy":
        raise ValueError(f"unsupported sketch operator backend: {backend}")
    H = np.asarray(H, dtype=np.float32)
    if H.shape[0] != graph.num_nodes:
        raise ValueError("H must have one row per graph node")

    direction = str(direction).lower()
    if direction == "forward":
        return apply_relation(graph, int(relation_id), H, normalize=True)
    if direction == "backward":
        return apply_relation_transpose(graph, int(relation_id), H, normalize=True)
    if direction == "symmetric":
        return apply_relation(graph, int(relation_id), H, normalize=True) + apply_relation_transpose(
            graph,
            int(relation_id),
            H,
            normalize=True,
        )
    raise ValueError(f"unsupported relation operator direction: {direction}")


def apply_fused_operator(
    graph: HeteroGraph,
    H: np.ndarray,
    relation_weights: dict[int, float] | None,
    *,
    symmetric_relation_operator: bool = True,
    reverse_relation_policy: str = "include_all",
    weights_cache: dict[int, np.ndarray] | None = None,
    backend: str = "numpy",
) -> np.ndarray:
    """Apply S_F H = sum_r alpha_r S_r H relation by relation."""

    H = np.asarray(H, dtype=np.float32)
    if H.shape[0] != graph.num_nodes:
        raise ValueError("H must have one row per graph node")
    if reverse_relation_policy not in {"auto", "include_all", "drop_detected_reverse_for_spectral_operator"}:
        raise ValueError(f"unsupported fusion.reverse_relation_policy: {reverse_relation_policy}")

    relation_ids = sorted(graph.relations)
    if relation_weights is None:
        uniform = 1.0 / max(len(relation_ids), 1)
        relation_weights = {relation_id: uniform for relation_id in relation_ids}
    relation_weights = _effective_relation_weights(
        graph,
        {int(relation_id): float(weight) for relation_id, weight in relation_weights.items()},
        reverse_relation_policy,
        symmetric_relation_operator,
    )

    out = np.zeros_like(H, dtype=np.float32)
    direction = "symmetric" if symmetric_relation_operator else "forward"
    for relation_id in relation_ids:
        weight = float(relation_weights.get(relation_id, 0.0))
        if weight == 0.0:
            continue
        out += np.float32(weight) * apply_relation_operator(
            graph,
            H,
            relation_id,
            direction=direction,
            weights_cache=weights_cache,
            backend=backend,
        )
    return out.astype(np.float32, copy=False)


def apply_fused_laplacian(
    graph: HeteroGraph,
    H: np.ndarray,
    relation_weights: dict[int, float] | None,
    *,
    symmetric_relation_operator: bool = True,
    reverse_relation_policy: str = "include_all",
    backend: str = "numpy",
) -> np.ndarray:
    """Apply L_F H = H - S_F H without building L_F."""

    H = np.asarray(H, dtype=np.float32)
    return H - apply_fused_operator(
        graph,
        H,
        relation_weights,
        symmetric_relation_operator=symmetric_relation_operator,
        reverse_relation_policy=reverse_relation_policy,
        backend=backend,
    )


def apply_relation_step(
    graph: HeteroGraph,
    H: np.ndarray,
    relation_id: int,
    direction: str,
) -> np.ndarray:
    """Apply one directional step in a chained meta-path sketch."""

    direction = str(direction).lower()
    if direction == "forward":
        return apply_relation_operator(graph, H, relation_id, direction="forward")
    if direction == "backward":
        return apply_relation_operator(graph, H, relation_id, direction="backward")
    raise ValueError(f"unsupported meta-path direction: {direction}")
