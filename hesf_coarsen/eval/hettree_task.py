from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from hesf_coarsen.eval.task_gnn import (
    TaskEvalResult,
    f1_scores,
    resolve_target_node_type,
    select_task_protocol_split,
    train_only_coarse_labels,
)
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


PathSpec = tuple[int, ...]


@dataclass(frozen=True)
class SemanticTreeFeatures:
    tensor: np.ndarray
    target_nodes: np.ndarray
    paths: list[PathSpec]
    feature_width: int
    type_ids: tuple[int, ...]


def infer_target_node_type(graph: HeteroGraph) -> int:
    labels = graph.labels
    if labels is None:
        raise ValueError("cannot infer a task target node type without graph labels")
    labels = np.asarray(labels).reshape(-1)
    best_type: int | None = None
    best_count = -1
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        nodes = nodes_of_type(graph, type_id)
        count = int(np.sum(labels[nodes] >= 0))
        if count > best_count:
            best_type = int(type_id)
            best_count = count
    if best_type is None or best_count <= 0:
        raise ValueError("cannot infer a task target node type without labeled nodes")
    return int(best_type)


def enumerate_target_paths(
    graph: HeteroGraph,
    *,
    target_type: int,
    max_hops: int = 2,
    max_paths: int | None = None,
) -> list[PathSpec]:
    paths: list[PathSpec] = [()]
    frontier: list[tuple[PathSpec, int]] = [((), int(target_type))]
    relation_specs = sorted(graph.relation_specs.items())
    for _depth in range(max(0, int(max_hops))):
        next_frontier: list[tuple[PathSpec, int]] = []
        for suffix, required_dst_type in frontier:
            for relation_id, spec in relation_specs:
                if int(spec.dst_type) != int(required_dst_type):
                    continue
                path = (int(relation_id), *suffix)
                paths.append(path)
                next_frontier.append((path, int(spec.src_type)))
        frontier = next_frontier
    deduped: list[PathSpec] = []
    seen: set[PathSpec] = set()
    for path in paths:
        if path in seen:
            continue
        deduped.append(path)
        seen.add(path)
        if max_paths is not None and len(deduped) >= int(max_paths):
            break
    return deduped


def _feature_width(graphs: Iterable[HeteroGraph]) -> int:
    width = 0
    for graph in graphs:
        for feature in (graph.features or {}).values():
            width = max(width, int(feature.shape[1]))
    return max(width, 1)


def _type_ids(graphs: Iterable[HeteroGraph]) -> tuple[int, ...]:
    ids: set[int] = set()
    for graph in graphs:
        ids.update(int(value) for value in np.unique(graph.node_type))
    return tuple(sorted(ids))


def _base_feature_matrix(
    graph: HeteroGraph,
    *,
    feature_width: int,
    type_ids: tuple[int, ...],
) -> np.ndarray:
    type_offset = int(feature_width)
    matrix = np.zeros((int(graph.num_nodes), int(feature_width) + len(type_ids)), dtype=np.float32)
    for type_id in sorted(int(value) for value in np.unique(graph.node_type)):
        nodes = nodes_of_type(graph, type_id)
        feature = (graph.features or {}).get(type_id)
        if feature is not None and len(nodes):
            local_width = min(int(feature.shape[1]), int(feature_width))
            matrix[nodes, :local_width] = np.asarray(feature[:, :local_width], dtype=np.float32)
        if type_id in type_ids:
            matrix[nodes, type_offset + type_ids.index(type_id)] = 1.0
    return matrix


def _apply_relation_step(graph: HeteroGraph, x: np.ndarray, relation_id: int) -> np.ndarray:
    rel = graph.relations[int(relation_id)]
    if rel.num_edges == 0:
        return np.zeros_like(x)
    src = np.asarray(rel.src, dtype=np.int64)
    dst = np.asarray(rel.dst, dtype=np.int64)
    weight = np.asarray(rel.weight, dtype=np.float32)
    degree = np.bincount(dst, weights=weight, minlength=int(graph.num_nodes)).astype(np.float32)
    norm = weight / np.maximum(degree[dst], np.float32(1.0e-12))
    try:
        from scipy import sparse  # type: ignore

        operator = sparse.csr_matrix(
            (norm, (dst, src)),
            shape=(int(graph.num_nodes), int(graph.num_nodes)),
            dtype=np.float32,
        )
        return np.asarray(operator @ x, dtype=np.float32)
    except Exception:
        out = np.zeros_like(x)
        chunk_edges = max(1, min(4096, len(src)))
        for start in range(0, len(src), chunk_edges):
            stop = min(start + chunk_edges, len(src))
            np.add.at(out, dst[start:stop], x[src[start:stop]] * norm[start:stop, None])
        return out


def build_semantic_tree_features(
    graph: HeteroGraph,
    *,
    target_type: int,
    paths: list[PathSpec],
    feature_width: int | None = None,
    type_ids: tuple[int, ...] | None = None,
) -> SemanticTreeFeatures:
    width = _feature_width([graph]) if feature_width is None else int(feature_width)
    ids = _type_ids([graph]) if type_ids is None else tuple(int(value) for value in type_ids)
    base = _base_feature_matrix(graph, feature_width=width, type_ids=ids)
    target_nodes = nodes_of_type(graph, int(target_type))
    blocks: list[np.ndarray] = []
    for path in paths:
        propagated = base
        for relation_id in path:
            propagated = _apply_relation_step(graph, propagated, int(relation_id))
        blocks.append(propagated[target_nodes].astype(np.float32, copy=False))
    if blocks:
        tensor = np.stack(blocks, axis=1).astype(np.float32, copy=False)
    else:
        tensor = np.zeros((len(target_nodes), 0, width + len(ids)), dtype=np.float32)
    return SemanticTreeFeatures(
        tensor=tensor,
        target_nodes=target_nodes.astype(np.int64, copy=False),
        paths=list(paths),
        feature_width=int(width),
        type_ids=ids,
    )


def _classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    macro_empty_class_policy: str,
) -> dict[str, float]:
    scores = f1_scores(y_true, y_pred, macro_empty_class_policy=macro_empty_class_policy)
    valid = (np.asarray(y_true).reshape(-1) >= 0) & (np.asarray(y_pred).reshape(-1) >= 0)
    if not np.any(valid):
        accuracy = 0.0
    else:
        truth = np.asarray(y_true).reshape(-1)[valid]
        pred = np.asarray(y_pred).reshape(-1)[valid]
        accuracy = float(np.mean(truth == pred))
    return {**scores, "accuracy": accuracy}


def _local_indices(global_nodes: np.ndarray, target_nodes: np.ndarray) -> np.ndarray:
    lookup = {int(node): index for index, node in enumerate(np.asarray(target_nodes).tolist())}
    return np.asarray(
        [lookup[int(node)] for node in np.asarray(global_nodes).reshape(-1) if int(node) in lookup],
        dtype=np.int64,
    )


def evaluate_hettree_task(
    original: HeteroGraph,
    coarse: HeteroGraph,
    original_to_coarse: np.ndarray,
    *,
    seed: int = 12345,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 1.0e-4,
    dropout: float = 0.25,
    max_hops: int = 2,
    max_paths: int | None = 32,
    device: str = "auto",
    target_node_type: str | int | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    macro_empty_class_policy: str = "truth_pred_union",
    official_split_nodes: Mapping[str, np.ndarray] | None = None,
) -> TaskEvalResult:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - only exercised without torch installed.
        return TaskEvalResult(
            {
                "model": "hettree_lite",
                "skipped": True,
                "skip_reason": f"torch_unavailable: {exc}",
            }
        )

    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
    if target_node_type is None or str(target_node_type) == "":
        target_type = infer_target_node_type(original)
    else:
        resolved = resolve_target_node_type(original, target_node_type)
        if resolved is None:
            target_type = infer_target_node_type(original)
        else:
            target_type = int(resolved)
    train_nodes, val_nodes, test_nodes, task_protocol = select_task_protocol_split(
        original,
        labels,
        seed=int(seed),
        target_node_type=int(target_type),
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        official_split_nodes=official_split_nodes,
    )
    if len(train_nodes) == 0 or len(test_nodes) == 0:
        return TaskEvalResult(
            {
                "model": "hettree_lite",
                "skipped": True,
                "skip_reason": "not_enough_labeled_nodes",
                **task_protocol,
            }
        )

    coarse_labels, label_protocol = train_only_coarse_labels(
        labels,
        original_to_coarse,
        train_nodes,
        num_coarse_nodes=coarse.num_nodes,
        test_nodes=test_nodes,
    )
    coarse_train_nodes = np.unique(original_to_coarse[train_nodes]).astype(np.int64, copy=False)
    coarse_train_nodes = coarse_train_nodes[coarse_labels[coarse_train_nodes] >= 0]
    if len(coarse_train_nodes) == 0:
        return TaskEvalResult(
            {
                "model": "hettree_lite",
                "skipped": True,
                "skip_reason": "no_labeled_coarse_train_nodes",
                **task_protocol,
                **label_protocol,
            }
        )

    if device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = str(device)
    dev = torch.device(device_name)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats()

    paths = enumerate_target_paths(
        original,
        target_type=int(target_type),
        max_hops=int(max_hops),
        max_paths=max_paths,
    )
    width = _feature_width([original, coarse])
    ids = _type_ids([original, coarse])
    feature_start = perf_counter()
    coarse_tree = build_semantic_tree_features(
        coarse,
        target_type=int(target_type),
        paths=paths,
        feature_width=width,
        type_ids=ids,
    )
    original_tree = build_semantic_tree_features(
        original,
        target_type=int(target_type),
        paths=paths,
        feature_width=width,
        type_ids=ids,
    )
    feature_time = float(perf_counter() - feature_start)

    coarse_train_local = _local_indices(coarse_train_nodes, coarse_tree.target_nodes)
    if len(coarse_train_local) == 0:
        return TaskEvalResult(
            {
                "model": "hettree_lite",
                "skipped": True,
                "skip_reason": "no_labeled_target_type_coarse_train_nodes",
                **task_protocol,
                **label_protocol,
            }
        )
    original_test_local = _local_indices(test_nodes, original_tree.target_nodes)
    if len(original_test_local) == 0:
        return TaskEvalResult(
            {
                "model": "hettree_lite",
                "skipped": True,
                "skip_reason": "no_target_type_test_nodes",
                **task_protocol,
                **label_protocol,
            }
        )

    num_classes = int(labels[labels >= 0].max(initial=0)) + 1

    class HetTreeLite(nn.Module):
        def __init__(self, input_dim: int, num_paths: int):
            super().__init__()
            self.path_linear = nn.Linear(int(input_dim), int(hidden_dim))
            self.attention = nn.Linear(int(hidden_dim), 1, bias=False)
            self.classifier = nn.Linear(int(hidden_dim), int(num_classes))
            self.dropout = nn.Dropout(float(dropout))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.path_linear.weight)
            nn.init.zeros_(self.path_linear.bias)
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

        def forward(self, x: Any) -> Any:
            h = torch.relu(self.path_linear(x))
            score = self.attention(torch.tanh(h)).squeeze(-1)
            alpha = torch.softmax(score, dim=1)
            pooled = torch.sum(h * alpha.unsqueeze(-1), dim=1)
            return self.classifier(self.dropout(pooled))

    coarse_x = torch.as_tensor(coarse_tree.tensor, dtype=torch.float32, device=dev)
    original_x = torch.as_tensor(original_tree.tensor, dtype=torch.float32, device=dev)
    coarse_y = torch.as_tensor(
        coarse_labels[coarse_tree.target_nodes],
        dtype=torch.long,
        device=dev,
    )
    train_idx = torch.as_tensor(coarse_train_local, dtype=torch.long, device=dev)
    model = HetTreeLite(input_dim=int(coarse_tree.tensor.shape[2]), num_paths=len(paths)).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()
    train_start = perf_counter()
    model.train()
    for _epoch in range(max(0, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits = model(coarse_x)
        loss = loss_fn(logits[train_idx], coarse_y[train_idx])
        loss.backward()
        optimizer.step()
    train_time = float(perf_counter() - train_start)

    model.eval()
    with torch.no_grad():
        coarse_pred_local = model(coarse_x).argmax(dim=1).detach().cpu().numpy()
        original_pred_local = model(original_x).argmax(dim=1).detach().cpu().numpy()

    original_test_targets = original_tree.target_nodes[original_test_local]
    transfer_pred = original_pred_local[original_test_local]
    transfer_scores = _classification_scores(
        labels[original_test_targets],
        transfer_pred,
        macro_empty_class_policy=macro_empty_class_policy,
    )

    coarse_pred_full = np.full(coarse.num_nodes, -1, dtype=np.int64)
    coarse_pred_full[coarse_tree.target_nodes] = coarse_pred_local
    projected_pred = coarse_pred_full[np.asarray(original_to_coarse, dtype=np.int64)[test_nodes]]
    projected_scores = _classification_scores(
        labels[test_nodes],
        projected_pred,
        macro_empty_class_policy=macro_empty_class_policy,
    )

    coarse_train_scores = _classification_scores(
        coarse_labels[coarse_tree.target_nodes[coarse_train_local]],
        coarse_pred_local[coarse_train_local],
        macro_empty_class_policy=macro_empty_class_policy,
    )
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    return TaskEvalResult(
        {
            "model": "hettree_lite",
            "hettree_reference": "AAAI25 HETTREE-inspired semantic-tree evaluator; official repository unavailable locally",
            "skipped": False,
            "device": device_name,
            "train_on": "coarse_train_labels_only",
            "eval_on": "original_test_transfer",
            "projection_eval_on": "original_test_projected",
            "target_node_type_id": int(target_type),
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "num_classes": int(num_classes),
            "path_count": int(len(paths)),
            "paths": ["self" if not path else "->".join(str(relation_id) for relation_id in path) for path in paths],
            "max_hops": int(max_hops),
            "input_dim": int(coarse_tree.tensor.shape[2]),
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "train_labeled_nodes": int(len(train_nodes)),
            "val_labeled_nodes": int(len(val_nodes)),
            "test_labeled_nodes": int(len(test_nodes)),
            "coarse_train_nodes": int(len(coarse_train_local)),
            **task_protocol,
            **label_protocol,
            "coarse_train_micro_f1": coarse_train_scores["micro_f1"],
            "coarse_train_macro_f1": coarse_train_scores["macro_f1"],
            "coarse_train_accuracy": coarse_train_scores["accuracy"],
            "projected_original_micro_f1": projected_scores["micro_f1"],
            "projected_original_macro_f1": projected_scores["macro_f1"],
            "projected_original_accuracy": projected_scores["accuracy"],
            "transfer_original_micro_f1": transfer_scores["micro_f1"],
            "transfer_original_macro_f1": transfer_scores["macro_f1"],
            "transfer_original_accuracy": transfer_scores["accuracy"],
            "micro_f1": transfer_scores["micro_f1"],
            "macro_f1": transfer_scores["macro_f1"],
            "accuracy": transfer_scores["accuracy"],
            "primary_task_metric_name": "transfer_original_macro_f1",
            "primary_task_metric": transfer_scores["macro_f1"],
            "feature_time": feature_time,
            "train_time": train_time,
            "total_time": float(feature_time + train_time),
            "peak_vram_allocated_mb": peak_vram_mb,
        }
    )
