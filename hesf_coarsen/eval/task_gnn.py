from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


@dataclass(frozen=True)
class TaskEvalResult:
    metrics: dict[str, Any]


def compose_assignments(original_nodes: int, assignment_paths: list[str]) -> np.ndarray:
    mapping = np.arange(int(original_nodes), dtype=np.int64)
    for path in assignment_paths:
        payload = np.load(path)
        local = payload["assignment"].astype(np.int64, copy=False)
        mapping = local[mapping]
    return mapping.astype(np.int64, copy=False)


def f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    macro_empty_class_policy: str = "truth_pred_union",
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    valid = (y_true >= 0) & (y_pred >= 0)
    if not np.any(valid):
        return {"micro_f1": 0.0, "macro_f1": 0.0}
    truth = y_true[valid].astype(np.int64, copy=False)
    pred = y_pred[valid].astype(np.int64, copy=False)
    policy = str(macro_empty_class_policy)
    if policy == "eval_present":
        labels = np.unique(truth)
    elif policy == "truth_pred_union":
        labels = np.union1d(truth, pred)
    else:
        raise ValueError(f"unsupported macro_empty_class_policy: {macro_empty_class_policy}")
    per_label: list[float] = []
    for label in labels:
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        denom = 2 * tp + fp + fn
        per_label.append(0.0 if denom == 0 else float(2 * tp / denom))
    return {
        "micro_f1": float(np.mean(truth == pred)),
        "macro_f1": float(np.mean(per_label) if per_label else 0.0),
    }


def labeled_split(labels: np.ndarray, seed: int, train_fraction: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    labeled = np.flatnonzero(np.asarray(labels).reshape(-1) >= 0).astype(np.int64)
    if len(labeled) == 0:
        return labeled, labeled
    rng = np.random.default_rng(int(seed))
    perm = labeled.copy()
    rng.shuffle(perm)
    train_count = max(1, int(round(len(perm) * float(train_fraction))))
    if train_count >= len(perm) and len(perm) > 1:
        train_count = len(perm) - 1
    return perm[:train_count], perm[train_count:]


def resolve_target_node_type(graph: HeteroGraph, target_node_type: str | int | None) -> int | None:
    if target_node_type is None or str(target_node_type) == "":
        return None
    if isinstance(target_node_type, (int, np.integer)):
        type_id = int(target_node_type)
        if np.any(graph.node_type == type_id):
            return type_id
        raise ValueError(f"target_node_type {type_id} is not present in graph")
    text = str(target_node_type)
    try:
        type_id = int(text)
    except ValueError:
        type_id = -1
    if type_id >= 0:
        if np.any(graph.node_type == type_id):
            return type_id
        raise ValueError(f"target_node_type {type_id} is not present in graph")

    name_to_type: dict[str, int] = {}
    for spec in graph.relation_specs.values():
        parts = str(spec.name).split("__")
        if len(parts) >= 3:
            name_to_type.setdefault(parts[0], int(spec.src_type))
            name_to_type.setdefault(parts[-1], int(spec.dst_type))
    if text in name_to_type:
        return int(name_to_type[text])
    raise ValueError(f"target_node_type {target_node_type!r} could not be resolved from relation schema")


def _stratified_three_way_split(
    labeled_nodes: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    labeled_nodes = np.asarray(labeled_nodes, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    for label in np.unique(labels[labeled_nodes]):
        group = labeled_nodes[labels[labeled_nodes] == label].copy()
        rng.shuffle(group)
        n = int(len(group))
        if n == 0:
            continue
        if n == 1:
            train_count, val_count = 1, 0
        else:
            train_count = max(1, int(np.floor(n * float(train_fraction))))
            train_count = min(train_count, n - 1)
            remaining = n - train_count
            if remaining >= 2 and float(val_fraction) > 0.0:
                val_count = max(1, int(np.floor(n * float(val_fraction))))
                val_count = min(val_count, remaining - 1)
            else:
                val_count = 0
        train_parts.append(group[:train_count])
        if val_count:
            val_parts.append(group[train_count : train_count + val_count])
        if train_count + val_count < n:
            test_parts.append(group[train_count + val_count :])
    train = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    test = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)
    for split in (train, val, test):
        rng.shuffle(split)
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


def select_task_protocol_split(
    graph: HeteroGraph,
    labels: np.ndarray,
    *,
    seed: int,
    target_node_type: str | int | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    official_split_nodes: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels).reshape(-1)
    type_id = resolve_target_node_type(graph, target_node_type)
    if type_id is None:
        candidate_nodes = np.arange(graph.num_nodes, dtype=np.int64)
        target_label = "all_labeled"
    else:
        candidate_nodes = nodes_of_type(graph, int(type_id))
        target_label = str(target_node_type)
    labeled_nodes = candidate_nodes[labels[candidate_nodes] >= 0].astype(np.int64, copy=False)
    split_policy = "synthetic_stratified"
    if official_split_nodes is None:
        train_nodes, val_nodes, test_nodes = _stratified_three_way_split(
            labeled_nodes,
            labels,
            seed=int(seed),
            train_fraction=float(train_fraction),
            val_fraction=float(val_fraction),
        )
    else:
        candidate_set = set(int(node) for node in candidate_nodes.tolist())

        def _official_nodes(*names: str) -> np.ndarray:
            for name in names:
                if name in official_split_nodes:
                    raw = np.asarray(official_split_nodes[name], dtype=np.int64).reshape(-1)
                    kept = [
                        int(node)
                        for node in raw.tolist()
                        if int(node) in candidate_set and labels[int(node)] >= 0
                    ]
                    return np.asarray(kept, dtype=np.int64)
            return np.asarray([], dtype=np.int64)

        train_nodes = _official_nodes("train")
        val_nodes = _official_nodes("valid", "val")
        test_nodes = _official_nodes("test")
        split_policy = "official"

    def _coverage(nodes: np.ndarray) -> float:
        if len(nodes) == 0:
            return 0.0
        return float(np.mean(labels[nodes] >= 0))

    def _class_count(nodes: np.ndarray) -> int:
        if len(nodes) == 0:
            return 0
        return int(len(np.unique(labels[nodes][labels[nodes] >= 0])))

    suffix = "target_type" if type_id is not None else "all_labeled"
    diagnostics: dict[str, Any] = {
        "target_node_type": target_label,
        "target_node_type_id": "" if type_id is None else int(type_id),
        "task_split_policy": split_policy,
        "official_split_consistency": (
            f"official_{suffix}" if official_split_nodes is not None else f"synthetic_stratified_{suffix}"
        ),
        "macro_f1_empty_class_policy": "truth_pred_union",
        "coarse_train_label_source": "train_only",
        "num_labeled_nodes_total": int(len(labeled_nodes)),
        "num_labeled_nodes_train": int(len(train_nodes)),
        "num_labeled_nodes_val": int(len(val_nodes)),
        "num_labeled_nodes_test": int(len(test_nodes)),
        "label_coverage_train": _coverage(train_nodes),
        "label_coverage_val": _coverage(val_nodes),
        "label_coverage_test": _coverage(test_nodes),
        "num_classes_present_train": _class_count(train_nodes),
        "num_classes_present_val": _class_count(val_nodes),
        "num_classes_present_test": _class_count(test_nodes),
    }
    return train_nodes, val_nodes, test_nodes, diagnostics


def train_only_coarse_labels(
    labels: np.ndarray,
    original_to_coarse: np.ndarray,
    train_nodes: np.ndarray,
    *,
    num_coarse_nodes: int,
    test_nodes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels).reshape(-1)
    original_to_coarse = np.asarray(original_to_coarse, dtype=np.int64).reshape(-1)
    train_nodes = np.asarray(train_nodes, dtype=np.int64).reshape(-1)
    coarse_labels = np.full(int(num_coarse_nodes), -1, dtype=np.int64)
    entropies: list[float] = []
    for coarse_node in range(int(num_coarse_nodes)):
        members = train_nodes[original_to_coarse[train_nodes] == coarse_node]
        member_labels = labels[members]
        member_labels = member_labels[member_labels >= 0].astype(np.int64, copy=False)
        if len(member_labels) == 0:
            continue
        values, counts = np.unique(member_labels, return_counts=True)
        order = np.lexsort((values, -counts))
        coarse_labels[coarse_node] = int(values[order[0]])
        probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
        entropies.append(float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-12)))))
    test_leakage_check = "not_applicable"
    if test_nodes is not None:
        test_nodes = np.asarray(test_nodes, dtype=np.int64).reshape(-1)
        train_set = set(int(node) for node in train_nodes.tolist())
        test_set = set(int(node) for node in test_nodes.tolist())
        test_leakage_check = "passed" if train_set.isdisjoint(test_set) else "failed"
    diagnostics = {
        "train_only_label_coverage": float(np.mean(coarse_labels >= 0)) if num_coarse_nodes else 0.0,
        "cluster_train_label_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "test_label_leakage_check": test_leakage_check,
    }
    return coarse_labels, diagnostics


def refine_curve_summary(checkpoint_metrics: dict[int, dict[str, float]]) -> dict[str, Any]:
    if not checkpoint_metrics:
        return {
            "best_refined_macro_f1": 0.0,
            "best_refined_epoch": 0,
            "refine_auc_macro_f1": 0.0,
            "refine_time_by_epoch": {},
        }
    ordered = sorted((int(epoch), values) for epoch, values in checkpoint_metrics.items())
    best_epoch, best_values = max(
        ordered,
        key=lambda item: (float(item[1].get("macro_f1", 0.0)), -int(item[0])),
    )
    epochs = np.asarray([epoch for epoch, _values in ordered], dtype=np.float64)
    macro = np.asarray(
        [float(values.get("macro_f1", 0.0)) for _epoch, values in ordered],
        dtype=np.float64,
    )
    if len(epochs) == 1 or float(epochs[-1] - epochs[0]) <= 0.0:
        auc = float(macro[-1])
    else:
        auc = float(np.trapezoid(macro, epochs) / max(float(epochs[-1] - epochs[0]), 1.0e-12))
    return {
        "best_refined_macro_f1": float(best_values.get("macro_f1", 0.0)),
        "best_refined_epoch": int(best_epoch),
        "refine_auc_macro_f1": auc,
        "refine_time_by_epoch": {
            str(int(epoch)): float(values.get("refine_time", 0.0))
            for epoch, values in ordered
        },
    }


def evaluate_rgcn_task(
    original: HeteroGraph,
    coarse: HeteroGraph,
    original_to_coarse: np.ndarray,
    *,
    seed: int = 12345,
    hidden_dim: int = 32,
    epochs: int = 20,
    refine_epochs: int = 10,
    refine_epochs_list: Iterable[int] | None = None,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device: str = "auto",
    full_graph_rgcn_lite: bool = False,
    full_graph_baselines: Iterable[str] | None = None,
    full_graph_tuned_epochs: int | None = None,
    coarse_model: str = "rgcn_lite",
    target_node_type: str | int | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    macro_empty_class_policy: str = "truth_pred_union",
    official_split_nodes: Mapping[str, np.ndarray] | None = None,
) -> TaskEvalResult:
    coarse_model_name = str(coarse_model).lower().replace("-", "_")
    if coarse_model_name in {"hgt_small", "full_graph_hgt_small"}:
        coarse_model_name = "hgt_lite"
    if coarse_model_name in {"han", "full_graph_han_small"}:
        coarse_model_name = "han_small"
    if coarse_model_name in {"rgcn", "full_graph_rgcn_lite_default"}:
        coarse_model_name = "rgcn_lite"
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - exercised only without torch installed.
        return TaskEvalResult(
            {
                "model": coarse_model_name,
                "coarse_model": coarse_model_name,
                "skipped": True,
                "skip_reason": f"torch_unavailable: {exc}",
            }
        )

    torch.manual_seed(int(seed))
    if device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = str(device)
    dev = torch.device(device_name)

    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
    train_nodes, val_nodes, test_nodes, task_protocol = select_task_protocol_split(
        original,
        labels,
        seed=int(seed),
        target_node_type=target_node_type,
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        official_split_nodes=official_split_nodes,
    )
    task_protocol["macro_f1_empty_class_policy"] = str(macro_empty_class_policy)
    if len(train_nodes) == 0 or len(test_nodes) == 0:
        return TaskEvalResult(
            {
                "model": coarse_model_name,
                "coarse_model": coarse_model_name,
                "skipped": True,
                "skip_reason": "not_enough_labeled_nodes",
                **task_protocol,
            }
        )
    num_classes = int(labels[labels >= 0].max(initial=0)) + 1
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
                "model": "rgcn_lite",
                "skipped": True,
                "skip_reason": "no_labeled_coarse_train_nodes",
            }
        )

    class TypeEncoder(nn.Module):
        def __init__(self, graph: HeteroGraph, model_hidden_dim: int):
            super().__init__()
            self.hidden_dim = int(model_hidden_dim)
            type_ids = sorted(int(type_id) for type_id in np.unique(graph.node_type))
            self.type_linears = nn.ModuleDict()
            self.type_embeddings = nn.ParameterDict()
            for type_id in type_ids:
                feature = (graph.features or {}).get(type_id)
                if feature is None:
                    self.type_embeddings[str(type_id)] = nn.Parameter(torch.zeros(self.hidden_dim))
                else:
                    self.type_linears[str(type_id)] = nn.Linear(int(feature.shape[1]), self.hidden_dim)

        def reset_parameters(self) -> None:
            for module in self.type_linears.values():
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            for parameter in self.type_embeddings.values():
                nn.init.normal_(parameter, std=0.02)

        def forward(self, data: dict[str, Any]) -> Any:
            h = torch.zeros((data["num_nodes"], self.hidden_dim), device=dev)
            for type_id, node_tensor in data["type_nodes"].items():
                key = str(type_id)
                if key in self.type_linears:
                    h[node_tensor] = self.type_linears[key](data["features"][type_id])
                else:
                    h[node_tensor] = self.type_embeddings[key]
            return h

    class RGCNLite(nn.Module):
        def __init__(
            self,
            graph: HeteroGraph,
            *,
            model_hidden_dim: int | None = None,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.hidden_dim = int(model_hidden_dim if model_hidden_dim is not None else hidden_dim)
            self.encoder = TypeEncoder(graph, self.hidden_dim)
            rel_ids = sorted(int(relation_id) for relation_id in graph.relations)
            self.rel_ids = rel_ids
            self.rel_weights = nn.ParameterDict(
                {
                    str(relation_id): nn.Parameter(
                        torch.empty(self.hidden_dim, self.hidden_dim)
                    )
                    for relation_id in rel_ids
                }
            )
            self.self_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.out = nn.Linear(self.hidden_dim, int(num_classes))
            self.dropout = nn.Dropout(float(dropout))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            self.encoder.reset_parameters()
            for parameter in self.rel_weights.values():
                nn.init.xavier_uniform_(parameter)
            nn.init.xavier_uniform_(self.self_linear.weight)
            nn.init.zeros_(self.self_linear.bias)
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def encode(self, data: dict[str, Any]) -> Any:
            return torch.relu(self.encoder(data))

        def forward(self, data: dict[str, Any]) -> Any:
            h = self.dropout(self.encode(data))
            out = self.self_linear(h)
            degree = torch.ones((data["num_nodes"], 1), device=dev)
            for relation_id, edge in data["edges"].items():
                src, dst, weight = edge
                if src.numel() == 0:
                    continue
                msg = h[src] @ self.rel_weights[str(relation_id)]
                msg = msg * weight[:, None]
                out.index_add_(0, dst, msg)
                degree.index_add_(0, dst, weight[:, None])
            h = torch.relu(out / degree.clamp_min(1.0e-6))
            return self.out(self.dropout(h))

    class HANSmall(nn.Module):
        def __init__(
            self,
            graph: HeteroGraph,
            *,
            model_hidden_dim: int | None = None,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.hidden_dim = int(model_hidden_dim if model_hidden_dim is not None else hidden_dim)
            self.encoder = TypeEncoder(graph, self.hidden_dim)
            self.rel_ids = sorted(int(relation_id) for relation_id in graph.relations)
            self.rel_weights = nn.ParameterDict(
                {
                    str(relation_id): nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
                    for relation_id in self.rel_ids
                }
            )
            self.self_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.semantic_attention = nn.Parameter(torch.empty(self.hidden_dim))
            self.out = nn.Linear(self.hidden_dim, int(num_classes))
            self.dropout = nn.Dropout(float(dropout))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            self.encoder.reset_parameters()
            for parameter in self.rel_weights.values():
                nn.init.xavier_uniform_(parameter)
            nn.init.xavier_uniform_(self.self_linear.weight)
            nn.init.zeros_(self.self_linear.bias)
            nn.init.normal_(self.semantic_attention, std=0.02)
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, data: dict[str, Any]) -> Any:
            h = self.dropout(torch.relu(self.encoder(data)))
            relation_outputs = []
            for relation_id, edge in data["edges"].items():
                src, dst, weight = edge
                if src.numel() == 0:
                    continue
                rel_out = torch.zeros((data["num_nodes"], self.hidden_dim), device=dev)
                degree = torch.zeros((data["num_nodes"], 1), device=dev)
                msg = h[src] @ self.rel_weights[str(relation_id)]
                msg = msg * weight[:, None]
                rel_out.index_add_(0, dst, msg)
                degree.index_add_(0, dst, weight[:, None])
                relation_outputs.append(rel_out / degree.clamp_min(1.0e-6))
            if relation_outputs:
                stacked = torch.stack(relation_outputs, dim=1)
                logits = torch.tanh(stacked) @ self.semantic_attention
                attention = torch.softmax(logits, dim=1)
                rel_h = torch.sum(stacked * attention[:, :, None], dim=1)
                h = torch.relu(self.self_linear(h) + rel_h)
            else:
                h = torch.relu(self.self_linear(h))
            return self.out(self.dropout(h))

    class HGTLite(nn.Module):
        def __init__(
            self,
            graph: HeteroGraph,
            *,
            model_hidden_dim: int | None = None,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.hidden_dim = int(model_hidden_dim if model_hidden_dim is not None else hidden_dim)
            self.encoder = TypeEncoder(graph, self.hidden_dim)
            type_ids = sorted(int(type_id) for type_id in np.unique(graph.node_type))
            self.q_linears = nn.ModuleDict({str(t): nn.Linear(self.hidden_dim, self.hidden_dim) for t in type_ids})
            self.k_linears = nn.ModuleDict({str(t): nn.Linear(self.hidden_dim, self.hidden_dim) for t in type_ids})
            self.v_linears = nn.ModuleDict({str(t): nn.Linear(self.hidden_dim, self.hidden_dim) for t in type_ids})
            self.rel_ids = sorted(int(relation_id) for relation_id in graph.relations)
            self.rel_weights = nn.ParameterDict(
                {
                    str(relation_id): nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
                    for relation_id in self.rel_ids
                }
            )
            self.rel_priors = nn.ParameterDict(
                {str(relation_id): nn.Parameter(torch.zeros(1)) for relation_id in self.rel_ids}
            )
            self.self_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.out = nn.Linear(self.hidden_dim, int(num_classes))
            self.dropout = nn.Dropout(float(dropout))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            self.encoder.reset_parameters()
            for modules in (self.q_linears, self.k_linears, self.v_linears):
                for module in modules.values():
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            for parameter in self.rel_weights.values():
                nn.init.xavier_uniform_(parameter)
            nn.init.xavier_uniform_(self.self_linear.weight)
            nn.init.zeros_(self.self_linear.bias)
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def _apply_type_linears(self, modules: Any, h: Any, data: dict[str, Any]) -> Any:
            out = torch.zeros_like(h)
            for type_id, node_tensor in data["type_nodes"].items():
                out[node_tensor] = modules[str(type_id)](h[node_tensor])
            return out

        def forward(self, data: dict[str, Any]) -> Any:
            h = self.dropout(torch.relu(self.encoder(data)))
            q = self._apply_type_linears(self.q_linears, h, data)
            k = self._apply_type_linears(self.k_linears, h, data)
            v = self._apply_type_linears(self.v_linears, h, data)
            out = self.self_linear(h)
            degree = torch.ones((data["num_nodes"], 1), device=dev)
            scale = float(self.hidden_dim) ** 0.5
            for relation_id, edge in data["edges"].items():
                src, dst, weight = edge
                if src.numel() == 0:
                    continue
                attn = torch.sigmoid(
                    torch.sum(q[dst] * k[src], dim=1) / scale + self.rel_priors[str(relation_id)]
                )
                rel_weight = weight * attn
                msg = (v[src] @ self.rel_weights[str(relation_id)]) * rel_weight[:, None]
                out.index_add_(0, dst, msg)
                degree.index_add_(0, dst, rel_weight[:, None])
            h = torch.relu(out / degree.clamp_min(1.0e-6))
            return self.out(self.dropout(h))

    def graph_data(graph: HeteroGraph) -> dict[str, Any]:
        type_nodes = {
            int(type_id): torch.as_tensor(nodes_of_type(graph, int(type_id)), dtype=torch.long, device=dev)
            for type_id in sorted(np.unique(graph.node_type))
        }
        features: dict[int, Any] = {}
        for type_id, feature in (graph.features or {}).items():
            features[int(type_id)] = torch.as_tensor(feature, dtype=torch.float32, device=dev)
        edges = {}
        for relation_id, rel in graph.relations.items():
            edges[int(relation_id)] = (
                torch.as_tensor(rel.src, dtype=torch.long, device=dev),
                torch.as_tensor(rel.dst, dtype=torch.long, device=dev),
                torch.as_tensor(rel.weight, dtype=torch.float32, device=dev),
            )
        return {"num_nodes": int(graph.num_nodes), "type_nodes": type_nodes, "features": features, "edges": edges}

    coarse_data = graph_data(coarse)
    original_data = graph_data(original)
    coarse_y = torch.as_tensor(coarse_labels, dtype=torch.long, device=dev)
    original_y = torch.as_tensor(labels, dtype=torch.long, device=dev)
    coarse_train = torch.as_tensor(coarse_train_nodes, dtype=torch.long, device=dev)
    original_train = torch.as_tensor(train_nodes, dtype=torch.long, device=dev)

    def task_model_factory(name: str, graph: HeteroGraph):
        if name == "rgcn_lite":
            return RGCNLite(graph, model_hidden_dim=int(hidden_dim), dropout=0.2)
        if name == "han_small":
            return HANSmall(graph, model_hidden_dim=max(int(hidden_dim), 32), dropout=0.25)
        if name == "hgt_lite":
            return HGTLite(graph, model_hidden_dim=max(int(hidden_dim), 32), dropout=0.25)
        raise ValueError(f"unsupported coarse_model: {coarse_model}")

    def train_model(
        model: Any,
        data: dict[str, Any],
        y: Any,
        idx: Any,
        n_epochs: int,
        *,
        lr_value: float | None = None,
        weight_decay_value: float | None = None,
    ) -> float:
        start = perf_counter()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(lr if lr_value is None else lr_value),
            weight_decay=float(weight_decay if weight_decay_value is None else weight_decay_value),
        )
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(max(0, int(n_epochs))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = loss_fn(logits[idx], y[idx])
            loss.backward()
            optimizer.step()
        return float(perf_counter() - start)

    def eval_model(model: Any, data: dict[str, Any]) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(data)
            return logits.argmax(dim=1).detach().cpu().numpy()

    def train_refine_checkpoints(
        model: Any,
        data: dict[str, Any],
        y: Any,
        idx: Any,
        checkpoints: list[int],
    ) -> tuple[dict[int, dict[str, float]], float]:
        checkpoint_set = set(checkpoints)
        scores: dict[int, dict[str, float]] = {}
        elapsed = 0.0
        if 0 in checkpoint_set:
            pred = eval_model(model, data)
            f1 = f1_scores(
                labels[test_nodes],
                pred[test_nodes],
                macro_empty_class_policy=macro_empty_class_policy,
            )
            scores[0] = {
                "micro_f1": f1["micro_f1"],
                "macro_f1": f1["macro_f1"],
                "refine_time": 0.0,
            }
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1, max(checkpoints, default=0) + 1):
            start = perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = loss_fn(logits[idx], y[idx])
            loss.backward()
            optimizer.step()
            elapsed += float(perf_counter() - start)
            if epoch in checkpoint_set:
                pred = eval_model(model, data)
                f1 = f1_scores(
                    labels[test_nodes],
                    pred[test_nodes],
                    macro_empty_class_policy=macro_empty_class_policy,
                )
                scores[epoch] = {
                    "micro_f1": f1["micro_f1"],
                    "macro_f1": f1["macro_f1"],
                    "refine_time": float(elapsed),
                }
        return scores, float(elapsed)

    model = task_model_factory(coarse_model_name, coarse).to(dev)
    train_time = train_model(model, coarse_data, coarse_y, coarse_train, int(epochs))
    coarse_pred = eval_model(model, coarse_data)
    coarse_train_f1 = f1_scores(
        coarse_labels[coarse_train_nodes],
        coarse_pred[coarse_train_nodes],
        macro_empty_class_policy=macro_empty_class_policy,
    )
    projected_pred = coarse_pred[original_to_coarse[test_nodes]]
    projected_f1 = f1_scores(
        labels[test_nodes],
        projected_pred,
        macro_empty_class_policy=macro_empty_class_policy,
    )

    refine_model = task_model_factory(coarse_model_name, original).to(dev)
    refine_model.load_state_dict(model.state_dict(), strict=False)
    checkpoint_metrics: dict[int, dict[str, float]] = {}
    if refine_epochs_list is None:
        refine_time = train_model(refine_model, original_data, original_y, original_train, int(refine_epochs))
        original_pred = eval_model(refine_model, original_data)
        refined_f1 = f1_scores(
            labels[test_nodes],
            original_pred[test_nodes],
            macro_empty_class_policy=macro_empty_class_policy,
        )
        checkpoint_metrics = {
            int(refine_epochs): {
                "micro_f1": refined_f1["micro_f1"],
                "macro_f1": refined_f1["macro_f1"],
                "refine_time": float(refine_time),
            }
        }
    else:
        checkpoints = sorted({max(0, int(value)) for value in refine_epochs_list})
        if not checkpoints:
            checkpoints = [0]
        checkpoint_metrics, refine_time = train_refine_checkpoints(
            refine_model,
            original_data,
            original_y,
            original_train,
            checkpoints,
        )
        primary_epoch = max(checkpoints)
        refined_f1 = {
            "micro_f1": checkpoint_metrics[primary_epoch]["micro_f1"],
            "macro_f1": checkpoint_metrics[primary_epoch]["macro_f1"],
        }
    primary_metric_name = "refined_original_macro_f1"
    primary_metric = refined_f1["macro_f1"]

    metrics = {
        "model": coarse_model_name,
        "coarse_model": coarse_model_name,
        "skipped": False,
        "device": device_name,
        "train_on": "coarse_train_labels_only",
        "eval_on": "original_test_refined",
        "projection_eval_on": "original_test_projected",
        "refine_eval_on": "original_test_refined",
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "num_classes": int(num_classes),
        "train_labeled_nodes": int(len(train_nodes)),
        "val_labeled_nodes": int(len(val_nodes)),
        "test_labeled_nodes": int(len(test_nodes)),
        "coarse_train_nodes": int(len(coarse_train_nodes)),
        **task_protocol,
        **label_protocol,
        "coarse_train_micro_f1": coarse_train_f1["micro_f1"],
        "coarse_train_macro_f1": coarse_train_f1["macro_f1"],
        "projected_original_micro_f1": projected_f1["micro_f1"],
        "projected_original_macro_f1": projected_f1["macro_f1"],
        "refined_original_micro_f1": refined_f1["micro_f1"],
        "refined_original_macro_f1": refined_f1["macro_f1"],
        "primary_task_metric_name": primary_metric_name,
        "primary_task_metric": primary_metric,
        "micro_f1": refined_f1["micro_f1"],
        "macro_f1": refined_f1["macro_f1"],
        "train_time": float(train_time),
        "refine_time": float(refine_time),
        "total_time": float(train_time + refine_time),
        **refine_curve_summary(checkpoint_metrics),
    }
    for checkpoint, values in sorted(checkpoint_metrics.items()):
        suffix = f"@{checkpoint}"
        metrics[f"refined_original_micro_f1{suffix}"] = values["micro_f1"]
        metrics[f"refined_original_macro_f1{suffix}"] = values["macro_f1"]
        metrics[f"refine_time{suffix}"] = values["refine_time"]

    baseline_names = [str(name) for name in (full_graph_baselines or [])]
    if full_graph_rgcn_lite and "full_graph_rgcn_lite_default" not in baseline_names:
        baseline_names.insert(0, "full_graph_rgcn_lite_default")
    tuned_epochs = (
        max(int(epochs), int(epochs) * 2)
        if full_graph_tuned_epochs is None
        else int(full_graph_tuned_epochs)
    )
    baseline_specs: dict[str, dict[str, Any]] = {
        "full_graph_rgcn_lite_default": {
            "factory": lambda: RGCNLite(original, model_hidden_dim=int(hidden_dim), dropout=0.2),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "full_graph_rgcn_lite_tuned": {
            "factory": lambda: RGCNLite(
                original,
                model_hidden_dim=max(int(hidden_dim), 64),
                dropout=0.3,
            ),
            "epochs": tuned_epochs,
            "lr": min(float(lr), 0.005),
            "weight_decay": min(float(weight_decay), 1e-4),
        },
        "full_graph_han_small": {
            "factory": lambda: HANSmall(original, model_hidden_dim=max(int(hidden_dim), 32), dropout=0.25),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "full_graph_hgt_small": {
            "factory": lambda: HGTLite(original, model_hidden_dim=max(int(hidden_dim), 32), dropout=0.25),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "full_graph_r_hgt_lite": {
            "factory": lambda: HGTLite(original, model_hidden_dim=max(int(hidden_dim), 32), dropout=0.25),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
    }
    for baseline_name in dict.fromkeys(baseline_names):
        spec = baseline_specs.get(str(baseline_name))
        if spec is None:
            metrics[f"{baseline_name}_skipped"] = True
            metrics[f"{baseline_name}_skip_reason"] = "unknown_full_graph_baseline"
            continue
        start = perf_counter()
        try:
            full_model = spec["factory"]().to(dev)
            full_train_time = train_model(
                full_model,
                original_data,
                original_y,
                original_train,
                int(spec["epochs"]),
                lr_value=float(spec["lr"]),
                weight_decay_value=float(spec["weight_decay"]),
            )
            full_pred = eval_model(full_model, original_data)
            full_f1 = f1_scores(
                labels[test_nodes],
                full_pred[test_nodes],
                macro_empty_class_policy=macro_empty_class_policy,
            )
            metrics.update(
                {
                    f"{baseline_name}_micro_f1": full_f1["micro_f1"],
                    f"{baseline_name}_macro_f1": full_f1["macro_f1"],
                    f"{baseline_name}_train_time": float(full_train_time),
                    f"{baseline_name}_total_time": float(perf_counter() - start),
                    f"{baseline_name}_epochs": int(spec["epochs"]),
                    f"{baseline_name}_skipped": False,
                }
            )
            if baseline_name == "full_graph_rgcn_lite_default":
                metrics.update(
                    {
                        "full_graph_rgcn_lite_micro_f1": full_f1["micro_f1"],
                        "full_graph_rgcn_lite_macro_f1": full_f1["macro_f1"],
                        "full_graph_rgcn_lite_train_time": float(full_train_time),
                    }
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics.update(
                {
                    f"{baseline_name}_skipped": True,
                    f"{baseline_name}_skip_reason": str(exc),
                    f"{baseline_name}_total_time": float(perf_counter() - start),
                }
            )

    return TaskEvalResult(metrics)
