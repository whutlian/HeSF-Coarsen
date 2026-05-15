from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    valid = (y_true >= 0) & (y_pred >= 0)
    if not np.any(valid):
        return {"micro_f1": 0.0, "macro_f1": 0.0}
    truth = y_true[valid].astype(np.int64, copy=False)
    pred = y_pred[valid].astype(np.int64, copy=False)
    labels = np.union1d(truth, pred)
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
) -> TaskEvalResult:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - exercised only without torch installed.
        return TaskEvalResult(
            {
                "model": "rgcn_lite",
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
    train_nodes, test_nodes = labeled_split(labels, seed=seed)
    if len(train_nodes) == 0 or len(test_nodes) == 0:
        return TaskEvalResult(
            {
                "model": "rgcn_lite",
                "skipped": True,
                "skip_reason": "not_enough_labeled_nodes",
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

    class RGCNLite(nn.Module):
        def __init__(self, graph: HeteroGraph):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            type_ids = sorted(int(type_id) for type_id in np.unique(graph.node_type))
            self.type_linears = nn.ModuleDict()
            self.type_embeddings = nn.ParameterDict()
            for type_id in type_ids:
                feature = (graph.features or {}).get(type_id)
                if feature is None:
                    self.type_embeddings[str(type_id)] = nn.Parameter(torch.zeros(self.hidden_dim))
                else:
                    self.type_linears[str(type_id)] = nn.Linear(int(feature.shape[1]), self.hidden_dim)
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
            self.dropout = nn.Dropout(0.2)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for module in self.type_linears.values():
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            for parameter in self.type_embeddings.values():
                nn.init.normal_(parameter, std=0.02)
            for parameter in self.rel_weights.values():
                nn.init.xavier_uniform_(parameter)
            nn.init.xavier_uniform_(self.self_linear.weight)
            nn.init.zeros_(self.self_linear.bias)
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def encode(self, data: dict[str, Any]) -> Any:
            h = torch.zeros((data["num_nodes"], self.hidden_dim), device=dev)
            for type_id, node_tensor in data["type_nodes"].items():
                key = str(type_id)
                if key in self.type_linears:
                    h[node_tensor] = self.type_linears[key](data["features"][type_id])
                else:
                    h[node_tensor] = self.type_embeddings[key]
            return torch.relu(h)

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

    def train_model(model: Any, data: dict[str, Any], y: Any, idx: Any, n_epochs: int) -> float:
        start = perf_counter()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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
            f1 = f1_scores(labels[test_nodes], pred[test_nodes])
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
                f1 = f1_scores(labels[test_nodes], pred[test_nodes])
                scores[epoch] = {
                    "micro_f1": f1["micro_f1"],
                    "macro_f1": f1["macro_f1"],
                    "refine_time": float(elapsed),
                }
        return scores, float(elapsed)

    model = RGCNLite(coarse).to(dev)
    train_time = train_model(model, coarse_data, coarse_y, coarse_train, int(epochs))
    coarse_pred = eval_model(model, coarse_data)
    coarse_train_f1 = f1_scores(coarse_labels[coarse_train_nodes], coarse_pred[coarse_train_nodes])
    projected_pred = coarse_pred[original_to_coarse[test_nodes]]
    projected_f1 = f1_scores(labels[test_nodes], projected_pred)

    refine_model = RGCNLite(original).to(dev)
    refine_model.load_state_dict(model.state_dict(), strict=False)
    checkpoint_metrics: dict[int, dict[str, float]] = {}
    if refine_epochs_list is None:
        refine_time = train_model(refine_model, original_data, original_y, original_train, int(refine_epochs))
        original_pred = eval_model(refine_model, original_data)
        refined_f1 = f1_scores(labels[test_nodes], original_pred[test_nodes])
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
        "model": "rgcn_lite",
        "skipped": False,
        "device": device_name,
        "train_on": "coarse_train_labels_only",
        "eval_on": "original_test_refined",
        "projection_eval_on": "original_test_projected",
        "refine_eval_on": "original_test_refined",
        "num_classes": int(num_classes),
        "train_labeled_nodes": int(len(train_nodes)),
        "test_labeled_nodes": int(len(test_nodes)),
        "coarse_train_nodes": int(len(coarse_train_nodes)),
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

    if full_graph_rgcn_lite:
        full_model = RGCNLite(original).to(dev)
        full_train_time = train_model(full_model, original_data, original_y, original_train, int(epochs))
        full_pred = eval_model(full_model, original_data)
        full_f1 = f1_scores(labels[test_nodes], full_pred[test_nodes])
        metrics.update(
            {
                "full_graph_rgcn_lite_micro_f1": full_f1["micro_f1"],
                "full_graph_rgcn_lite_macro_f1": full_f1["macro_f1"],
                "full_graph_rgcn_lite_train_time": float(full_train_time),
            }
        )

    return TaskEvalResult(metrics)
