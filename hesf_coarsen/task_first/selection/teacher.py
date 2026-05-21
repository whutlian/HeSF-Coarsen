from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.hettree_task import (
    build_semantic_tree_features,
    enumerate_target_paths,
    evaluate_hettree_task,
    infer_target_node_type,
)
from hesf_coarsen.eval.task_gnn import f1_scores
from hesf_coarsen.io.schema import HeteroGraph
from hesf_coarsen.ops.fused_operator import apply_fused_smoothing
from hesf_coarsen.task_first.selection.config import TeacherConfig


def _mask_nodes(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


def _proxy_teacher_logits(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    target_node_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    target_nodes = np.flatnonzero(graph.node_type == int(target_node_type)).astype(np.int64)
    train_targets = target_nodes[train_mask[target_nodes] & (labels[target_nodes] >= 0)]
    classes = sorted(int(value) for value in np.unique(labels[train_targets]) if int(value) >= 0)
    if not classes:
        classes = [0]
    class_to_pos = {label: index for index, label in enumerate(classes)}
    logits = np.zeros((graph.num_nodes, len(classes)), dtype=np.float32)
    for node in train_targets:
        logits[int(node), class_to_pos[int(labels[int(node)])]] = 4.0
    response = logits.copy()
    for _step in range(2):
        response = apply_fused_smoothing(graph, response).astype(np.float32, copy=False)
    logits = logits + response
    probs = _softmax(logits)
    pred = np.asarray([classes[int(idx)] for idx in np.argmax(probs, axis=1)], dtype=np.int64)
    embeddings = probs.astype(np.float32, copy=True)
    return logits.astype(np.float32), pred, embeddings


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _target_local_indices(target_nodes: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    lookup = {int(node): idx for idx, node in enumerate(np.asarray(target_nodes).reshape(-1))}
    return np.asarray([lookup[int(node)] for node in np.asarray(nodes).reshape(-1) if int(node) in lookup], dtype=np.int64)


def _scores(labels: np.ndarray, target_nodes: np.ndarray, local_idx: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(local_idx) == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "accuracy": 0.0}
    y_true = np.asarray(labels)[target_nodes[local_idx]]
    y_pred = np.asarray(pred)[local_idx]
    valid = (y_true >= 0) & (y_pred >= 0)
    accuracy = float(np.mean(y_true[valid] == y_pred[valid])) if np.any(valid) else 0.0
    return {
        **f1_scores(y_true, y_pred, macro_empty_class_policy="truth_pred_union"),
        "accuracy": accuracy,
    }


def _train_target_logits(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    test_nodes: np.ndarray,
    *,
    target_type: int,
    seed: int,
    epochs: int,
    hidden_dim: int,
    lr: float,
    dropout: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        return {"skipped": True, "skip_reason": f"torch_unavailable: {exc}"}

    dev = torch.device("cuda" if str(device) == "auto" and torch.cuda.is_available() else str(device if device != "auto" else "cpu"))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    paths = enumerate_target_paths(graph, target_type=int(target_type), max_hops=2, max_paths=32)
    tree = build_semantic_tree_features(graph, target_type=int(target_type), paths=paths)
    target_nodes = tree.target_nodes.astype(np.int64, copy=False)
    train_local = _target_local_indices(target_nodes, train_nodes)
    val_local = _target_local_indices(target_nodes, val_nodes)
    test_local = _target_local_indices(target_nodes, test_nodes)
    train_local = train_local[np.asarray(labels)[target_nodes[train_local]] >= 0]
    if len(train_local) == 0:
        return {"skipped": True, "skip_reason": "no_teacher_train_targets"}
    classes = sorted(int(value) for value in np.unique(labels[train_nodes]) if int(value) >= 0)
    if not classes:
        classes = sorted(int(value) for value in np.unique(labels[target_nodes]) if int(value) >= 0)
    class_to_pos = {label: idx for idx, label in enumerate(classes)}
    y = np.full(len(target_nodes), -1, dtype=np.int64)
    for idx, node in enumerate(target_nodes):
        label = int(labels[int(node)])
        if label in class_to_pos:
            y[idx] = class_to_pos[label]
    if np.any(y[train_local] < 0):
        train_local = train_local[y[train_local] >= 0]
    num_classes = max(1, len(classes))

    class HetTreeTeacher(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.path_linear = nn.Linear(int(input_dim), int(hidden_dim))
            self.attention = nn.Linear(int(hidden_dim), 1, bias=False)
            self.classifier = nn.Linear(int(hidden_dim), int(num_classes))
            self.dropout = nn.Dropout(float(dropout))

        def encode(self, x: Any) -> Any:
            h = torch.relu(self.path_linear(x))
            score = self.attention(torch.tanh(h)).squeeze(-1)
            alpha = torch.softmax(score, dim=1)
            return torch.sum(h * alpha.unsqueeze(-1), dim=1)

        def forward(self, x: Any) -> Any:
            return self.classifier(self.dropout(self.encode(x)))

    x = torch.as_tensor(tree.tensor, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(y, dtype=torch.long, device=dev)
    train_idx = torch.as_tensor(train_local, dtype=torch.long, device=dev)
    model = HetTreeTeacher(int(tree.tensor.shape[2])).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_epoch = -1
    best_val = -float("inf")
    patience_left = max(1, int(patience))
    early_stopped = False
    for epoch in range(max(0, int(epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits[train_idx], y_t[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_pos = model(x).argmax(dim=1).detach().cpu().numpy()
        pred_labels = np.asarray([classes[int(pos)] for pos in pred_pos], dtype=np.int64)
        val_score = _scores(labels, target_nodes, val_local, pred_labels)["macro_f1"] if len(val_local) else -float(loss.detach().cpu().item())
        if val_score > best_val + 1.0e-6:
            best_val = float(val_score)
            best_epoch = int(epoch)
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            patience_left = max(1, int(patience))
        else:
            patience_left -= 1
            if len(val_local) and patience_left <= 0:
                early_stopped = True
                break
    model.load_state_dict({name: tensor.to(dev) for name, tensor in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits_t = model(x)
        embeddings_t = model.encode(x)
    target_logits = logits_t.detach().cpu().numpy().astype(np.float32)
    probs = _softmax(target_logits)
    pred_pos = np.argmax(probs, axis=1)
    pred_labels = np.asarray([classes[int(pos)] for pos in pred_pos], dtype=np.int64)
    train_scores = _scores(labels, target_nodes, train_local, pred_labels)
    val_scores = _scores(labels, target_nodes, val_local, pred_labels)
    test_scores = _scores(labels, target_nodes, test_local, pred_labels)
    full_logits = np.zeros((graph.num_nodes, target_logits.shape[1]), dtype=np.float32)
    full_probs = np.zeros_like(full_logits)
    full_embeddings = np.zeros((graph.num_nodes, embeddings_t.shape[1]), dtype=np.float32)
    full_logits[target_nodes] = target_logits
    full_probs[target_nodes] = probs
    full_embeddings[target_nodes] = embeddings_t.detach().cpu().numpy().astype(np.float32)
    full_pred = np.full(graph.num_nodes, -1, dtype=np.int64)
    full_pred[target_nodes] = pred_labels
    return {
        "skipped": False,
        "target_nodes": target_nodes,
        "logits_by_target": target_logits,
        "probs_by_target": probs,
        "embeddings_by_target": embeddings_t.detach().cpu().numpy().astype(np.float32),
        "predictions_by_target": pred_labels,
        "full_logits": full_logits,
        "full_probs": full_probs,
        "full_embeddings": full_embeddings,
        "full_predictions": full_pred,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "classes": classes,
    }


def train_full_graph_lite_teacher(
    graph: HeteroGraph,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    cfg: TeacherConfig,
    *,
    output_dir: str | Path | None = None,
    seed: int = 12345,
    epochs: int = 10,
    hidden_dim: int = 32,
    device: str = "auto",
    use_config_grid: bool = False,
    restarts: int | None = None,
) -> dict[str, Any]:
    target_type = infer_target_node_type(graph)
    labels = np.asarray(labels)
    train_nodes = _mask_nodes(train_mask)
    val_nodes = _mask_nodes(val_mask)
    test_nodes = _mask_nodes(test_mask)
    metrics = evaluate_hettree_task(
        graph,
        graph,
        np.arange(graph.num_nodes, dtype=np.int64),
        seed=int(seed),
        epochs=int(epochs),
        hidden_dim=int(hidden_dim),
        device=str(device),
        target_node_type=int(target_type),
        official_split_nodes={"train": train_nodes, "val": val_nodes, "test": test_nodes},
        primary_eval_mode="compressed_projected",
        early_stopping=True,
        patience=int(cfg.patience),
        monitor="projected_val_macro_f1",
    ).metrics
    grid_epochs = tuple(int(value) for value in cfg.epochs_grid) if use_config_grid else (int(epochs),)
    grid_hidden = tuple(int(value) for value in cfg.hidden_dim_grid) if use_config_grid else (int(hidden_dim),)
    grid_lr = tuple(float(value) for value in cfg.lr_grid) if use_config_grid else (0.005,)
    grid_dropout = tuple(float(value) for value in cfg.dropout_grid) if use_config_grid else (0.25,)
    grid_wd = tuple(float(value) for value in cfg.weight_decay_grid) if use_config_grid else (1.0e-4,)
    restart_count = int(restarts if restarts is not None else (cfg.restarts if use_config_grid else 1))
    grid_results: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None
    best_score = -float("inf")
    for epoch_value in grid_epochs:
        for hidden_value in grid_hidden:
            for lr_value in grid_lr:
                for dropout_value in grid_dropout:
                    for wd_value in grid_wd:
                        for restart in range(max(1, restart_count)):
                            run_seed = int(seed) + int(restart) * 1009
                            trained = _train_target_logits(
                                graph,
                                labels,
                                train_nodes,
                                val_nodes,
                                test_nodes,
                                target_type=int(target_type),
                                seed=run_seed,
                                epochs=int(epoch_value),
                                hidden_dim=int(hidden_value),
                                lr=float(lr_value),
                                dropout=float(dropout_value),
                                weight_decay=float(wd_value),
                                patience=int(cfg.patience),
                                device=str(device),
                            )
                            row = {
                                "seed": int(seed),
                                "restart": int(restart),
                                "run_seed": int(run_seed),
                                "epochs": int(epoch_value),
                                "hidden_dim": int(hidden_value),
                                "lr": float(lr_value),
                                "dropout": float(dropout_value),
                                "weight_decay": float(wd_value),
                                "skipped": bool(trained.get("skipped", False)),
                                "skip_reason": trained.get("skip_reason", ""),
                                "best_epoch": trained.get("best_epoch", -1),
                                "validation_macro_f1": (trained.get("val_scores") or {}).get("macro_f1", 0.0),
                                "test_macro_f1": (trained.get("test_scores") or {}).get("macro_f1", 0.0),
                                "test_accuracy": (trained.get("test_scores") or {}).get("accuracy", 0.0),
                            }
                            grid_results.append(row)
                            score = float(row["validation_macro_f1"])
                            if not row["skipped"] and score > best_score:
                                best_score = score
                                best_run = trained | {"best_config": row}
    if best_run is None:
        proxy_logits, proxy_pred, proxy_embeddings = _proxy_teacher_logits(graph, labels, train_mask, int(target_type))
        logits = proxy_logits
        pred = proxy_pred
        embeddings = proxy_embeddings
        target_nodes = np.flatnonzero(graph.node_type == int(target_type)).astype(np.int64)
        logits_by_target = logits[target_nodes]
        probs_by_target = _softmax(logits_by_target)
        embeddings_by_target = embeddings[target_nodes]
        predictions_by_target = pred[target_nodes]
        best_config: dict[str, Any] = {"fallback": "proxy_label_smoothing"}
        logits_source = "proxy_label_smoothing"
        teacher_reliable = False
        best_epoch = -1
    else:
        logits = np.asarray(best_run["full_logits"], dtype=np.float32)
        pred = np.asarray(best_run["full_predictions"], dtype=np.int64)
        embeddings = np.asarray(best_run["full_embeddings"], dtype=np.float32)
        target_nodes = np.asarray(best_run["target_nodes"], dtype=np.int64)
        logits_by_target = np.asarray(best_run["logits_by_target"], dtype=np.float32)
        probs_by_target = np.asarray(best_run["probs_by_target"], dtype=np.float32)
        embeddings_by_target = np.asarray(best_run["embeddings_by_target"], dtype=np.float32)
        predictions_by_target = np.asarray(best_run["predictions_by_target"], dtype=np.int64)
        best_config = dict(best_run["best_config"])
        logits_source = "trained_teacher"
        teacher_reliable = bool(float((best_run.get("val_scores") or {}).get("macro_f1", 0.0)) >= 0.35)
        best_epoch = int(best_run.get("best_epoch", -1))
    support_proxy = None
    if cfg.proxy_logits_mode != "disabled":
        support_proxy, _proxy_pred, _proxy_embeddings = _proxy_teacher_logits(graph, labels, train_mask, int(target_type))
    teacher_metrics = {
        "model": str(cfg.model),
        "evaluator_status": "diagnostic_lite_only",
        "full_graph_teacher_primary_eval_mode": metrics.get("primary_eval_mode", "compressed_projected"),
        "full_graph_teacher_primary_task_metric_name": metrics.get("primary_task_metric_name", ""),
        "full_graph_teacher_macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "full_graph_teacher_micro_f1": float(metrics.get("micro_f1", 0.0) or 0.0),
        "full_graph_teacher_accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "full_graph_teacher_projected_macro_f1": float(metrics.get("projected_original_macro_f1", 0.0) or 0.0),
        "full_graph_teacher_transfer_macro_f1": float(metrics.get("transfer_original_macro_f1", 0.0) or 0.0),
        "full_graph_teacher_projected_accuracy": float(metrics.get("projected_original_accuracy", 0.0) or 0.0),
        "full_graph_teacher_transfer_accuracy": float(metrics.get("transfer_original_accuracy", 0.0) or 0.0),
        "projected_vs_transfer_macro_gap": float(metrics.get("projected_vs_transfer_macro_gap", 0.0) or 0.0),
        "projected_vs_transfer_accuracy_gap": float(metrics.get("projected_vs_transfer_accuracy_gap", 0.0) or 0.0),
        "validation_macro_f1": float(metrics.get("validation_macro_f1", 0.0) or 0.0),
        "validation_accuracy": float(metrics.get("validation_accuracy", 0.0) or 0.0),
        "test_labels_used_for_training": False,
        "teacher_uses_test_labels_for_training": False,
        "logits_source": logits_source,
        "teacher_reliable_for_importance": teacher_reliable,
        "teacher_best_epoch": int(best_epoch),
        "teacher_best_config_hash": json.dumps(best_config, sort_keys=True),
        "train_nodes": int(len(train_nodes)),
        "val_nodes": int(len(val_nodes)),
        "test_nodes": int(len(test_nodes)),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if cfg.save_logits:
            np.save(root / "teacher_logits.npy", logits)
            np.save(root / "target_logits.npy", logits_by_target)
            np.save(root / "all_target_probabilities.npy", probs_by_target)
        np.save(root / "teacher_pred.npy", pred)
        np.save(root / "teacher_predictions.npy", predictions_by_target)
        if cfg.save_embeddings:
            np.save(root / "teacher_embeddings.npy", embeddings)
            np.save(root / "target_embeddings.npy", embeddings_by_target)
        _write_metrics_csv(root / "teacher_metrics.csv", teacher_metrics)
        _write_rows_csv(root / "teacher_grid_results.csv", grid_results)
        (root / "teacher_config.json").write_text(
            json.dumps(
                asdict(cfg) | {"best_config": best_config},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return {
        "target_nodes": target_nodes,
        "logits_by_target": logits_by_target,
        "probs_by_target": probs_by_target,
        "embeddings_by_target": embeddings_by_target,
        "predictions_by_target": predictions_by_target,
        "support_logits_proxy": support_proxy,
        "support_contribution": None,
        "logits": logits,
        "probs": _softmax(logits),
        "predictions": pred,
        "embeddings": embeddings,
        "metrics": teacher_metrics,
        "grid_results": grid_results,
        "best_config": best_config,
        "raw_eval_metrics": metrics,
        "config_hash": json.dumps(best_config, sort_keys=True),
        "logits_source": logits_source,
        "teacher_uses_test_labels_for_training": False,
    }
