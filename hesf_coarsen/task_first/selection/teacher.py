from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.eval.hettree_task import evaluate_hettree_task, infer_target_node_type
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
    ).metrics
    logits, pred, embeddings = _proxy_teacher_logits(graph, labels, train_mask, int(target_type))
    teacher_metrics = {
        "model": str(cfg.model),
        "evaluator_status": "diagnostic_lite_only",
        "full_graph_teacher_macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "full_graph_teacher_micro_f1": float(metrics.get("micro_f1", 0.0) or 0.0),
        "full_graph_teacher_accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "validation_macro_f1": float(metrics.get("validation_macro_f1", 0.0) or 0.0),
        "validation_accuracy": float(metrics.get("validation_accuracy", 0.0) or 0.0),
        "test_labels_used_for_training": False,
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
        np.save(root / "teacher_pred.npy", pred)
        if cfg.save_embeddings:
            np.save(root / "teacher_embeddings.npy", embeddings)
        _write_metrics_csv(root / "teacher_metrics.csv", teacher_metrics)
        (root / "teacher_config.json").write_text(
            json.dumps(
                {
                    "enabled": bool(cfg.enabled),
                    "model": str(cfg.model),
                    "require_official_for_paper_claim": bool(cfg.require_official_for_paper_claim),
                    "tune_full_graph_lite": bool(cfg.tune_full_graph_lite),
                    "save_logits": bool(cfg.save_logits),
                    "save_embeddings": bool(cfg.save_embeddings),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return {
        "logits": logits,
        "predictions": pred,
        "embeddings": embeddings,
        "metrics": teacher_metrics,
        "raw_eval_metrics": metrics,
        "config_hash": f"{cfg.model}:lite:{int(seed)}:{int(epochs)}:{int(hidden_dim)}",
        "teacher_uses_test_labels_for_training": False,
    }
