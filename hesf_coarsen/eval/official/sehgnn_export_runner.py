from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hesf_coarsen.eval.official.metrics import classification_metrics_from_logits
from hesf_coarsen.eval.official.runner_utils import write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_feature_matrix(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"feature matrix must be 2-D: {path}")
    return np.nan_to_num(arr, copy=False)


def _weighted_mean_by_target(
    *,
    target_index: np.ndarray,
    neighbor_index: np.ndarray,
    weights: np.ndarray,
    target_count: int,
    neighbor_features: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    target_index = np.asarray(target_index, dtype=np.int64).reshape(-1)
    neighbor_index = np.asarray(neighbor_index, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if not (target_index.shape == neighbor_index.shape == weights.shape):
        raise ValueError("edge index and weight arrays must have equal length")
    out = np.zeros((int(target_count), int(neighbor_features.shape[1])), dtype=np.float32)
    denom = np.zeros(int(target_count), dtype=np.float32)
    if target_index.size == 0:
        return out
    valid = (
        (target_index >= 0)
        & (target_index < int(target_count))
        & (neighbor_index >= 0)
        & (neighbor_index < int(neighbor_features.shape[0]))
        & np.isfinite(weights)
    )
    target_index = target_index[valid]
    neighbor_index = neighbor_index[valid]
    weights = weights[valid]
    for start in range(0, int(target_index.size), int(chunk_size)):
        end = min(start + int(chunk_size), int(target_index.size))
        rows = target_index[start:end]
        nbrs = neighbor_index[start:end]
        w = weights[start:end]
        np.add.at(out, rows, neighbor_features[nbrs] * w[:, None])
        np.add.at(denom, rows, w)
    nonzero = denom > 0
    out[nonzero] /= denom[nonzero, None]
    return out


def build_target_feature_blocks(export_dir: Path, target_type: str) -> dict[str, np.ndarray]:
    export_dir = Path(export_dir)
    meta = _load_json(export_dir / "metadata.json")
    target_type = str(target_type or meta.get("target_type", ""))
    type_counts = {str(k): int(v) for k, v in dict(meta.get("num_nodes_by_type", {})).items()}
    if target_type not in type_counts:
        raise ValueError(f"target_type {target_type!r} missing from export metadata")
    node_feature_dir = export_dir / "node_features"
    edge_dir = export_dir / "edges"
    feature_cache: dict[str, np.ndarray] = {}

    def feature_for(node_type: str) -> np.ndarray:
        node_type = str(node_type)
        if node_type not in feature_cache:
            feature_cache[node_type] = _as_feature_matrix(node_feature_dir / f"{node_type}.npy")
        return feature_cache[node_type]

    blocks: dict[str, np.ndarray] = {target_type: feature_for(target_type)}
    target_count = int(type_counts[target_type])
    for schema in list(meta.get("relation_schemas", [])):
        name = str(schema["name"])
        src_type = str(schema["src_type"])
        dst_type = str(schema["dst_type"])
        edge_path = edge_dir / f"{name}.npy"
        if not edge_path.exists():
            continue
        edges = np.asarray(np.load(edge_path), dtype=np.float32)
        if edges.ndim != 2 or edges.shape[1] < 2:
            raise ValueError(f"edge file must have at least two columns: {edge_path}")
        weights = edges[:, 2] if edges.shape[1] >= 3 else np.ones(edges.shape[0], dtype=np.float32)
        if src_type == target_type:
            blocks[f"{name}__dst_mean"] = _weighted_mean_by_target(
                target_index=edges[:, 0],
                neighbor_index=edges[:, 1],
                weights=weights,
                target_count=target_count,
                neighbor_features=feature_for(dst_type),
            )
        if dst_type == target_type:
            blocks[f"{name}__src_mean"] = _weighted_mean_by_target(
                target_index=edges[:, 1],
                neighbor_index=edges[:, 0],
                weights=weights,
                target_count=target_count,
                neighbor_features=feature_for(src_type),
            )
    return {key: np.asarray(value, dtype=np.float32) for key, value in blocks.items()}


def _load_official_sehgnn(repo_dir: Path) -> type:
    model_path = Path(repo_dir) / "hgb" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"official SeHGNN model not found: {model_path}")
    spec = importlib.util.spec_from_file_location("_gate21_official_sehgnn_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official SeHGNN model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SeHGNN


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _predict(model: Any, feats: Mapping[str, Any], indices: np.ndarray, *, device: Any, batch_size: int) -> Any:
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(indices.size), int(batch_size)):
            batch_np = indices[start : start + int(batch_size)]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            batch_feats = {key: value[batch] for key, value in feats.items()}
            outputs.append(model(batch, batch_feats, {}, None).detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)


def train_export(
    *,
    export_dir: Path,
    repo_dir: Path,
    dataset_name: str,
    target_type: str,
    seed: int,
    result_json: Path,
    logits_dir: Path,
    epochs: int = 12,
    embed_size: int = 64,
    hidden: int = 64,
    batch_size: int = 2048,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device_name: str = "cpu",
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    start_time = time.perf_counter()
    export_dir = Path(export_dir)
    logits_dir = Path(logits_dir)
    logits_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(int(seed))

    labels = np.asarray(np.load(export_dir / "labels.npy"), dtype=np.int64).reshape(-1)
    train_idx = np.asarray(np.load(export_dir / "splits" / "train_idx.npy"), dtype=np.int64).reshape(-1)
    val_idx = np.asarray(np.load(export_dir / "splits" / "val_idx.npy"), dtype=np.int64).reshape(-1)
    test_idx = np.asarray(np.load(export_dir / "splits" / "test_idx.npy"), dtype=np.int64).reshape(-1)
    labeled = np.concatenate([train_idx, val_idx, test_idx])
    valid_labels = labels[labeled]
    valid_labels = valid_labels[valid_labels >= 0]
    if valid_labels.size == 0:
        raise ValueError("Gate21 export has no non-negative labels")
    num_classes = int(valid_labels.max()) + 1
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError("train/val/test splits must all be non-empty")

    feature_blocks = build_target_feature_blocks(export_dir, target_type)
    SeHGNN = _load_official_sehgnn(Path(repo_dir))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feats = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in feature_blocks.items()}
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    data_size = {key: int(value.shape[1]) for key, value in feature_blocks.items()}
    model_dataset = dataset_name if dataset_name in {"DBLP", "ACM"} else "DBLP"
    model = SeHGNN(
        model_dataset,
        int(embed_size),
        int(hidden),
        int(num_classes),
        feature_blocks.keys(),
        [],
        str(target_type),
        0.5,
        0.1,
        0.0,
        1,
        1,
        "none",
        False,
        data_size=data_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_val_loss = math.inf
    rng = np.random.default_rng(int(seed))
    train_times: list[float] = []

    for epoch in range(int(epochs)):
        model.train()
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        epoch_start = time.perf_counter()
        for start in range(0, int(shuffled.size), int(batch_size)):
            batch_np = shuffled[start : start + int(batch_size)]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            batch_feats = {key: value[batch] for key, value in feats.items()}
            logits = model(batch, batch_feats, {}, None)
            loss = loss_fn(logits, labels_t[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_times.append(float(time.perf_counter() - epoch_start))

        val_logits_epoch = _predict(model, feats, val_idx, device=device, batch_size=int(batch_size))
        val_loss = float(loss_fn(val_logits_epoch.to(device), labels_t[torch.as_tensor(val_idx, dtype=torch.long, device=device)]).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits = _predict(model, feats, val_idx, device=device, batch_size=int(batch_size)).numpy()
    test_logits = _predict(model, feats, test_idx, device=device, batch_size=int(batch_size)).numpy()
    val_metrics = classification_metrics_from_logits(val_logits, labels[val_idx])
    test_metrics = classification_metrics_from_logits(test_logits, labels[test_idx])
    run_name = f"sehgnn_official_{dataset_name}_{int(seed)}_{export_dir.parent.name}_{export_dir.name}"
    val_logits_path = logits_dir / f"{run_name}_val_logits.npy"
    test_logits_path = logits_dir / f"{run_name}_test_logits.npy"
    np.save(val_logits_path, val_logits.astype(np.float32, copy=False))
    np.save(test_logits_path, test_logits.astype(np.float32, copy=False))
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    else:
        try:
            import psutil  # type: ignore

            peak_memory_mb = float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
        except Exception:
            peak_memory_mb = 0.0
    result = {
        "status": "success",
        "validation_macro_f1": float(val_metrics["macro_f1"]),
        "validation_micro_f1": float(val_metrics["micro_f1"]),
        "validation_accuracy": float(val_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_micro_f1": float(test_metrics["micro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "val_logits_path": str(val_logits_path),
        "test_logits_path": str(test_logits_path),
        "best_epoch": int(best_epoch),
        "train_time_sec": float(time.perf_counter() - start_time),
        "peak_memory_mb": peak_memory_mb,
        "feature_keys": sorted(feature_blocks),
        "feature_dims": data_size,
        "average_epoch_train_time_sec": float(np.mean(train_times)) if train_times else 0.0,
        "calibration_uses_test_labels": False,
        "selector_uses_test_labels": False,
        "uses_hettree_lite": False,
    }
    write_json(Path(result_json), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--target-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--logits-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        result = train_export(
            export_dir=args.export_dir,
            repo_dir=args.repo_dir,
            dataset_name=args.dataset_name,
            target_type=args.target_type,
            seed=int(args.seed),
            result_json=args.result_json,
            logits_dir=args.logits_dir,
            epochs=int(args.epochs),
            embed_size=int(args.embed_size),
            hidden=int(args.hidden),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device_name=str(args.device),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except RuntimeError as exc:
        status = "failed_oom" if "out of memory" in str(exc).lower() else "failed_runtime"
        write_json(args.result_json, {"status": status, "error_message": str(exc)})
        raise
    except Exception as exc:
        write_json(args.result_json, {"status": "failed_runtime", "error_message": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
