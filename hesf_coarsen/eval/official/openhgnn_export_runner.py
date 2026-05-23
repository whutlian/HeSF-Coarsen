from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hesf_coarsen.eval.official.metrics import classification_metrics_from_logits
from hesf_coarsen.eval.official.runner_utils import write_json
from hesf_coarsen.eval.official.sehgnn_export_runner import build_target_feature_blocks


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


def _install_openhgnn_model_stubs() -> None:
    import torch.nn as nn

    openhgnn_pkg = sys.modules.get("openhgnn")
    if openhgnn_pkg is None:
        openhgnn_pkg = types.ModuleType("openhgnn")
        openhgnn_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["openhgnn"] = openhgnn_pkg

    models_pkg = types.ModuleType("openhgnn.models")
    models_pkg.__path__ = []  # type: ignore[attr-defined]

    class BaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    def register_model(_name: str):
        def decorator(cls):
            return cls

        return decorator

    models_pkg.BaseModel = BaseModel  # type: ignore[attr-defined]
    models_pkg.register_model = register_model  # type: ignore[attr-defined]
    sys.modules["openhgnn.models"] = models_pkg


def load_openhgnn_sehgnn_class(repo_dir: Path) -> type:
    model_path = Path(repo_dir) / "openhgnn" / "models" / "SeHGNN.py"
    if not model_path.exists():
        raise FileNotFoundError(f"OpenHGNN SeHGNN model not found: {model_path}")
    _install_openhgnn_model_stubs()
    module_name = "openhgnn.models.SeHGNN"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load OpenHGNN SeHGNN from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.SeHGNN


class _Args:
    pass


def _predict(model: Any, fk_all: dict[str, dict[str, Any]], indices: np.ndarray, *, batch_size: int) -> Any:
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(indices.size), int(batch_size)):
            batch_np = indices[start : start + int(batch_size)]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=next(model.parameters()).device)
            feats = {key: value[batch] for key, value in fk_all["0"].items()}
            outputs.append(model({"0": feats, "1": {}, "2": None}).detach().cpu())
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
    device_name: str = "cuda",
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
    feature_blocks = build_target_feature_blocks(export_dir, target_type)
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    args = _Args()
    args.data_size = {key: int(value.shape[1]) for key, value in feature_blocks.items()}
    args.nfeat = int(embed_size)
    args.hidden = int(hidden)
    args.nclass = int(num_classes)
    args.num_feats = int(len(feature_blocks))
    args.num_label_feats = 0
    args.dropout = 0.5
    args.input_drop = 0.1
    args.att_drop = 0.0
    args.label_drop = 0.0
    args.n_layers_1 = 2
    args.n_layers_2 = 1
    args.n_layers_3 = 2
    args.act = "none"
    args.residual = False
    args.bns = False
    args.label_bns = False
    args.label_residual = False
    args.dataset = str(dataset_name)
    args.tgt_key = str(target_type)

    Model = load_openhgnn_sehgnn_class(Path(repo_dir))
    model = Model(args).to(device)
    feats_all = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in feature_blocks.items()}
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(int(seed))
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(int(epochs)):
        model.train()
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        for start in range(0, int(shuffled.size), int(batch_size)):
            batch_np = shuffled[start : start + int(batch_size)]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            logits = model({"0": {key: value[batch] for key, value in feats_all.items()}, "1": {}, "2": None})
            loss = loss_fn(logits, labels_t[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_logits_epoch = _predict(model, {"0": feats_all}, val_idx, batch_size=int(batch_size))
        val_loss = float(loss_fn(val_logits_epoch.to(device), labels_t[torch.as_tensor(val_idx, dtype=torch.long, device=device)]).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits = _predict(model, {"0": feats_all}, val_idx, batch_size=int(batch_size)).numpy()
    test_logits = _predict(model, {"0": feats_all}, test_idx, batch_size=int(batch_size)).numpy()
    val_metrics = classification_metrics_from_logits(val_logits, labels[val_idx])
    test_metrics = classification_metrics_from_logits(test_logits, labels[test_idx])
    run_name = f"openhgnn_sehgnn_{dataset_name}_{int(seed)}_{export_dir.parent.name}_{export_dir.name}"
    val_logits_path = logits_dir / f"{run_name}_val_logits.npy"
    test_logits_path = logits_dir / f"{run_name}_test_logits.npy"
    np.save(val_logits_path, val_logits.astype(np.float32, copy=False))
    np.save(test_logits_path, test_logits.astype(np.float32, copy=False))
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
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
        "feature_dims": args.data_size,
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
    parser.add_argument("--device", default="cuda")
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
