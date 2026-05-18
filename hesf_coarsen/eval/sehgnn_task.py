from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from hesf_coarsen.eval.hettree_task import (
    _classification_scores,
    _feature_width,
    _local_indices,
    _type_ids,
    build_semantic_tree_features,
    enumerate_target_paths,
    infer_target_node_type,
)
from hesf_coarsen.eval.task_gnn import (
    TaskEvalResult,
    resolve_target_node_type,
    select_task_protocol_split,
    train_only_coarse_labels,
)
from hesf_coarsen.io.schema import HeteroGraph


@dataclass(frozen=True)
class SeHGNNReference:
    repository: str = "ICT-GIMLab/SeHGNN"
    architecture: str = "per-metapath projection + transformer semantic fusion"


def evaluate_sehgnn_task(
    original: HeteroGraph,
    coarse: HeteroGraph,
    original_to_coarse: np.ndarray,
    *,
    seed: int = 12345,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 1.0e-4,
    dropout: float = 0.35,
    input_dropout: float = 0.1,
    attention_dropout: float = 0.2,
    num_heads: int = 1,
    num_feature_projection_layers: int = 2,
    num_task_layers: int = 2,
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
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - exercised only without torch installed.
        return TaskEvalResult(
            {
                "model": "sehgnn_lite",
                "skipped": True,
                "skip_reason": f"torch_unavailable: {exc}",
            }
        )

    labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1))
    if target_node_type is None or str(target_node_type) == "":
        target_type = infer_target_node_type(original)
    else:
        resolved = resolve_target_node_type(original, target_node_type)
        target_type = infer_target_node_type(original) if resolved is None else int(resolved)

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
                "model": "sehgnn_lite",
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
                "model": "sehgnn_lite",
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

    feature_start = perf_counter()
    paths = enumerate_target_paths(
        original,
        target_type=int(target_type),
        max_hops=int(max_hops),
        max_paths=max_paths,
    )
    width = _feature_width([original, coarse])
    ids = _type_ids([original, coarse])
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
    original_test_local = _local_indices(test_nodes, original_tree.target_nodes)
    if len(coarse_train_local) == 0 or len(original_test_local) == 0:
        return TaskEvalResult(
            {
                "model": "sehgnn_lite",
                "skipped": True,
                "skip_reason": "no_target_type_train_or_test_nodes",
                **task_protocol,
                **label_protocol,
            }
        )

    num_classes = int(labels[labels >= 0].max(initial=0)) + 1
    input_dim = int(coarse_tree.tensor.shape[2])
    channels = int(coarse_tree.tensor.shape[1])
    model_hidden_dim = max(int(hidden_dim), int(num_heads) * 4)
    if model_hidden_dim % (int(num_heads) * 4) != 0:
        step = int(num_heads) * 4
        model_hidden_dim = ((model_hidden_dim + step - 1) // step) * step
    reference = SeHGNNReference()

    class LinearPerMetapath(nn.Module):
        def __init__(self, cin: int, cout: int, num_metapaths: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(int(num_metapaths), int(cin), int(cout)))
            self.bias = nn.Parameter(torch.zeros(int(num_metapaths), int(cout)))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.bias)

        def forward(self, x: Any) -> Any:
            return torch.einsum("bcd,cdh->bch", x, self.weight) + self.bias.unsqueeze(0)

    class TransformerFusion(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.query = nn.Linear(int(hidden), int(hidden) // 4)
            self.key = nn.Linear(int(hidden), int(hidden) // 4)
            self.value = nn.Linear(int(hidden), int(hidden))
            self.gamma = nn.Parameter(torch.tensor([0.0]))
            self.attn_drop = nn.Dropout(float(attention_dropout))

        def forward(self, x: Any) -> Any:
            batch, metapaths, hidden = x.shape
            heads = int(num_heads)
            q = self.query(x).view(batch, metapaths, heads, -1).permute(0, 2, 1, 3)
            k = self.key(x).view(batch, metapaths, heads, -1).permute(0, 2, 3, 1)
            v = self.value(x).view(batch, metapaths, heads, -1).permute(0, 2, 1, 3)
            beta = F.softmax(torch.relu(q @ k / float(q.size(-1)) ** 0.5), dim=-1)
            beta = self.attn_drop(beta)
            out = self.gamma * (beta @ v)
            return out.permute(0, 2, 1, 3).reshape(batch, metapaths, hidden) + x

    class SeHGNNLite(nn.Module):
        def __init__(self):
            super().__init__()
            blocks: list[nn.Module] = [
                LinearPerMetapath(input_dim, model_hidden_dim, channels),
                nn.LayerNorm([channels, model_hidden_dim]),
                nn.PReLU(),
                nn.Dropout(float(dropout)),
            ]
            for _ in range(max(0, int(num_feature_projection_layers) - 1)):
                blocks.extend(
                    [
                        LinearPerMetapath(model_hidden_dim, model_hidden_dim, channels),
                        nn.LayerNorm([channels, model_hidden_dim]),
                        nn.PReLU(),
                        nn.Dropout(float(dropout)),
                    ]
                )
            self.input_drop = nn.Dropout(float(input_dropout))
            self.feature_projection = nn.Sequential(*blocks)
            self.semantic_fusion = TransformerFusion(model_hidden_dim)
            self.fc_after_concat = nn.Linear(channels * model_hidden_dim, model_hidden_dim)
            task_layers: list[nn.Module] = [nn.PReLU(), nn.Dropout(float(dropout))]
            for _ in range(max(0, int(num_task_layers) - 1)):
                task_layers.extend(
                    [
                        nn.Linear(model_hidden_dim, model_hidden_dim),
                        nn.LayerNorm(model_hidden_dim),
                        nn.PReLU(),
                        nn.Dropout(float(dropout)),
                    ]
                )
            task_layers.append(nn.Linear(model_hidden_dim, int(num_classes)))
            self.task_mlp = nn.Sequential(*task_layers)

        def forward(self, x: Any) -> Any:
            x = self.input_drop(x)
            x = self.feature_projection(x)
            x = self.semantic_fusion(x).transpose(1, 2)
            hidden = self.fc_after_concat(x.reshape(x.shape[0], -1))
            return self.task_mlp(hidden)

    coarse_x = torch.as_tensor(coarse_tree.tensor, dtype=torch.float32, device=dev)
    original_x = torch.as_tensor(original_tree.tensor, dtype=torch.float32, device=dev)
    coarse_y = torch.as_tensor(coarse_labels[coarse_tree.target_nodes], dtype=torch.long, device=dev)
    train_idx = torch.as_tensor(coarse_train_local, dtype=torch.long, device=dev)
    model = SeHGNNLite().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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
            "model": "sehgnn_lite",
            "architecture_reference": reference.repository,
            "architecture_note": reference.architecture,
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
            "input_dim": int(input_dim),
            "hidden_dim": int(model_hidden_dim),
            "epochs": int(epochs),
            "sehgnn_num_channels": int(channels),
            "sehgnn_num_heads": int(num_heads),
            "sehgnn_feature_projection_layers": int(num_feature_projection_layers),
            "sehgnn_task_layers": int(num_task_layers),
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
