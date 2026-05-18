from __future__ import annotations

from typing import Any

import numpy as np

from hesf_coarsen.eval.hettree_task import evaluate_hettree_task
from hesf_coarsen.eval.sehgnn_task import evaluate_sehgnn_task
from hesf_coarsen.eval.task_gnn import TaskEvalResult
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type
from hesf_coarsen.accuracy.full_target_protocol import make_protocol_row
from hesf_coarsen.accuracy.model_fidelity_registry import fidelity_record


def validate_target_preserve_adapter(
    original: HeteroGraph,
    hybrid: HeteroGraph,
    original_to_hybrid: np.ndarray,
    *,
    target_node_type: int,
) -> dict[str, Any]:
    mapping = np.asarray(original_to_hybrid, dtype=np.int64).reshape(-1)
    target_nodes = nodes_of_type(original, int(target_node_type))
    if mapping.shape != (original.num_nodes,):
        return {"target_preserve_check": "failed", "reason": "mapping_shape_mismatch"}
    mapped = mapping[target_nodes]
    valid = bool(np.all((mapped >= 0) & (mapped < hybrid.num_nodes)))
    same_type = bool(valid and np.all(hybrid.node_type[mapped] == int(target_node_type)))
    one_to_one = bool(len(np.unique(mapped)) == len(mapped))
    return {
        "target_preserve_check": "passed" if valid and same_type and one_to_one else "failed",
        "mapped_target_valid": valid,
        "mapped_target_same_type": same_type,
        "mapped_target_one_to_one": one_to_one,
        "mapped_target_count": int(len(mapped)),
    }


def evaluate_full_target_inference(
    *,
    original: HeteroGraph,
    hybrid: HeteroGraph,
    original_to_hybrid: np.ndarray,
    target_node_type: int,
    model_name: str,
    seed: int = 12345,
    epochs: int = 80,
    hidden_dim: int = 64,
    device: str = "auto",
    **kwargs: Any,
) -> TaskEvalResult:
    model = str(model_name).lower().replace("-", "_")
    common = {
        "seed": int(seed),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "device": str(device),
        "target_node_type": int(target_node_type),
        **kwargs,
    }
    if model == "sehgnn_lite":
        metrics = evaluate_sehgnn_task(original, hybrid, original_to_hybrid, **common).metrics
    elif model == "hettree_lite":
        metrics = evaluate_hettree_task(original, hybrid, original_to_hybrid, **common).metrics
    else:
        raise ValueError(f"unsupported full-target model: {model_name}")

    adapter = validate_target_preserve_adapter(
        original,
        hybrid,
        original_to_hybrid,
        target_node_type=int(target_node_type),
    )
    fidelity = fidelity_record(model)
    real_row = make_protocol_row(
        metrics,
        eval_mode="real_full_target_inference",
        model_name=model,
        model_fidelity=fidelity["model_fidelity"],
        official_repo=fidelity["official_repo"],
        official_preprocess=fidelity["official_preprocess"],
        adapter_mode="target_preserve_direct",
        path_set=fidelity["path_set"],
        split_policy=str(metrics.get("task_split_policy", fidelity["split_policy"])),
        max_hops=metrics.get("max_hops", fidelity["max_hops"]),
    )
    mode_b_macro = real_row["macro_f1"]
    mode_b_micro = real_row["micro_f1"]
    mode_b_accuracy = real_row["accuracy"]
    metrics.update(
        {
            **adapter,
            "model_name": model,
            "task_eval_protocol": "compressed_support_train_full_target_inference",
            "task_eval_mode": "real_full_target_inference",
            "eval_mode": "real_full_target_inference",
            "official_repo": fidelity["official_repo"],
            "official_preprocess": fidelity["official_preprocess"],
            "adapter_mode": "target_preserve_direct",
            "path_set": fidelity["path_set"],
            "target_domain": real_row["target_domain"],
            "support_domain": real_row["support_domain"],
            "inference_domain": real_row["inference_domain"],
            "metric_source": real_row["metric_source"],
            "model_fidelity": fidelity["model_fidelity"],
            "full_target_inference": True,
            "mode_b_original_macro_f1": mode_b_macro,
            "mode_b_original_micro_f1": mode_b_micro,
            "mode_b_original_accuracy": mode_b_accuracy,
            "primary_task_metric_name": "mode_b_original_macro_f1",
            "primary_task_metric": mode_b_macro,
            "macro_f1": mode_b_macro,
            "micro_f1": mode_b_micro,
            "accuracy": mode_b_accuracy,
        }
    )
    return TaskEvalResult(metrics)
