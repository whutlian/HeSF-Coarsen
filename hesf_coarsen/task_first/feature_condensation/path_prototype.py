from __future__ import annotations

import numpy as np

from hesf_coarsen.task_first.feature_condensation.semantic_tree_cache import SemanticTreeCache


def class_path_prototype_cache(
    cache: SemanticTreeCache,
    *,
    labels: np.ndarray,
    train_nodes: np.ndarray,
) -> SemanticTreeCache:
    tensor = np.asarray(cache.tensor, dtype=np.float32)
    if tensor.size == 0:
        return cache
    target_lookup = {int(node): idx for idx, node in enumerate(np.asarray(cache.target_nodes, dtype=np.int64).tolist())}
    train_local = [target_lookup[int(node)] for node in np.asarray(train_nodes, dtype=np.int64).tolist() if int(node) in target_lookup]
    if not train_local:
        return cache
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = tensor.copy()
    prototypes: list[np.ndarray] = []
    for cls in sorted(int(value) for value in np.unique(labels_arr[labels_arr >= 0])):
        cls_local = [idx for idx in train_local if int(labels_arr[int(cache.target_nodes[idx])]) == cls]
        if not cls_local:
            continue
        proto = np.mean(tensor[np.asarray(cls_local, dtype=np.int64)], axis=0, keepdims=True)
        prototypes.append(proto.reshape(tensor.shape[1], tensor.shape[2]).astype(np.float32, copy=False))
    if prototypes:
        proto_arr = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
        flat_proto = proto_arr.reshape((len(prototypes), -1))
        flat_tensor = tensor.reshape((tensor.shape[0], -1))
        distances = np.sum((flat_tensor[:, None, :] - flat_proto[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1).astype(np.int64, copy=False)
        out = proto_arr[nearest]
    return SemanticTreeCache(
        tensor=out.astype(np.float32, copy=False),
        target_nodes=np.asarray(cache.target_nodes, dtype=np.int64),
        paths=list(cache.paths),
        feature_width=int(cache.feature_width),
        type_ids=tuple(cache.type_ids),
    )
