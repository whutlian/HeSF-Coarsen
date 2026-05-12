from __future__ import annotations

import importlib.util
from math import ceil
from typing import Any

import numpy as np

from hesf_coarsen.progress import progress_iter


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch() -> Any:
    if not torch_available():
        raise RuntimeError("PyTorch is not installed")
    import torch

    return torch


def get_torch_device(preferred: str = "auto", max_fraction: float = 0.5) -> str:
    """Select a Torch device without moving graph structure to GPU."""

    torch = _torch()
    preferred = str(preferred)
    if preferred == "cpu":
        return "cpu"
    if preferred not in {"auto", "cuda"}:
        raise ValueError("preferred must be one of: auto, cpu, cuda")
    if torch.cuda.is_available():
        if max_fraction <= 0.0 or max_fraction > 1.0:
            raise ValueError("max_fraction must be in (0, 1]")
        return "cuda"
    if preferred == "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return "cpu"


def _check_allocation(array: np.ndarray, max_bytes: int | None) -> None:
    if max_bytes is not None and array.nbytes > max_bytes:
        raise MemoryError(
            f"dense Torch block needs {array.nbytes} bytes, exceeds configured "
            f"limit {max_bytes}"
        )


def _check_bytes(num_bytes: int, max_bytes: int | None) -> None:
    if max_bytes is not None and num_bytes > max_bytes:
        raise MemoryError(
            f"dense Torch block needs {num_bytes} bytes, exceeds configured "
            f"limit {max_bytes}"
        )


def torch_row_normalize(
    X: np.ndarray,
    device: str = "auto",
    eps: float = 1e-6,
    max_bytes: int | None = None,
) -> np.ndarray:
    torch = _torch()
    X = np.asarray(X, dtype=np.float32)
    _check_allocation(X, max_bytes)
    selected = get_torch_device(device) if device == "auto" else device
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=selected)
        norms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
        normalized = tensor / torch.clamp(norms, min=eps)
        return normalized.cpu().numpy()


def torch_pairwise_squared_distance(
    X: np.ndarray,
    pairs: np.ndarray,
    device: str = "auto",
    batch_size: int = 65_536,
    max_bytes: int | None = None,
) -> np.ndarray:
    torch = _torch()
    X = np.asarray(X, dtype=np.float32)
    pairs = np.asarray(pairs, dtype=np.int64)
    _check_allocation(X, max_bytes)
    selected = get_torch_device(device) if device == "auto" else device
    out = np.empty(pairs.shape[0], dtype=np.float32)
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32, device=selected)
        for start in range(0, pairs.shape[0], batch_size):
            stop = min(start + batch_size, pairs.shape[0])
            batch_pairs = torch.as_tensor(pairs[start:stop], dtype=torch.long, device=selected)
            diff = tensor[batch_pairs[:, 0]] - tensor[batch_pairs[:, 1]]
            out[start:stop] = torch.sum(diff * diff, dim=1).cpu().numpy()
    return out


def torch_weighted_pairwise_dense_cost(
    blocks: list[tuple[np.ndarray, float]] | tuple[tuple[np.ndarray, float], ...],
    pairs: np.ndarray,
    device: str = "auto",
    batch_size: int = 65_536,
    max_bytes: int | None = None,
    progress_config: dict | None = None,
    progress_desc: str = "torch score dense batches",
) -> np.ndarray:
    """Compute weighted pairwise squared distances from row-local dense batches.

    Unlike ``torch_pairwise_squared_distance``, this helper does not move a full
    dense matrix to Torch. Each candidate batch copies only the unique rows
    touched by that batch, which is the path intended for large candidate sets.
    """

    torch = _torch()
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_pairs, 2]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    prepared: list[tuple[np.ndarray, float]] = []
    for X, weight in blocks:
        weight = float(weight)
        if weight == 0.0:
            continue
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("dense scoring blocks must be 2D")
        prepared.append((array, weight))

    out = np.zeros(pairs.shape[0], dtype=np.float32)
    if not prepared or len(pairs) == 0:
        return out

    selected = get_torch_device(device) if device == "auto" else device
    with torch.no_grad():
        starts = range(0, pairs.shape[0], batch_size)
        for start in progress_iter(
            starts,
            total=ceil(pairs.shape[0] / batch_size) if len(pairs) else 0,
            desc=progress_desc,
            config=progress_config,
            unit="batch",
        ):
            stop = min(start + batch_size, pairs.shape[0])
            batch_pairs = pairs[start:stop]
            unique_nodes, inverse = np.unique(batch_pairs.reshape(-1), return_inverse=True)
            local_pairs = inverse.reshape(-1, 2)
            block_bytes = sum(
                int(len(unique_nodes) * array.shape[1] * np.dtype(np.float32).itemsize)
                for array, _weight in prepared
            )
            _check_bytes(block_bytes, max_bytes)
            local_pairs_tensor = torch.as_tensor(local_pairs, dtype=torch.long, device=selected)
            batch_cost = torch.zeros(stop - start, dtype=torch.float32, device=selected)
            for array, weight in prepared:
                block = np.asarray(array[unique_nodes], dtype=np.float32)
                tensor = torch.as_tensor(block, dtype=torch.float32, device=selected)
                diff = tensor[local_pairs_tensor[:, 0]] - tensor[local_pairs_tensor[:, 1]]
                batch_cost += float(weight) * torch.sum(diff * diff, dim=1)
            out[start:stop] = batch_cost.cpu().numpy()
    return out
