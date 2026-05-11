from __future__ import annotations

import numpy as np


def generate_probe(
    num_nodes: int,
    dim: int,
    seed: int,
    probe: str = "rademacher",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if probe == "rademacher":
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(num_nodes, dim))
    if probe == "gaussian":
        return rng.normal(size=(num_nodes, dim)).astype(np.float32)
    raise ValueError(f"unsupported probe type: {probe}")
