from __future__ import annotations

import numpy as np


def compute_simhash_buckets(
    Z: np.ndarray,
    node_type: np.ndarray,
    partition_id: np.ndarray | None,
    bits: int,
    seed: int,
) -> np.ndarray:
    if bits <= 0 or bits > 24:
        raise ValueError("bits must be in [1, 24] for packed int64 buckets")
    Z = np.asarray(Z, dtype=np.float32)
    node_type = np.asarray(node_type, dtype=np.int64)
    if partition_id is None:
        partition = np.zeros(Z.shape[0], dtype=np.int64)
    else:
        partition = np.asarray(partition_id, dtype=np.int64)
    rng = np.random.default_rng(seed)
    planes = rng.normal(size=(Z.shape[1], bits)).astype(np.float32)
    hash_bits = np.zeros(Z.shape[0], dtype=np.int64)
    # Avoid BLAS-backed matmul here. On Windows, importing torch before NumPy
    # MKL matmul can initialize duplicate OpenMP runtimes and abort the process.
    chunk_size = max(1, min(Z.shape[0], 65_536))
    for start in range(0, Z.shape[0], chunk_size):
        stop = min(start + chunk_size, Z.shape[0])
        signs = np.einsum("nd,db->nb", Z[start:stop], planes, optimize=False) >= 0
        for bit in range(bits):
            hash_bits[start:stop] |= signs[:, bit].astype(np.int64) << bit
    return (node_type << 48) | (partition << bits) | hash_bits
