from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Assignment:
    assignment: np.ndarray
    supernode_type: np.ndarray

    def __post_init__(self) -> None:
        self.assignment = np.asarray(self.assignment, dtype=np.int64)
        self.supernode_type = np.asarray(self.supernode_type, dtype=np.int32)
        if self.assignment.ndim != 1:
            raise ValueError("assignment must be 1D")
        if self.supernode_type.ndim != 1:
            raise ValueError("supernode_type must be 1D")
        if len(self.assignment) and self.assignment.min() < 0:
            raise ValueError("assignment must map every node")
        if len(self.assignment) and self.assignment.max(initial=-1) >= len(self.supernode_type):
            raise ValueError("assignment references missing supernode")

    @property
    def num_supernodes(self) -> int:
        return int(len(self.supernode_type))

    def cluster_sizes(self) -> np.ndarray:
        return np.bincount(self.assignment, minlength=self.num_supernodes).astype(np.int64)
