"""HeSF-Coarsen research prototype."""

from hesf_coarsen.config import DEFAULT_CONFIG, load_config
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec

__all__ = [
    "DEFAULT_CONFIG",
    "HeteroGraph",
    "RelationAdj",
    "RelationSpec",
    "load_config",
]
