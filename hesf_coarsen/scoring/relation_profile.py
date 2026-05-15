from __future__ import annotations

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph


def _normalize_mode(mode: str | None) -> str:
    normalized = str(mode or "relationwise").lower().replace("-", "_")
    aliases = {
        "relation_aware": "relationwise",
        "relationwise": "relationwise",
        "per_relation": "relationwise",
        "single": "single_relation_sum",
        "flatten": "single_relation_sum",
        "flatten_sum": "single_relation_sum",
        "single_relation": "single_relation_sum",
        "single_relation_sum": "single_relation_sum",
        "disabled": "disabled",
        "none": "disabled",
        "off": "disabled",
    }
    if normalized not in aliases:
        raise ValueError(
            "scoring.relation_profile_mode must be one of: "
            "relationwise, single_relation_sum, disabled"
        )
    return aliases[normalized]


def compute_relation_profiles(graph: HeteroGraph, mode: str | None = "relationwise") -> np.ndarray:
    mode = _normalize_mode(mode)
    if mode == "disabled":
        return np.zeros((graph.num_nodes, 0), dtype=np.float32)
    if mode == "single_relation_sum":
        profile = np.zeros((graph.num_nodes, 2), dtype=np.float32)
        for rel in graph.relations.values():
            np.add.at(profile[:, 0], rel.src, rel.weight)
            np.add.at(profile[:, 1], rel.dst, rel.weight)
        denom = profile.sum(axis=1, keepdims=True)
        return profile / np.maximum(denom, 1e-6)

    relation_ids = sorted(graph.relations)
    profile = np.zeros((graph.num_nodes, 2 * len(relation_ids)), dtype=np.float32)
    for idx, relation_id in enumerate(relation_ids):
        rel = graph.relations[relation_id]
        np.add.at(profile[:, 2 * idx], rel.src, rel.weight)
        np.add.at(profile[:, 2 * idx + 1], rel.dst, rel.weight)
    denom = profile.sum(axis=1, keepdims=True)
    return profile / np.maximum(denom, 1e-6)
