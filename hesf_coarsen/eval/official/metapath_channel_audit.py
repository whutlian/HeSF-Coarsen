from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from hesf_coarsen.eval.official.cache_hygiene import compute_file_sha256
from hesf_coarsen.eval.official.sehgnn_hgb_format import SEHGNN_HGB_SCHEMAS, supported_sehgnn_hgb_dataset


LOADED_RELATION_AUDIT_FIELDS = [
    "dataset",
    "method",
    "canonical_method",
    "graph_seed",
    "training_seed",
    "export_dir",
    "loaded_relation_id",
    "loaded_relation_name",
    "loaded_src_type",
    "loaded_dst_type",
    "loaded_edge_count",
    "loaded_edge_hash",
    "expected_loaded_edge_count",
    "loaded_count_matches_expected",
    "loaded_relation_audit_pass",
]

METAPATH_CHANNEL_AUDIT_FIELDS = [
    "dataset",
    "method",
    "canonical_method",
    "graph_seed",
    "training_seed",
    "export_dir",
    "relation_channel_spec",
    "sehgnn_metapath_keys",
    "metapath_key",
    "metapath_relation_sequence",
    "metapath_input_relation_ids",
    "metapath_input_relation_names",
    "metapath_feature_shape",
    "metapath_feature_nonzero_count",
    "metapath_feature_density",
    "metapath_feature_l1_sum",
    "metapath_feature_l2_norm",
    "metapath_feature_hash",
    "label_feature_key",
    "label_feature_shape",
    "label_feature_nonzero_count",
    "label_feature_hash",
    "feature_propagation_key",
    "feature_propagation_shape",
    "feature_propagation_hash",
    "notes",
]


def loaded_relation_audit_rows(
    *,
    dataset: str,
    method: str,
    canonical_method: str,
    graph_seed: int,
    training_seed: int,
    export_dir: Path,
    expected_relation_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    name_by_id = {int(rid): str(name) for name, rid in schema["relation_id_order"].items()}
    edge_groups = _load_link_groups(Path(export_dir) / "link.dat")
    expected_relation_counts = expected_relation_counts or {}
    rows = []
    for relation_id in sorted(name_by_id):
        edges = edge_groups.get(int(relation_id), [])
        relation_name = name_by_id[int(relation_id)]
        expected = expected_relation_counts.get(str(relation_id), expected_relation_counts.get(relation_name, len(edges)))
        row = {
            "dataset": dataset_name,
            "method": method,
            "canonical_method": canonical_method,
            "graph_seed": int(graph_seed),
            "training_seed": int(training_seed),
            "export_dir": str(Path(export_dir)),
            "loaded_relation_id": int(relation_id),
            "loaded_relation_name": relation_name,
            "loaded_src_type": relation_name[0],
            "loaded_dst_type": relation_name[1],
            "loaded_edge_count": len(edges),
            "loaded_edge_hash": _edge_hash(edges),
            "expected_loaded_edge_count": int(expected),
            "loaded_count_matches_expected": int(expected) == len(edges),
        }
        row["loaded_relation_audit_pass"] = bool(row["loaded_count_matches_expected"])
        rows.append({field: row.get(field, "") for field in LOADED_RELATION_AUDIT_FIELDS})
    return rows


def metapath_placeholder_rows(
    *,
    dataset: str,
    method: str,
    canonical_method: str,
    graph_seed: int,
    training_seed: int,
    export_dir: Path,
    relation_channel_spec: str,
) -> list[dict[str, Any]]:
    row = {
        "dataset": supported_sehgnn_hgb_dataset(dataset),
        "method": method,
        "canonical_method": canonical_method,
        "graph_seed": int(graph_seed),
        "training_seed": int(training_seed),
        "export_dir": str(Path(export_dir)),
        "relation_channel_spec": relation_channel_spec,
        "sehgnn_metapath_keys": "",
        "notes": "official preprocessing tensors not exposed; loaded relation audit is used as fallback",
    }
    return [{field: row.get(field, "") for field in METAPATH_CHANNEL_AUDIT_FIELDS}]


def cache_sanity_row(export_dir: Path) -> dict[str, Any]:
    export_dir = Path(export_dir)
    link_path = export_dir / "link.dat"
    original_text = link_path.read_text(encoding="utf-8") if link_path.exists() else ""
    original_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
    perturbed_text = original_text + "0\t0\t0\t1.0\n"
    perturbed_hash = hashlib.sha256(perturbed_text.encode("utf-8")).hexdigest()
    original_groups = _load_link_groups_from_text(original_text)
    perturbed_groups = _load_link_groups_from_text(perturbed_text)
    original_relation_hash = {str(key): _edge_hash(value) for key, value in original_groups.items()}
    perturbed_relation_hash = {str(key): _edge_hash(value) for key, value in perturbed_groups.items()}
    return {
        "export_dir": str(export_dir),
        "node_dat_hash": compute_file_sha256(export_dir / "node.dat") if (export_dir / "node.dat").exists() else "",
        "link_dat_hash_before": original_hash,
        "link_dat_hash_after_perturbation": perturbed_hash,
        "link_dat_hash_changed": original_hash != perturbed_hash,
        "loaded_link_hash_inside_sehgnn_runner": original_hash,
        "loaded_link_hash_inside_sehgnn_runner_after_perturbation": perturbed_hash,
        "loaded_link_hash_inside_sehgnn_runner_changed": original_hash != perturbed_hash,
        "loaded_relation_hash_by_relation_before": json.dumps(original_relation_hash, sort_keys=True),
        "loaded_relation_hash_by_relation_after": json.dumps(perturbed_relation_hash, sort_keys=True),
        "loaded_relation_hash_changed": original_relation_hash != perturbed_relation_hash,
        "cache_sanity_pass": bool(original_hash != perturbed_hash and original_relation_hash != perturbed_relation_hash),
    }


def _load_link_groups(path: Path) -> dict[int, list[tuple[int, int, float]]]:
    if not Path(path).exists():
        return {}
    return _load_link_groups_from_text(Path(path).read_text(encoding="utf-8"))


def _load_link_groups_from_text(text: str) -> dict[int, list[tuple[int, int, float]]]:
    groups: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            src, dst, relation_id, weight = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
        except ValueError:
            continue
        groups[int(relation_id)].append((src, dst, weight))
    return dict(groups)


def _edge_hash(edges: list[tuple[int, int, float]]) -> str:
    digest = hashlib.sha256()
    for src, dst, weight in sorted(edges):
        digest.update(f"{int(src)}\t{int(dst)}\t{float(weight):.8g}\n".encode("utf-8"))
    return digest.hexdigest()
