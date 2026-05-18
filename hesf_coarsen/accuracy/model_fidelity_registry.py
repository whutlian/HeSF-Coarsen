from __future__ import annotations

from typing import Any, Mapping


REQUIRED_FIDELITY_FIELDS = (
    "model_name",
    "model_fidelity",
    "official_repo",
    "official_preprocess",
    "adapter_mode",
    "split_policy",
    "path_set",
    "max_hops",
)


def fidelity_record(model_name: str) -> dict[str, Any]:
    key = str(model_name).lower().replace("-", "_")
    if key in {"sehgnn_lite", "local_sehgnn"}:
        return {
            "model_name": "sehgnn_lite",
            "model_fidelity": "lite_adapter",
            "repository": "ICT-GIMLab/SeHGNN architecture-inspired local adapter",
            "official_repo": "no",
            "official_preprocess": "no",
            "adapter_mode": "lite",
            "split_policy": "synthetic_stratified",
            "path_set": "lite",
            "max_hops": 2,
        }
    if key in {"hettree_lite", "local_hettree"}:
        return {
            "model_name": "hettree_lite",
            "model_fidelity": "lite_adapter",
            "repository": "HETTREE semantic-tree-inspired local adapter",
            "official_repo": "no",
            "official_preprocess": "no",
            "adapter_mode": "lite",
            "split_policy": "synthetic_stratified",
            "path_set": "lite",
            "max_hops": 2,
        }
    if key in {"official_sehgnn", "sehgnn_official"}:
        return {
            "model_name": "official_sehgnn",
            "model_fidelity": "official_not_integrated",
            "repository": "https://github.com/ICT-GIMLab/SeHGNN",
            "official_repo": "available_not_integrated",
            "official_preprocess": "no",
            "adapter_mode": "not_integrated",
            "split_policy": "unknown_official",
            "path_set": "official_expected",
            "max_hops": "",
        }
    if key in {"official_hettree", "hettree_official"}:
        return {
            "model_name": "official_hettree",
            "model_fidelity": "unavailable",
            "repository": "https://github.com/microsoft/HetTree",
            "official_repo": "unavailable",
            "official_preprocess": "no",
            "adapter_mode": "not_integrated",
            "split_policy": "unknown_official",
            "path_set": "official_expected",
            "max_hops": "",
        }
    if key in {"freehgc", "official_freehgc"}:
        return {
            "model_name": "freehgc",
            "model_fidelity": "unavailable",
            "repository": "https://github.com/PKU-DAIR/FreeHGC",
            "official_repo": "not_vendored",
            "official_preprocess": "no",
            "adapter_mode": "not_integrated",
            "split_policy": "paper_protocol_unmatched",
            "path_set": "not_applicable",
            "max_hops": "",
        }
    raise ValueError(f"unknown model fidelity key: {model_name}")


def validate_fidelity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIDELITY_FIELDS if row.get(field, "") == ""]
    return {"ok": not missing, "missing_fields": missing}


def all_fidelity_records() -> list[dict[str, Any]]:
    return [
        fidelity_record("sehgnn_lite"),
        fidelity_record("hettree_lite"),
        fidelity_record("official_sehgnn"),
        fidelity_record("official_hettree"),
        fidelity_record("freehgc"),
    ]
