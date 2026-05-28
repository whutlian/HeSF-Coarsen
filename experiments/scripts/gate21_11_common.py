from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("outputs/gate21_11_icde_submission_lockdown")
DEFAULT_GATE21_10_ROOT = Path("results/gate21_10_paper_ready")
DEFAULT_GATE21_9_ROOT = Path("outputs/gate21_9_icde_evidence")

SUMMARY_FILES = (
    "gate21_11_decision.json",
    "gate21_11_decision.md",
    "gate21_11_official_main_by_method.csv",
    "gate21_11_budgeted_selector_by_method.csv",
    "gate21_11_channel_planner_trace.csv",
    "gate21_11_external_tp_5x5_runs.csv",
    "gate21_11_external_tp_by_method.csv",
    "gate21_11_external_tp_budget_audit.csv",
    "gate21_11_freehgc_standard_runs.csv",
    "gate21_11_freehgc_standard_by_method.csv",
    "gate21_11_freehgc_tp_adapter_audit.csv",
    "gate21_11_freehgc_env_audit.csv",
    "gate21_11_metapath_tensor_dump.csv",
    "gate21_11_cache_hash_assertions.csv",
    "gate21_11_feature_ablation_task_runs.csv",
    "gate21_11_feature_ablation_by_method.csv",
    "gate21_11_adapter_package_audit.csv",
    "gate21_11_adapter_by_method.csv",
    "gate21_11_system_cost_runs.csv",
    "gate21_11_system_cost_by_method.csv",
    "gate21_11_cross_dataset_task_runs.csv",
    "gate21_11_cross_dataset_by_method.csv",
    "gate21_11_coverage_semantic_diagnostics.csv",
    "gate21_11_apv16_deterministic_proof.json",
)

PROTOCOL_FIELDS = (
    "schema_compatible",
    "keeps_all_target_nodes",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "uses_feature_adapter",
    "uses_weighted_superedges",
    "uses_synthetic_nodes",
    "uses_patched_loader",
    "uses_patched_model",
    "eligible_for_official_main_table",
    "eligible_for_adapter_table",
    "eligible_for_standard_condensation_table",
    "eligible_for_tp_workload_table",
    "eligible_for_decision",
)

COMPRESSION_FIELDS = (
    "support_node_ratio",
    "support_edge_ratio",
    "structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "official_text_hgb_byte_ratio",
    "static_inference_package_ratio",
    "transform_recipe_package_ratio",
    "reconstructable_package_ratio",
    "preprocessed_cache_byte_ratio",
)


def ensure_layout(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {"root": root}
    for name in (
        "official_main",
        "budgeted_selector",
        "external_tp",
        "freehgc",
        "metapath_cache",
        "feature_ablation",
        "adapter",
        "system_cost",
        "cross_dataset",
        "coverage",
        "audits",
        "summary",
    ):
        paths[name] = root / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return dict(default or {})
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default or {})


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    write_csv(Path(path), list(rows), fieldnames=fieldnames)


def write_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(Path(path), payload)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def float_value(value: Any) -> float | None:
    if value in {"", None, "NaN", "nan"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    vals = [float_value(row.get(field)) for row in rows]
    finite = [val for val in vals if val is not None]
    return "NaN" if not finite else sum(finite) / len(finite)


def add_protocol_fields(row: Mapping[str, Any], *, table: str = "") -> dict[str, Any]:
    out = dict(row)
    out.setdefault("dataset", "DBLP")
    out.setdefault("protocol", "schema_preserving_tp")
    out.setdefault("schema_compatible", True)
    out.setdefault("keeps_all_target_nodes", True)
    out.setdefault("official_hgb_exported", False)
    out.setdefault("official_sehgnn_unmodified", False)
    out.setdefault("uses_feature_adapter", False)
    out.setdefault("uses_weighted_superedges", False)
    out.setdefault("uses_synthetic_nodes", False)
    out.setdefault("uses_patched_loader", False)
    out.setdefault("uses_patched_model", False)
    out.setdefault("eligible_for_official_main_table", table in {"official_main", "budgeted_selector"})
    out.setdefault("eligible_for_adapter_table", table == "adapter")
    out.setdefault("eligible_for_standard_condensation_table", table == "freehgc_standard")
    out.setdefault("eligible_for_tp_workload_table", table in {"official_main", "external_tp"})
    out.setdefault("eligible_for_decision", True)
    for field in COMPRESSION_FIELDS:
        out.setdefault(field, "")
    return out


def parse_bool_arg(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

