from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("results/gate21_13_full")
DEFAULT_GATE21_12_ROOT = Path("experiments/results/gate21_12_executed_evidence_completion")

COMPONENT_DIRS = (
    "raw_runs",
    "exports",
    "sehgnn_logs",
    "cache_dumps",
    "adapter_packages",
    "external_baselines",
    "freehgc",
    "system_cost",
    "cross_dataset",
    "summary",
    "official_main",
    "budgeted_selector",
    "metapath_cache",
    "feature_ablation",
    "adapter",
    "audits",
)

SUMMARY_FILES = (
    "gate21_13_official_main_by_method.csv",
    "gate21_13_budgeted_selector_by_method.csv",
    "gate21_13_selector_hash_audit.csv",
    "gate21_13_deterministic_selector_proof.csv",
    "gate21_13_external_tp_runs.csv",
    "gate21_13_external_tp_by_method_budget.csv",
    "gate21_13_external_tp_budget_fairness.csv",
    "gate21_13_external_tp_failure_report.csv",
    "gate21_13_freehgc_env_audit.csv",
    "gate21_13_freehgc_standard_runs.csv",
    "gate21_13_freehgc_standard_by_ratio.csv",
    "gate21_13_freehgc_tp_adapter_audit.csv",
    "gate21_13_freehgc_tp_runs.csv",
    "gate21_13_freehgc_tp_failure_proof.json",
    "gate21_13_metapath_tensor_dump.csv",
    "gate21_13_cache_hash_assertions.csv",
    "gate21_13_metapath_key_diff.csv",
    "gate21_13_feature_ablation_runs.csv",
    "gate21_13_feature_ablation_by_method.csv",
    "gate21_13_feature_ablation_shape_assertions.csv",
    "gate21_13_adapter_runs.csv",
    "gate21_13_adapter_by_method.csv",
    "gate21_13_adapter_package_audit.csv",
    "gate21_13_system_cost_runs.csv",
    "gate21_13_system_cost_by_method.csv",
    "gate21_13_cross_dataset_runs.csv",
    "gate21_13_cross_dataset_by_method.csv",
    "gate21_13_cross_dataset_selector_plans.csv",
    "gate21_13_selector_modes.csv",
    "gate21_13_selector_pareto_frontier.csv",
    "gate21_13_by_method.csv",
    "gate21_13_failure_audit.csv",
    "gate21_13_decision.json",
    "gate21_13_decision.md",
    "gate21_13_icde_evidence_manifest.json",
    "gate21_13_requirement_checklist.md",
    "gate21_13_prompt_completion_checklist.md",
    "gate21_13_manifest.json",
)

PROTOCOL_FIELDS = (
    "method_family",
    "protocol_family",
    "schema_compatible",
    "target_preserving",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "uses_adapter_loader",
    "uses_synthetic_nodes",
    "uses_weighted_superedges",
    "eligible_for_official_main_table",
    "eligible_for_adapter_table",
    "eligible_for_standard_condensation_table",
    "eligible_for_tp_workload_table",
    "eligible_for_decision",
    "diagnostic_only",
)

COMPRESSION_FIELDS = (
    "support_node_ratio",
    "support_edge_ratio",
    "structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "static_inference_package_ratio",
    "transform_recipe_package_ratio",
    "reconstructable_package_ratio",
)


def ensure_layout(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {"root": root}
    for name in COMPONENT_DIRS:
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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ready"}


def float_value(value: Any) -> float | None:
    if value in {"", None, "NaN", "nan"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def finite_metric(row: Mapping[str, Any], *names: str) -> bool:
    return all(float_value(row.get(name)) is not None for name in names)


def mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [value for value in (float_value(row.get(field)) for row in rows) if value is not None]
    return "NaN" if not values else sum(values) / len(values)


def std_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [value for value in (float_value(row.get(field)) for row in rows) if value is not None]
    if not values:
        return "NaN"
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def rate_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    if not rows:
        return "NaN"
    return sum(1 for row in rows if bool_value(row.get(field))) / len(rows)


def add_gate21_13_protocol_fields(row: Mapping[str, Any], *, family: str, protocol: str, diagnostic_only: bool = False) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("dataset", "DBLP")
    out.setdefault("method_family", family)
    out.setdefault("protocol_family", protocol)
    out.setdefault("schema_compatible", True)
    out.setdefault("target_preserving", out.get("keeps_all_target_nodes", True))
    out.setdefault("official_hgb_exported", False)
    out.setdefault("official_sehgnn_unmodified", False)
    out.setdefault("uses_adapter_loader", bool_value(out.get("uses_feature_adapter")) or bool_value(out.get("uses_patched_loader")))
    out.setdefault("uses_synthetic_nodes", False)
    out.setdefault("uses_weighted_superedges", False)
    out.setdefault("eligible_for_official_main_table", False)
    out.setdefault("eligible_for_adapter_table", family == "feature_adapter")
    out.setdefault("eligible_for_standard_condensation_table", family == "standard_condensation")
    out.setdefault("eligible_for_tp_workload_table", family in {"official_main", "external_tp_baseline"})
    out.setdefault("eligible_for_decision", not diagnostic_only)
    out.setdefault("diagnostic_only", diagnostic_only)
    out.setdefault("uses_test_metrics_for_selection", False)
    out.setdefault("uses_test_labels_for_selection", False)
    out.setdefault("selection_signal_source", out.get("validation_probe_source", "not_applicable"))
    for field in COMPRESSION_FIELDS:
        out.setdefault(field, out.get(f"actual_{field}", ""))
    return out


def normalize_failure(row: Mapping[str, Any], *, default_type: str, default_reason: str) -> dict[str, Any]:
    out = dict(row)
    if not bool_value(out.get("success", out.get("training_executed", False))):
        out.setdefault("success", False)
        out.setdefault("training_executed", False)
        out.setdefault("test_micro_f1", "NaN")
        out.setdefault("test_macro_f1", "NaN")
        out.setdefault("failure_type", default_type)
        out.setdefault("failure_reason", out.get("failure_message", default_reason) or default_reason)
    return out


def parse_bool_arg(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
