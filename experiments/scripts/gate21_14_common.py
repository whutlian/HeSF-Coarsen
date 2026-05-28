from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("outputs/gate21_14_full_execution_push")
DEFAULT_CROSS_OUTPUT_ROOT = Path("outputs/gate21_14_full_execution_push_cross_dataset")
DEFAULT_GATE21_13_ROOT = Path("results/gate21_13_summary")
DEFAULT_GATE21_6_PACKAGE_ROOT = Path("results/gate21_6_chatgpt_packages/gate21_6_main_results_for_chatgpt/gate21_6_icde_ready")
DEFAULT_GATE21_0_ROOT = Path("outputs/gate21_0_sehgnn_native_export")

COMPONENT_DIRS = (
    "official_main",
    "budgeted_selector",
    "external_tp",
    "freehgc",
    "feature_ablation",
    "metapath_cache",
    "coverage",
    "adapter",
    "system_cost",
    "cross_dataset",
    "pareto",
    "audits",
    "logs",
)

SUMMARY_FILES = (
    "gate21_14_decision.json",
    "gate21_14_decision.md",
    "gate21_14_by_method.csv",
    "gate21_14_official_main_by_method.csv",
    "gate21_14_budgeted_selector_by_method.csv",
    "gate21_14_selector_hash_audit.csv",
    "gate21_14_external_tp_by_method.csv",
    "gate21_14_external_tp_runs.csv",
    "gate21_14_external_tp_budget_audit.csv",
    "gate21_14_freehgc_standard_by_method.csv",
    "gate21_14_freehgc_tp_by_method.csv",
    "gate21_14_freehgc_protocol_audit.csv",
    "gate21_14_freehgc_score_selector_by_method.csv",
    "gate21_14_feature_ablation_by_method.csv",
    "gate21_14_feature_ablation_runs.csv",
    "gate21_14_metapath_tensor_dump.csv",
    "gate21_14_cache_hash_assertions.csv",
    "gate21_14_adapter_by_method.csv",
    "gate21_14_adapter_package_audit.csv",
    "gate21_14_system_workload_cost_by_method.csv",
    "gate21_14_system_workload_cost_runs.csv",
    "gate21_14_cross_dataset_by_method.csv",
    "gate21_14_cross_dataset_runs.csv",
    "gate21_14_coverage_semantic_diagnostics.csv",
    "gate21_14_pareto_frontier.csv",
    "gate21_14_requirement_checklist.md",
    "gate21_14_prompt_completion_checklist.md",
    "gate21_14_failure_audit.csv",
    "gate21_14_manifest.json",
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


def task_ready(row: Mapping[str, Any]) -> bool:
    return (
        bool_value(row.get("training_executed"))
        and bool_value(row.get("success", True))
        and finite_metric(row, "test_micro_f1", "test_macro_f1")
        and not str(row.get("failure_type", "")).strip()
    )


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


def stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_size(path: Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return int(sum(item.stat().st_size for item in p.rglob("*") if item.is_file()))


def budget_token(value: Any) -> str:
    parsed = float_value(value)
    return str(value) if parsed is None else f"{parsed:.3f}"


def budget_match(row: Mapping[str, Any], tolerance: float = 0.02) -> bool:
    requested = float_value(row.get("requested_budget"))
    if requested is None:
        return False
    budget_type = str(row.get("budget_type", ""))
    actual_field = "actual_structural_storage_ratio" if "structural" in budget_type else "actual_support_node_ratio"
    actual = float_value(row.get(actual_field))
    return actual is not None and abs(actual - requested) <= tolerance


def parse_bool_arg(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def gate21_14_protocol_fields(row: Mapping[str, Any], *, family: str, protocol: str, diagnostic_only: bool = False) -> dict[str, Any]:
    out = dict(row)
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
    out.setdefault("eligible_for_adapter_table", family == "adapter")
    out.setdefault("eligible_for_external_tp_table", family == "external_tp")
    out.setdefault("eligible_for_standard_condensation_table", family == "freehgc_standard")
    out.setdefault("eligible_for_decision", not diagnostic_only)
    out.setdefault("diagnostic_only", diagnostic_only)
    out.setdefault("uses_test_metrics_for_selection", False)
    out.setdefault("uses_test_labels_for_selection", False)
    out.setdefault("no_test_metric_used_for_selection", not bool_value(out.get("uses_test_metrics_for_selection")))
    out.setdefault("no_test_leakage", not bool_value(out.get("uses_test_metrics_for_selection")) and not bool_value(out.get("uses_test_labels_for_selection")))
    for field in COMPRESSION_FIELDS:
        out.setdefault(field, out.get(f"actual_{field}", ""))
    return out
