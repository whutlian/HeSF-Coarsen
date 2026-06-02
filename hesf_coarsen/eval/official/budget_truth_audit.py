from __future__ import annotations

from typing import Any, Iterable, Mapping

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


BUDGET_MATCH_FIELDS = {
    "structural_storage_ratio": "semantic_structural_storage_ratio",
    "support_edge_ratio": "actual_support_edge_ratio",
    "raw_hgb_text_byte_ratio": "raw_hgb_text_byte_ratio",
    "support_node_ratio": "actual_support_node_ratio",
    "keyword_feature_ratio": "keyword_feature_ratio",
    "PK_edge_ratio": "PK_edge_ratio",
    "channel_edge_ratio": "channel_edge_ratio",
}


EXPLICIT_BUDGET_FIELDS = (
    "requested_budget_type",
    "requested_budget",
    "actual_edge_ratio",
    "actual_support_edge_ratio",
    "actual_support_node_ratio",
    "semantic_structural_storage_ratio",
    "raw_hgb_text_byte_ratio",
    "static_inference_package_ratio",
    "reconstructable_package_ratio",
    "budget_match_for_requested_metric",
    "budget_metric_used_for_match",
    "budget_match_failure_type",
    "budget_match_failure_reason",
)


def annotate_budget_truth(row: Mapping[str, Any], *, tolerance: float = 0.03) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = normalize_dataset(out.get("dataset"))
    out["actual_support_edge_ratio"] = _first_numeric(
        out.get("actual_support_edge_ratio"),
        out.get("support_edge_ratio"),
        out.get("actual_edge_ratio"),
    )
    out["actual_edge_ratio"] = _first_numeric(out.get("actual_edge_ratio"), out.get("actual_support_edge_ratio"))
    out["actual_support_node_ratio"] = _first_numeric(out.get("actual_support_node_ratio"), out.get("support_node_ratio"))
    out["semantic_structural_storage_ratio"] = _first_numeric(
        out.get("semantic_structural_storage_ratio"),
        out.get("semantic_structural_ratio"),
    )
    out["raw_hgb_text_byte_ratio"] = _first_numeric(out.get("raw_hgb_text_byte_ratio"), out.get("raw_hgb_text_ratio"))
    out["static_inference_package_ratio"] = _first_numeric(out.get("static_inference_package_ratio"), out.get("raw_hgb_text_byte_ratio"))
    out["reconstructable_package_ratio"] = _first_numeric(out.get("reconstructable_package_ratio"), out.get("raw_hgb_text_byte_ratio"))
    if float_value(out.get("channel_edge_ratio")) is None:
        channel_values = [float_value(out.get("actor_channel_ratio")), float_value(out.get("keyword_channel_ratio"))]
        finite_channels = [value for value in channel_values if value is not None]
        out["channel_edge_ratio"] = max(finite_channels) if finite_channels else ""

    budget_type = str(out.get("requested_budget_type", "")).strip()
    requested = float_value(out.get("requested_budget"))
    metric_field = BUDGET_MATCH_FIELDS.get(budget_type, "")
    out["budget_metric_used_for_match"] = metric_field

    if not budget_type:
        out["budget_match_for_requested_metric"] = False
        out["budget_match_failure_type"] = "missing_requested_budget_type"
        out["budget_match_failure_reason"] = "requested_budget_type is missing."
        return out
    if requested is None:
        out["budget_match_for_requested_metric"] = False
        out["budget_match_failure_type"] = "missing_requested_budget"
        out["budget_match_failure_reason"] = "requested_budget is missing or non-numeric."
        return out
    if not metric_field:
        out["budget_match_for_requested_metric"] = False
        out["budget_match_failure_type"] = "unsupported_budget_type"
        out["budget_match_failure_reason"] = f"No explicit metric is defined for requested_budget_type={budget_type}."
        return out

    actual = float_value(out.get(metric_field))
    if actual is None:
        out["budget_match_for_requested_metric"] = False
        out["budget_match_failure_type"] = "metric_missing"
        out["budget_match_failure_reason"] = f"{metric_field} is required for requested_budget_type={budget_type}."
        return out

    match = actual <= requested + float(tolerance)
    out["budget_match_for_requested_metric"] = bool(match)
    if match:
        out["budget_match_failure_type"] = ""
        out["budget_match_failure_reason"] = ""
    else:
        out["budget_match_failure_type"] = "budget_mismatch"
        out["budget_match_failure_reason"] = (
            f"{metric_field}={actual:.12g} exceeds requested_budget={requested:.12g} "
            f"with tolerance={float(tolerance):.12g}."
        )
    return out


def build_budget_truth_audit(rows: Iterable[Mapping[str, Any]], *, tolerance: float = 0.03) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        annotated = annotate_budget_truth(row, tolerance=tolerance)
        audit_rows.append(
            {
                "dataset": annotated.get("dataset", ""),
                "method": annotated.get("method", ""),
                "requested_budget_type": annotated.get("requested_budget_type", ""),
                "requested_budget": annotated.get("requested_budget", ""),
                "actual_support_edge_ratio": annotated.get("actual_support_edge_ratio", ""),
                "semantic_structural_storage_ratio": annotated.get("semantic_structural_storage_ratio", ""),
                "raw_hgb_text_byte_ratio": annotated.get("raw_hgb_text_byte_ratio", ""),
                "budget_match": annotated.get("budget_match_for_requested_metric", ""),
                "budget_metric_used_for_match": annotated.get("budget_metric_used_for_match", ""),
                "why_raw_bytes_differ_from_edge_budget": _raw_byte_explanation(annotated),
                "budget_match_failure_type": annotated.get("budget_match_failure_type", ""),
                "budget_match_failure_reason": annotated.get("budget_match_failure_reason", ""),
            }
        )
    return audit_rows


def full_fallback_exclusion(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if bool_value(out.get("constraint_safe_fallback")):
        out["eligible_for_main_table"] = False
        out["eligible_for_compression_claim"] = False
        out["success"] = False
        out["failure_type"] = "constraint_safe_full_fallback"
        out["failure_reason"] = (
            out.get("failure_reason")
            or "Full-HGB fallback is loader sanity evidence only and is excluded from Gate21.18 compression results."
        )
    return out


def _raw_byte_explanation(row: Mapping[str, Any]) -> str:
    budget_type = str(row.get("requested_budget_type", ""))
    raw = float_value(row.get("raw_hgb_text_byte_ratio"))
    edge = float_value(row.get("actual_support_edge_ratio"))
    if budget_type == "support_edge_ratio" and raw is not None and edge is not None and raw > edge + 0.10:
        return "node.dat/features dominate raw HGB text bytes; edge budget is not a raw-byte deployment budget."
    if budget_type == "structural_storage_ratio":
        return "structural budget is matched against semantic_structural_storage_ratio, not raw HGB text bytes."
    return ""


def _first_numeric(*values: Any) -> Any:
    for value in values:
        if float_value(value) is not None:
            return value
    return "" if not values else values[0] if values[0] not in (None, "None") else ""
