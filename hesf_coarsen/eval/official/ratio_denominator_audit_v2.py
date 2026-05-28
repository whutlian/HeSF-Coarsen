from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def ratio_denominator_audit_v2(row: Mapping[str, Any], *, tolerance: float = 1e-6) -> dict[str, Any]:
    out = dict(row)
    artifact_bytes = _float(out.get("artifact_bytes", out.get("static_inference_package_bytes", out.get("total_artifact_bytes", out.get("disk_bytes")))))
    original = _float(out.get("original_native_full_hgb_text_bytes", out.get("raw_hgb_text_bytes", out.get("native_full_text_bytes"))))
    export_full = _float(out.get("current_export_full_text_bytes", out.get("export_full_text_bytes")))
    compressed_control = _float(out.get("current_compressed_control_text_bytes", out.get("current_control_text_bytes", out.get("zstd_bytes", out.get("gzip_bytes")))))

    out.pop("ratio_vs_native_full_text", None)
    out["artifact_bytes"] = "" if artifact_bytes is None else artifact_bytes
    out["original_native_full_hgb_text_bytes"] = "" if original is None else original
    out["current_export_full_text_bytes"] = "" if export_full is None else export_full
    out["current_compressed_control_text_bytes"] = "" if compressed_control is None else compressed_control

    errors: list[str] = []
    ratios = {
        "ratio_vs_original_native_full_hgb_text": _ratio(artifact_bytes, original),
        "ratio_vs_current_export_full_text": _ratio(artifact_bytes, export_full),
        "ratio_vs_current_compressed_control_text": _ratio(artifact_bytes, compressed_control),
    }
    for field, value in ratios.items():
        out[field] = "" if value is None else value
        provided = _float(row.get(field))
        if provided is not None and value is not None and abs(provided - value) > tolerance:
            errors.append(f"{field}_mismatch")
    for field, value in (
        ("artifact_bytes", artifact_bytes),
        ("original_native_full_hgb_text_bytes", original),
        ("current_export_full_text_bytes", export_full),
        ("current_compressed_control_text_bytes", compressed_control),
    ):
        if value is None or value <= 0:
            errors.append(f"{field}_missing_or_nonpositive")
    out["ratio_denominator_audit_v2_pass"] = not errors
    out["ratio_denominator_audit_v2_errors"] = ";".join(errors)
    return out


def audit_ratio_denominators_v2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [ratio_denominator_audit_v2(row) for row in rows]


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
