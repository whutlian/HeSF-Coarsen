from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def storage_denominator_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    method_bytes = _float(out.get("method_artifact_bytes", out.get("artifact_bytes", out.get("disk_bytes"))))
    original = _float(out.get("original_native_full_hgb_text_bytes"))
    export = _float(out.get("export_full_hgb_text_bytes", out.get("current_export_full_text_bytes")))
    control = _float(out.get("current_control_artifact_bytes", out.get("current_compressed_control_text_bytes")))
    out.pop("ratio_vs_native_full_text", None)
    out["method_artifact_bytes"] = "" if method_bytes is None else method_bytes
    out["ratio_vs_original_native_full_hgb_text"] = _ratio(method_bytes, original)
    out["ratio_vs_export_full_hgb_text"] = _ratio(method_bytes, export)
    out["ratio_vs_current_control_artifact"] = _ratio(method_bytes, control)
    out["ratio_field_name_consistent"] = all(out[field] != "" for field in ("ratio_vs_original_native_full_hgb_text", "ratio_vs_export_full_hgb_text", "ratio_vs_current_control_artifact"))
    out["matches_official_main_denominator"] = bool(original and export and original >= export)
    return out


def storage_denominator_audit_many(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [storage_denominator_audit(row) for row in rows]


def _ratio(numerator: float | None, denominator: float | None) -> float | str:
    if numerator is None or denominator is None or denominator <= 0:
        return ""
    return numerator / denominator


def _float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
