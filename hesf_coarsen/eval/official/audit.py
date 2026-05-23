from __future__ import annotations

from typing import Any, Mapping


def audit_export_record(record: Mapping[str, Any]) -> dict[str, Any]:
    passed = (
        str(record.get("export_status", "")) == "success"
        and bool(record.get("mapping_bijective", False))
        and bool(record.get("split_disjoint", False))
        and bool(record.get("no_test_label_export_leakage", False))
        and int(record.get("target_count_original", 0)) == int(record.get("target_count_exported", -1))
    )
    return {**dict(record), "export_audit_pass": bool(passed)}


def hettree_exclusion_record() -> dict[str, str]:
    return {
        "dependency": "HETTREE-official",
        "status": "excluded_code_unavailable",
        "reason": "GitHub code link unavailable/404; Gate21 must not block on HETTREE.",
    }
