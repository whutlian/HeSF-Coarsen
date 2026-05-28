from __future__ import annotations

from typing import Any, Mapping, Sequence

from hesf_coarsen.eval.official.end_to_end_system_cost import summarize_gate21_11_system_cost


def summarize_gate21_12_system_cost(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return summarize_gate21_11_system_cost(rows)


def summarize_gate21_13_system_cost(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return summarize_gate21_11_system_cost(rows)


def storage_only_baseline_context_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        method = str(row.get("method", row.get("artifact_method", "")))
        is_storage_only = bool(row.get("archive_only_compression")) or "gzip" in method.lower() or "binary csr" in method.lower()
        if not is_storage_only:
            continue
        out.append(
            {
                "method": method,
                "storage_only_baseline": True,
                "official_main_table_eligible": False,
                "interpretation": "archival_or_loader_adapter_only_not_relation_channel_workload_reduction",
                "requires_loader_adapter": bool(row.get("uses_loader_adapter", row.get("requires_loader_adapter", False))),
                "archive_only_compression": bool(row.get("archive_only_compression", False)),
            }
        )
    return out
