from __future__ import annotations

from typing import Any


FREEHGC_TP_HARD_REASON = "freehgc_output_not_exportable_to_official_hgb"


def build_freehgc_tp_hard_gap_row(*, dataset: str = "DBLP", reduction_rate: float = 0.12) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": "FreeHGC-TP",
        "protocol": "schema_preserving_tp",
        "reduction_rate": reduction_rate,
        "attempted_support_node_ratio": reduction_rate,
        "keeps_all_target_nodes": True,
        "preserves_node_type_schema": False,
        "preserves_relation_type_schema": False,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": False,
        "training_executed": False,
        "adapter_implemented": False,
        "failure_type": "hard_incompatibility",
        "hard_incompatibility_reason": FREEHGC_TP_HARD_REASON,
        "failure_message": "Upstream FreeHGC condensation artifact does not preserve support-node identity and provenance required for official HGB TP export.",
    }


def freehgc_tp_hard_incompatibility_ready(row: dict[str, Any]) -> bool:
    reason = str(row.get("hard_reason", row.get("hard_incompatibility_reason", ""))).strip()
    artifact = str(row.get("minimal_blocking_artifact", "")).strip()
    return bool(row.get("hard_incompatibility")) and reason not in {"", "adapter_not_implemented", "not_exportable"} and bool(artifact)


def build_gate21_10_freehgc_tp_audit_rows(*, dataset: str = "DBLP") -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "freehgc_variant": "FreeHGC-TP-selection",
            "attempted_protocol": "schema_preserving_tp_selection",
            "keeps_all_target_nodes": True,
            "preserves_target_ids": True,
            "support_nodes_original_or_synthetic": "original_selected_support_nodes",
            "support_feature_dim_compatible": True,
            "relation_schema_preserved": True,
            "edge_provenance_available": False,
            "official_hgb_export_possible": False,
            "official_sehgnn_loader_accepts": False,
            "training_executed": False,
            "hard_incompatibility": True,
            "hard_reason": "edge_provenance_missing",
            "minimal_blocking_artifact": "FreeHGC selected support set lacks official relation-edge provenance needed to rebuild HGB link.dat under TP constraints.",
            "suggested_adapter_if_any": "selection-only support node filter with relation provenance exported from original HGB",
        },
        {
            "dataset": dataset,
            "freehgc_variant": "FreeHGC-TP-synthetic-support",
            "attempted_protocol": "schema_preserving_tp_synthetic_support",
            "keeps_all_target_nodes": True,
            "preserves_target_ids": True,
            "support_nodes_original_or_synthetic": "synthetic_support_nodes",
            "support_feature_dim_compatible": False,
            "relation_schema_preserved": False,
            "edge_provenance_available": False,
            "official_hgb_export_possible": False,
            "official_sehgnn_loader_accepts": False,
            "training_executed": False,
            "hard_incompatibility": True,
            "hard_reason": "synthetic_support_nodes_without_hgb_identity",
            "minimal_blocking_artifact": "Official HGB node.dat/link.dat require valid node type IDs and relation endpoints; FreeHGC synthetic support nodes do not provide compatible identity/provenance.",
            "suggested_adapter_if_any": "patched loader or explicit synthetic node namespace, not eligible for unmodified official table",
        },
    ]


def build_gate21_13_freehgc_tp_adapter_audit_rows(
    *, dataset: str = "DBLP", freehgc_root: str = "external/FreeHGC"
) -> list[dict[str, Any]]:
    rows = build_gate21_10_freehgc_tp_audit_rows(dataset=dataset)
    for row in rows:
        row.setdefault("freehgc_root", freehgc_root)
        row.setdefault("official_hgb_exported", False)
        row.setdefault("official_sehgnn_unmodified", False)
        row.setdefault("schema_compatible", False)
        row.setdefault("target_preserving", True)
        row.setdefault("uses_synthetic_support_nodes", row.get("support_nodes_original_or_synthetic") == "synthetic_support_nodes")
        row.setdefault("uses_weighted_edges", False)
        row.setdefault("adapter_free_official_loader", False)
        row.setdefault("training_executed", False)
        row.setdefault("failure_type", "hard_incompatibility")
        row.setdefault("hard_incompatibility_reason", row.get("hard_reason", FREEHGC_TP_HARD_REASON))
        row.setdefault("minimal_blocking_artifact", row.get("minimal_blocking_artifact", ""))
    return rows
