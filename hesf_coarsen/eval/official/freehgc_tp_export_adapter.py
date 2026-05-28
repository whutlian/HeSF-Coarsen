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
