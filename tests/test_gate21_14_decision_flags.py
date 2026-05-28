from __future__ import annotations

from hesf_coarsen.eval.official.gate21_14_decision import gate21_14_decision


def _official(method: str) -> dict[str, object]:
    return {
        "dataset": "DBLP",
        "method": method,
        "schema_compatible": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
        "uses_weighted_superedges": False,
        "uses_synthetic_nodes": False,
        "uses_adapter_loader": False,
        "eligible_for_official_main_table": True,
        "uses_test_metrics_for_selection": False,
        "training_executed": True,
        "success": True,
        "test_micro_f1": 0.94,
        "test_macro_f1": 0.93,
    }


def test_icde_ready_stays_false_when_required_real_evidence_is_missing() -> None:
    flags = gate21_14_decision(official_rows=[_official("HeSF-RCS-APV12"), _official("HeSF-RCS-APV16")])

    assert flags["OFFICIAL_DBLP_APV12_ANCHOR_PASS"] is True
    assert flags["OFFICIAL_DBLP_APV16_ANCHOR_PASS"] is True
    assert flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] is False
    assert flags["ICDE_EVIDENCE_READY"] is False


def test_nan_or_hard_failure_rows_do_not_set_readiness() -> None:
    flags = gate21_14_decision(
        external_tp_runs=[
            {
                "method": "Random-HG-TP",
                "budget_type": "structural_storage_ratio",
                "requested_budget": 0.12,
                "graph_seed": 1,
                "training_seed": 1,
                "eligible_for_external_tp_table": True,
                "budget_matched_within_tolerance": True,
                "training_executed": False,
                "success": False,
                "test_micro_f1": "NaN",
                "test_macro_f1": "NaN",
                "failure_type": "hard_failure",
            }
        ]
    )

    assert flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] is False
