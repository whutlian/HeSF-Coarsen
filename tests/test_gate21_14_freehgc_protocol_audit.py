from __future__ import annotations

from hesf_coarsen.eval.official.gate21_14_decision import gate21_14_decision


def test_freehgc_hard_failure_audit_does_not_mark_protocol_ready() -> None:
    flags = gate21_14_decision(
        freehgc_protocol_audit=[
            {
                "required_files_present": False,
                "hard_failure_reason": "freehgc_hgb_required_files_missing",
                "success": False,
            }
        ],
        freehgc_standard_by_method=[
            {
                "method": "FreeHGC-standard",
                "ratio": 0.012,
                "success_count": 0,
                "ready_5seed": False,
                "mean_micro": "NaN",
            }
        ],
        freehgc_tp_by_method=[
            {
                "method": "FreeHGC-TP-selection",
                "official_hgb_exported": False,
                "training_executed": False,
                "ready": False,
            }
        ],
    )

    assert flags["FREEHGC_STANDARD_5SEED_READY"] is False
    assert flags["FREEHGC_TP_SELECTION_READY"] is False
    assert flags["FREEHGC_SCORE_SELECTOR_READY"] is False
