from __future__ import annotations

import math


def test_gate21_10_ready_flags_reject_nan_smoke_and_hard_gap_as_task_ready() -> None:
    from hesf_coarsen.eval.official.gate21_10_decision import gate21_10_decision

    decision = gate21_10_decision(
        official_rows=[
            {"method": "HeSF-RCS-APV12", "test_micro_f1": 0.945, "test_macro_f1": 0.94, "official_sehgnn_unmodified": True},
            {"method": "HeSF-RCS-APV16", "test_micro_f1": 0.949, "test_macro_f1": 0.946, "official_sehgnn_unmodified": True},
        ],
        auto_selector_rows=[{"budget_target": 0.16, "AP_keep": 1, "PV_keep": 1, "PA_keep": 0.5, "VP_keep": 0.5, "PT_keep": 0, "TP_keep": 0, "selection_uses_test_metrics": False}],
        external_tp_rows=[
            {"method": "Random-HG-TP", "graph_seed": 1, "training_seed": 1, "training_executed": True, "test_micro_f1": math.nan}
        ],
        freehgc_tp_rows=[
            {"freehgc_variant": "FreeHGC-TP-selection", "hard_incompatibility": True, "hard_reason": "edge_provenance_missing", "training_executed": False}
        ],
    )

    assert decision["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] is False
    assert decision["FREEHGC_TP_TASK_RESULTS_READY"] is False
    assert decision["FREEHGC_TP_HARD_INCOMPATIBILITY_PROOF_READY"] is True
    assert decision["ICDE_EVIDENCE_READY"] is False


def test_gate21_10_requires_freehgc_config_seeds_ratios_and_split_for_standard_5seed_ready() -> None:
    from hesf_coarsen.eval.official.gate21_10_decision import gate21_10_decision

    incomplete = gate21_10_decision(
        freehgc_standard_rows=[{"ratio": 0.012, "seed": 1, "success": True, "test_micro_f1": 0.7}],
        freehgc_env_rows=[{"upstream_config_verified": True, "split_matches_hgb_official": True, "required_files_present": True}],
    )
    complete_rows = [
        {"ratio": ratio, "seed": seed, "success": True, "test_micro_f1": 0.7, "test_macro_f1": 0.68}
        for ratio in (0.012, 0.024, 0.048, 0.096, 0.120)
        for seed in range(1, 6)
    ]
    complete = gate21_10_decision(
        freehgc_standard_rows=complete_rows,
        freehgc_env_rows=[{"upstream_config_verified": True, "split_matches_hgb_official": True, "required_files_present": True}],
    )

    assert incomplete["FREEHGC_STANDARD_SINGLE_SEED_PARTIAL_READY"] is True
    assert incomplete["FREEHGC_STANDARD_5SEED_READY"] is False
    assert complete["FREEHGC_STANDARD_5SEED_READY"] is True
