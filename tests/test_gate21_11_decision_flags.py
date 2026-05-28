from __future__ import annotations

import math


def test_gate21_11_rejects_smoke_nan_and_hard_failure_as_task_ready() -> None:
    from hesf_coarsen.eval.official.gate21_11_decision import gate21_11_decision

    flags = gate21_11_decision(
        external_tp_runs=[
            {
                "method": "Random-HG-TP",
                "budget_family": "structural_ratio",
                "requested_budget": 0.16,
                "graph_seed": 1,
                "training_seed": 1,
                "training_executed": True,
                "success": True,
                "test_micro_f1": math.nan,
                "test_macro_f1": 0.8,
            }
        ],
        freehgc_tp_audit=[
            {
                "variant": "tp-selection",
                "training_executed": False,
                "hard_failure": True,
                "failure_type": "edge_provenance_missing",
                "failure_reason": "FreeHGC output lacks relation-edge provenance.",
            }
        ],
    )

    assert flags["EXTERNAL_TP_5X5_READY"] is False
    assert flags["EXTERNAL_TP_SMOKE_ONLY"] is False
    assert flags["FREEHGC_TP_SELECTION_TASK_READY"] is False
    assert flags["FREEHGC_TP_HARD_INCOMPATIBILITY_PROVEN"] is True


def test_gate21_11_freehgc_standard_requires_verified_env_and_non_nan_5seed_metrics() -> None:
    from hesf_coarsen.eval.official.gate21_11_decision import gate21_11_decision

    rows = [
        {
            "ratio": ratio,
            "seed": seed,
            "success": True,
            "training_executed": True,
            "test_micro_f1": 0.7,
            "test_macro_f1": 0.68,
        }
        for ratio in (0.012, 0.024, 0.048, 0.096, 0.120)
        for seed in range(1, 6)
    ]

    unverified = gate21_11_decision(
        freehgc_standard_runs=rows,
        freehgc_env=[{"upstream_config_verified": False, "split_matches_official_or_documented": True}],
    )
    verified = gate21_11_decision(
        freehgc_standard_runs=rows,
        freehgc_env=[{"upstream_config_verified": True, "split_matches_official_or_documented": True}],
    )

    assert unverified["FREEHGC_STANDARD_5SEED_READY"] is False
    assert verified["FREEHGC_STANDARD_5SEED_READY"] is True


def test_gate21_11_full_evidence_requires_every_readiness_family() -> None:
    from hesf_coarsen.eval.official.gate21_11_decision import gate21_11_decision

    flags = gate21_11_decision(
        official_rows=[
            {"method": "export-full-SeHGNN", "test_micro_f1": 0.95, "test_macro_f1": 0.94},
            {"method": "HeSF-RCS-APV12", "test_micro_f1": 0.944, "test_macro_f1": 0.94, "official_sehgnn_unmodified": True, "eligible_for_official_main_table": True, "structural_storage_ratio": 0.12},
            {"method": "HeSF-RCS-APV16", "test_micro_f1": 0.949, "test_macro_f1": 0.946, "official_sehgnn_unmodified": True, "eligible_for_official_main_table": True, "structural_storage_ratio": 0.16},
        ],
        budgeted_selector_rows=[
            {"requested_budget_name": "budget12", "requested_structural_budget": 0.12, "selected_canonical_method": "HeSF-RCS-APV12", "AP_keep": 1, "PV_keep": 1, "PA_keep": 0, "VP_keep": 0, "uses_test_metrics_for_selection": False},
            {"requested_budget_name": "budget16", "requested_structural_budget": 0.16, "selected_canonical_method": "HeSF-RCS-APV16", "AP_keep": 1, "PV_keep": 1, "PA_keep": 0.5, "VP_keep": 0.5, "uses_test_metrics_for_selection": False},
        ],
    )

    assert flags["OFFICIAL_MAIN_DBLP_READY"] is True
    assert flags["BUDGETED_PLANNER_DBLP_012_PASS"] is True
    assert flags["BUDGETED_PLANNER_DBLP_016_PASS"] is True
    assert flags["ICDE_SUBMISSION_EVIDENCE_READY"] is False
