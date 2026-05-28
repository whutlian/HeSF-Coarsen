from __future__ import annotations

import math


def test_gate21_12_required_summary_files_are_listed() -> None:
    from experiments.scripts.gate21_12_common import SUMMARY_FILES

    required = {
        "gate21_12_manifest.json",
        "gate21_12_decision.json",
        "gate21_12_decision.md",
        "gate21_12_by_method.csv",
        "gate21_12_official_main_by_method.csv",
        "gate21_12_budgeted_selector_by_method.csv",
        "gate21_12_external_tp_5x5_by_method.csv",
        "gate21_12_freehgc_standard_by_method.csv",
        "gate21_12_freehgc_tp_by_method.csv",
        "gate21_12_metapath_tensor_dump.csv",
        "gate21_12_cache_hash_assertions.csv",
        "gate21_12_feature_ablation_by_method.csv",
        "gate21_12_adapter_by_method.csv",
        "gate21_12_system_cost_by_method.csv",
        "gate21_12_cross_dataset_by_method.csv",
        "gate21_12_storage_audit.csv",
        "gate21_12_coverage_diagnostics.csv",
        "gate21_12_failure_audit.csv",
        "gate21_12_apv16_deterministic_proof.json",
        "gate21_12_cache_namespace_audit.csv",
        "gate21_12_feature_ablation_runs.csv",
        "gate21_12_feature_ablation_shape_audit.csv",
        "gate21_12_adapter_runs.csv",
        "gate21_12_adapter_package_audit.csv",
        "gate21_12_system_cost_runs.csv",
        "gate21_12_storage_only_baselines.csv",
        "gate21_12_cross_dataset_runs.csv",
        "gate21_12_cross_dataset_selector_plans.csv",
        "gate21_12_freehgc_standard_runs.csv",
        "gate21_12_freehgc_tp_runs.csv",
        "gate21_12_freehgc_env_audit.csv",
        "gate21_12_freehgc_failure_proof.json",
    }

    assert required.issubset(set(SUMMARY_FILES))


def test_gate21_12_decision_rejects_plan_smoke_nan_and_hard_failure_as_ready() -> None:
    from hesf_coarsen.eval.official.gate21_12_decision import gate21_12_decision

    flags = gate21_12_decision(
        official_rows=[
            {
                "method": "HeSF-RCS-APV12",
                "row_kind": "planner_plan",
                "eligible_for_official_main_table": False,
                "official_hgb_exported": False,
                "training_executed": False,
                "test_micro_f1": 0.944,
                "test_macro_f1": 0.940,
            }
        ],
        budgeted_selector_rows=[
            {
                "row_kind": "planner_plan",
                "requested_structural_budget": 0.12,
                "selected_canonical_method": "HeSF-RCS-APV12",
                "uses_test_metrics_for_selection": False,
                "BUDGETED_SELECTOR_HASH_AUDIT_PASS": True,
            }
        ],
        external_tp_rows=[
            {
                "method": "Random-HG-TP",
                "training_executed": True,
                "official_hgb_exported": True,
                "official_sehgnn_unmodified": True,
                "test_micro_f1": math.nan,
                "test_macro_f1": 0.8,
            }
        ],
        freehgc_tp_rows=[
            {
                "variant": "FreeHGC-TP-selection",
                "training_executed": False,
                "failure_type": "hard_incompatibility",
                "failure_reason": "edge_provenance_missing",
                "hard_failure": True,
            }
        ],
    )

    assert flags["OFFICIAL_MAIN_DBLP_APV12_READY"] is False
    assert flags["OFFICIAL_MAIN_BUDGETED_SELECTOR_READY"] is False
    assert flags["EXTERNAL_TP_5X5_TASK_RESULTS_READY"] is False
    assert flags["FREEHGC_TP_SELECTION_TASK_READY"] is False
    assert flags["FREEHGC_TP_HARD_FAILURE_PROOF_READY"] is True
    assert flags["ICDE_SUBMISSION_EVIDENCE_READY"] is False


def test_gate21_12_decision_accepts_hash_audited_official_apv12_apv16() -> None:
    from hesf_coarsen.eval.official.budgeted_channel_planner import plan_gate21_12_budgeted_channels
    from hesf_coarsen.eval.official.gate21_12_decision import gate21_12_decision

    selector = plan_gate21_12_budgeted_channels("DBLP", structural_budgets=[0.12, 0.16])
    linked = [row for row in selector["selector_rows"] if row["row_kind"] == "linked_task_result"]
    flags = gate21_12_decision(
        official_rows=linked,
        budgeted_selector_rows=selector["selector_rows"],
        selector_hash_audit=[selector["hash_audit"]],
        apv16_deterministic_proof=selector["apv16_deterministic_proof"],
    )

    assert flags["OFFICIAL_MAIN_DBLP_APV12_READY"] is True
    assert flags["OFFICIAL_MAIN_DBLP_APV16_READY"] is True
    assert flags["OFFICIAL_MAIN_BUDGETED_SELECTOR_READY"] is True
    assert flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] is True
    assert flags["APV16_DETERMINISTIC_PROOF_PASS"] is True
    assert flags["ICDE_SUBMISSION_EVIDENCE_READY"] is False
