from __future__ import annotations

import math


def test_gate21_13_required_summary_files_are_listed() -> None:
    from experiments.scripts.gate21_13_common import SUMMARY_FILES

    required = {
        "gate21_13_manifest.json",
        "gate21_13_decision.json",
        "gate21_13_decision.md",
        "gate21_13_icde_evidence_manifest.json",
        "gate21_13_official_main_by_method.csv",
        "gate21_13_budgeted_selector_by_method.csv",
        "gate21_13_selector_hash_audit.csv",
        "gate21_13_deterministic_selector_proof.csv",
        "gate21_13_external_tp_by_method_budget.csv",
        "gate21_13_freehgc_standard_by_ratio.csv",
        "gate21_13_freehgc_tp_adapter_audit.csv",
        "gate21_13_metapath_tensor_dump.csv",
        "gate21_13_feature_ablation_by_method.csv",
        "gate21_13_adapter_by_method.csv",
        "gate21_13_system_cost_by_method.csv",
        "gate21_13_cross_dataset_by_method.csv",
        "gate21_13_requirement_checklist.md",
        "gate21_13_prompt_completion_checklist.md",
    }

    assert required.issubset(set(SUMMARY_FILES))


def test_gate21_13_decision_rejects_smoke_nan_hard_failure_and_empty_hashes() -> None:
    from hesf_coarsen.eval.official.gate21_13_decision import gate21_13_decision

    flags = gate21_13_decision(
        official_rows=[
            {
                "method": "HeSF-RCS-APV12",
                "row_kind": "planner_plan",
                "eligible_for_official_main_table": False,
                "official_hgb_exported": False,
                "official_sehgnn_unmodified": False,
                "training_executed": False,
                "test_micro_f1": 0.944,
                "test_macro_f1": 0.940,
            }
        ],
        selector_hash_audit=[{"selector_hash_audit_pass": False}],
        external_tp_rows=[
            {
                "method": "Random-HG-TP",
                "budget_type": "support_node_ratio",
                "requested_budget": 0.5,
                "training_executed": True,
                "success": True,
                "official_hgb_exported": True,
                "official_sehgnn_unmodified": True,
                "test_micro_f1": math.nan,
                "test_macro_f1": 0.8,
                "budget_matched_within_tolerance": True,
            }
        ],
        freehgc_tp_runs=[
            {
                "method": "FreeHGC-TP-selection",
                "training_executed": False,
                "failure_type": "hard_incompatibility",
                "failure_reason": "edge_provenance_missing",
                "hard_failure": True,
            }
        ],
        metapath_rows=[
            {
                "method": "HeSF-RCS-APV16",
                "real_tensor_dumped": False,
                "feature_tensor_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "feature_tensor_bytes": 0,
            }
        ],
        cache_rows=[
            {
                "method": "HeSF-RCS-APV16",
                "assertion_pass": True,
                "cache_file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        ],
    )

    assert flags["OFFICIAL_DBLP_APV12_PASS"] is False
    assert flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] is False
    assert flags["EXTERNAL_TP_5X5_REQUIRED_READY"] is False
    assert flags["FREEHGC_TP_SELECTION_READY"] is False
    assert flags["FREEHGC_TP_SYNTHETIC_READY_OR_HARD_INCOMPATIBILITY_PROVEN"] is True
    assert flags["METAPATH_TENSOR_DUMP_READY"] is False
    assert flags["CACHE_HASH_REAL_PASS"] is False
    assert flags["ICDE_EVIDENCE_READY"] is False


def test_gate21_13_decision_accepts_hash_audited_official_apv12_apv16_as_partial() -> None:
    from hesf_coarsen.eval.official.gate21_13_decision import decision_status, gate21_13_decision
    from hesf_coarsen.eval.official.selector_result_linkage import gate21_13_budgeted_selector_linkage

    selector = gate21_13_budgeted_selector_linkage("DBLP", [0.12, 0.16, 0.20, 0.30])
    official = [row for row in selector["selector_rows"] if row["row_kind"] == "linked_task_result"]
    flags = gate21_13_decision(
        official_rows=official,
        selector_hash_audit=selector["hash_audit_rows"],
        deterministic_proof_rows=selector["deterministic_proof_rows"],
    )

    assert flags["OFFICIAL_DBLP_APV12_PASS"] is True
    assert flags["OFFICIAL_DBLP_APV16_PASS"] is True
    assert flags["BUDGETED_SELECTOR_HASH_AUDIT_PASS"] is True
    assert flags["APV16_DETERMINISTIC_PROOF_PASS"] is True
    assert flags["ICDE_EVIDENCE_READY"] is False
    assert decision_status(flags) == "GATE21_13_PARTIAL_EXECUTED_EVIDENCE"
