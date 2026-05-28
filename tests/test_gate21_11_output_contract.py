from __future__ import annotations


def test_gate21_11_required_summary_files_are_listed() -> None:
    from experiments.scripts.gate21_11_common import SUMMARY_FILES

    required = {
        "gate21_11_decision.json",
        "gate21_11_decision.md",
        "gate21_11_official_main_by_method.csv",
        "gate21_11_budgeted_selector_by_method.csv",
        "gate21_11_channel_planner_trace.csv",
        "gate21_11_external_tp_5x5_runs.csv",
        "gate21_11_external_tp_by_method.csv",
        "gate21_11_external_tp_budget_audit.csv",
        "gate21_11_freehgc_standard_runs.csv",
        "gate21_11_freehgc_standard_by_method.csv",
        "gate21_11_freehgc_tp_adapter_audit.csv",
        "gate21_11_freehgc_env_audit.csv",
        "gate21_11_metapath_tensor_dump.csv",
        "gate21_11_cache_hash_assertions.csv",
        "gate21_11_feature_ablation_task_runs.csv",
        "gate21_11_feature_ablation_by_method.csv",
        "gate21_11_adapter_package_audit.csv",
        "gate21_11_adapter_by_method.csv",
        "gate21_11_system_cost_runs.csv",
        "gate21_11_system_cost_by_method.csv",
        "gate21_11_cross_dataset_task_runs.csv",
        "gate21_11_cross_dataset_by_method.csv",
        "gate21_11_coverage_semantic_diagnostics.csv",
        "gate21_11_apv16_deterministic_proof.json",
    }

    assert required.issubset(set(SUMMARY_FILES))
