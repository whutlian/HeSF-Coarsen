from __future__ import annotations


def test_gate21_11_external_tp_by_method_requires_25_ready_runs_per_method_budget() -> None:
    from hesf_coarsen.eval.official.external_tp_5x5_runner import summarize_gate21_11_external_tp

    runs = [
        {
            "dataset": "DBLP",
            "method": "Random-HG-TP",
            "budget_family": "structural_ratio",
            "requested_budget": 0.16,
            "graph_seed": graph_seed,
            "training_seed": training_seed,
            "training_executed": True,
            "success": True,
            "official_hgb_exported": True,
            "official_sehgnn_unmodified": True,
            "budget_matched_within_tolerance": True,
            "test_micro_f1": 0.8,
            "test_macro_f1": 0.79,
        }
        for graph_seed in range(1, 6)
        for training_seed in range(1, 6)
    ]
    summary = summarize_gate21_11_external_tp(runs, required_methods=("Random-HG-TP",))

    assert summary[0]["ready_run_count"] == 25
    assert summary[0]["expected_run_count"] == 25
    assert summary[0]["eligible_for_main_comparison"] is True


def test_gate21_11_system_cost_requires_preprocess_train_memory_and_cache() -> None:
    from hesf_coarsen.eval.official.end_to_end_system_cost import gate21_11_system_cost_ready

    assert gate21_11_system_cost_ready(
        [
            {
                "training_executed": True,
                "official_sehgnn_preprocess_time_seconds": 1.0,
                "training_time_seconds": 2.0,
                "peak_cpu_rss_mb": 100.0,
                "preprocessed_cache_bytes": 10,
                "test_micro_f1": 0.9,
                "test_macro_f1": 0.88,
            }
        ]
    )
    assert not gate21_11_system_cost_ready([{"training_executed": True, "training_time_seconds": 2.0}])
