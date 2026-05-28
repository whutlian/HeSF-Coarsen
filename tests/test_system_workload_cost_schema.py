from __future__ import annotations


def test_system_workload_cost_ready_requires_workload_and_task_fields() -> None:
    from hesf_coarsen.eval.official.system_workload_cost import system_workload_cost_ready

    bytes_only = [{"artifact_method": "gzip_hgb_text", "artifact_bytes": 10, "training_executed": False}]
    measured = [
        {
            "artifact_method": "APV12 official text",
            "artifact_bytes": 10,
            "load_time_seconds": 1,
            "official_sehgnn_preprocess_time_seconds": 1,
            "training_time_seconds": 1,
            "eval_time_seconds": 1,
            "peak_cpu_rss_mb": 100,
            "preprocessed_cache_bytes": 100,
            "training_executed": True,
            "task_micro_f1": 0.94,
            "task_macro_f1": 0.93,
        }
    ]

    assert system_workload_cost_ready(bytes_only) is False
    assert system_workload_cost_ready(measured) is True
