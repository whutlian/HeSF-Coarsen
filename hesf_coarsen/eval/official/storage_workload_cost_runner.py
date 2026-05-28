from __future__ import annotations


GATE21_9_WORKLOAD_FIELDS = (
    "artifact_construction_time_seconds",
    "export_time_seconds",
    "load_time_seconds",
    "decompress_time_seconds",
    "official_sehgnn_preprocess_time_seconds",
    "training_time_seconds",
    "eval_time_seconds",
    "total_workload_time_seconds",
    "peak_cpu_rss_mb",
    "peak_gpu_memory_mb",
    "preprocessed_cache_bytes",
)


def workload_cost_fields() -> tuple[str, ...]:
    return GATE21_9_WORKLOAD_FIELDS
