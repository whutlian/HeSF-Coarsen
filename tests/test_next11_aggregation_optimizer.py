from pathlib import Path

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_next11_ogbn_aggregation_optimizer import summarize_next11_ogbn_aggregation_optimizer


def test_aggregation_optimizer_marks_unimplemented_and_computes_speedup(tmp_path: Path):
    inp = tmp_path / "runs"
    out = tmp_path / "out"
    write_csv(
        inp / "aggregation_optimizer_runs.csv",
        [
            {
                "size": "full-local",
                "method": "HeSF-LVC-P",
                "aggregation_variant": "A0_current_sort_reducer",
                "run_status": "available",
                "aggregation_total_sec": 40.0,
                "matching_sec": 5.0,
                "candidate_pairs": 100,
                "selected_merges": 50,
                "peak_rss_gb": 10.0,
                "correctness_passed": "true",
            },
            {
                "size": "full-local",
                "method": "HeSF-LVC-P",
                "aggregation_variant": "A2_chunk_size_sweep_sort",
                "run_status": "available",
                "aggregation_total_sec": 30.0,
                "matching_sec": 5.0,
                "candidate_pairs": 100,
                "selected_merges": 50,
                "peak_rss_gb": 11.0,
                "correctness_passed": "true",
            },
            {
                "size": "full-local",
                "method": "HeSF-LVC-P",
                "aggregation_variant": "A3_int64_key_sort_or_radix_if_available",
                "run_status": "not_implemented",
                "reason": "radix backend unavailable",
            },
        ],
    )
    write_csv(
        inp / "aggregation_optimizer_stage_breakdown.csv",
        [{"size": "full-local", "method": "HeSF-LVC-P", "aggregation_variant": "A0_current_sort_reducer", "relation_loop_sec": 1, "assignment_map_sec": 1, "key_build_sec": 1, "sort_sec": 1, "reduce_sec": 1, "shard_write_sec": 1, "kway_merge_sec": 1, "output_write_sec": 1}],
    )
    write_csv(
        inp / "aggregation_optimizer_by_relation.csv",
        [{"size": "full-local", "method": "HeSF-LVC-P", "aggregation_variant": "A0_current_sort_reducer", "input_edges": 10, "coarse_edges": 7, "uniqueness_ratio": 0.7, "duplicate_collapse_ratio": 0.3, "edges_per_sec": 100, "rss_before_gb": 1, "rss_after_gb": 1.1}],
    )
    write_csv(inp / "aggregation_optimizer_correctness_checks.csv", [{"size": "full-local", "method": "HeSF-LVC-P", "aggregation_variant": "A0_current_sort_reducer", "correctness_passed": "true"}])

    summarize_next11_ogbn_aggregation_optimizer(input=inp, output=out)

    speedup = (out / "aggregation_optimizer_speedup_summary.csv").read_text(encoding="utf-8")
    assert "1.333" in speedup or "1.33" in speedup
    runs = (out / "aggregation_optimizer_runs.csv").read_text(encoding="utf-8")
    assert "not_implemented" in runs
    assert (out / "figures/full_local_speedup.png").exists()

