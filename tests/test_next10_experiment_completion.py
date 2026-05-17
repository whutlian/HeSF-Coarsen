import json
from pathlib import Path

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.summarize_next9_hgb_guard_ablation import summarize_next9_hgb_guard_ablation
from experiments.scripts.summarize_next9_hgb_rebuttal import summarize_next9_hgb_rebuttal
from experiments.scripts.summarize_next9_ogbn_aggregation import summarize_next9_ogbn_aggregation


def test_rebuttal_summary_emits_next10_dataset_tables_and_keeps_flatten_sum(tmp_path: Path):
    next8 = tmp_path / "next8"
    runs = tmp_path / "runs"
    out = tmp_path / "out"
    write_csv(
        next8 / "per_seed_table.csv",
        [
            {
                "method": "flatten-sum",
                "dataset": "ACM",
                "seed": "12345",
                "projected_macro_f1": 0.5,
                "refined_macro_f1@0": 0.51,
                "refined_macro_f1@1": 0.52,
                "refined_macro_f1@3": 0.53,
                "refined_macro_f1@5": 0.54,
                "best_macro_f1": 0.55,
            }
        ],
    )
    write_csv(
        runs / "run_final_summary.csv",
        [
            {
                "variant": "H2-single-relation-sum",
                "dataset": "ACM",
                "seed": "12345",
                "cumulative_spectral.relation_energy_relative_error.0": 0.2,
                "cumulative_spectral.relation_energy_before.0": 10.0,
                "cumulative_spectral.relation_energy_after.0": 8.0,
                "original_edge_count_by_relation.0": 100,
                "coarse_edge_count_by_relation.0": 70,
                "relation_weight_before.0": 100.0,
                "relation_weight_after.0": 100.0,
            }
        ],
    )

    summarize_next9_hgb_rebuttal(next8_summary_dir=next8, run_summary_dirs=[runs], output=out)

    for name in (
        "relation_mass_drift_by_dataset.csv",
        "relation_js_drift_by_dataset.csv",
        "relation_energy_error_by_dataset.csv",
        "coarse_edge_collapse_by_dataset.csv",
        "self_loop_share_by_relation.csv",
        "duplicate_collapse_ratio_by_relation.csv",
        "checkpoint_refine_masking_curve.csv",
        "paper_rebuttal_table.csv",
    ):
        assert (out / name).exists(), name

    paper = (out / "paper_rebuttal_table.csv").read_text(encoding="utf-8")
    assert "flatten-sum" in paper
    assert "relation_energy_error_mean" in paper
    assert "duplicate_collapse_ratio_mean" in paper


def test_guard_summary_consumes_actual_ablation_rows_without_placeholders(tmp_path: Path):
    actual = tmp_path / "actual"
    out = tmp_path / "out"
    write_csv(
        actual / "run_final_summary.csv",
        [
            {
                "method": "HeSF-LVC-P",
                "variant": "P_spectral_guard",
                "dataset": "ACM",
                "seed": "12345",
                "target_hit": "true",
                "cumulative_dee": 0.01,
                "cumulative_ree_max": 0.02,
                "cumulative_sipe": 0.3,
                "task_projected_macro_f1": 0.6,
                "task_refined_macro_f1@5": 0.7,
                "task_best_refined_macro_f1": 0.71,
                "task_refine_auc_macro_f1": 0.68,
                "spectral_guard.guard_enabled": "true",
                "spectral_guard.guard_triggered": "true",
                "spectral_guard.trigger_reason": "q95",
                "spectral_guard.rejected_by_spec_count": 4,
                "spectral_guard.rejected_by_spec_share": 0.1,
                "spectral_guard.target_pressure_accept_count": 0,
                "source_aware_guard.source_selected_share_before.onehop": 0.5,
                "source_aware_guard.source_selected_share_after.onehop": 0.1,
                "source_aware_guard.source_avg_delta_spec_before.onehop": 8.0,
                "source_aware_guard.source_avg_delta_spec_after.onehop": 2.0,
                "cluster_size_histogram": '{"1": 3}',
            },
            {
                "method": "HeSF-LVC-P",
                "variant": "P_baseline",
                "dataset": "ACM",
                "seed": "12345",
                "target_hit": "true",
                "cumulative_dee": 0.02,
                "cumulative_ree_max": 0.03,
                "cumulative_sipe": 0.4,
                "task_projected_macro_f1": 0.61,
                "task_refined_macro_f1@5": 0.705,
                "task_best_refined_macro_f1": 0.715,
                "task_refine_auc_macro_f1": 0.69,
            },
        ],
    )

    summarize_next9_hgb_guard_ablation(actual_summary=actual, output=out)

    main = (out / "guard_ablation_main_table.csv").read_text(encoding="utf-8")
    delta = (out / "guard_delta_vs_baseline.csv").read_text(encoding="utf-8")
    trigger = (out / "guard_trigger_diagnostics.csv").read_text(encoding="utf-8")
    assert "not_run" not in main
    assert "P_spectral_guard" in main
    assert "best_macro_f1_drop_vs_baseline" in delta
    assert "activation_rate" in trigger
    assert "onehop_high_delta_selected_share" in trigger


def test_ogbn_summary_reads_fresh_aggregation_diagnostics(tmp_path: Path):
    run = tmp_path / "runs" / "next10_200k_P"
    out = tmp_path / "out"
    write_json(
        run / "metadata.json",
        {
            "size": "200k",
            "method": "HeSF-LVC-P",
            "status": "success",
        },
    )
    write_json(
        run / "level_1" / "diagnostics.json",
        {
            "candidate_count_total": 20,
            "candidate_retained_pair_count": 10,
            "matched_units": 5,
            "coarse_edge_count_by_relation": {"0": 7},
            "runtime_by_stage": {"matching": 0.2, "aggregation": 0.4},
            "large_graph_envelope": {"process_rss_bytes": 1024**3},
            "aggregation": {
                "aggregation_total_sec": 0.4,
                "aggregation_relation_loop_sec": 0.3,
                "aggregation_assignment_map_sec": 0.01,
                "aggregation_key_build_sec": 0.02,
                "aggregation_sort_sec": 0.03,
                "aggregation_reduce_sec": 0.04,
                "aggregation_dedup_sec": 0.05,
                "aggregation_shard_write_sec": 0.06,
                "aggregation_kway_merge_sec": 0.07,
                "aggregation_output_write_sec": 0.08,
                "aggregation_by_relation": [
                    {
                        "relation_id": 0,
                        "relation_name": "r0",
                        "original_edges": 10,
                        "coarse_edges_before_dedup": 10,
                        "coarse_edges_after_dedup": 7,
                        "uniqueness_ratio": 0.7,
                        "aggregation_sec": 0.3,
                        "edges_per_sec": 33.0,
                        "rss_before_gb": 1.0,
                        "rss_after_gb": 1.1,
                        "edge_weight_original_sum": 10.0,
                        "edge_weight_coarse_sum": 10.0,
                        "edge_weight_abs_error": 0.0,
                    }
                ],
            },
        },
    )

    summarize_next9_ogbn_aggregation(input_runs=tmp_path / "runs", output=out)

    stage = (out / "aggregation_stage_breakdown.csv").read_text(encoding="utf-8")
    relation = (out / "aggregation_by_relation.csv").read_text(encoding="utf-8")
    preservation = (out / "edge_weight_preservation_checks.csv").read_text(encoding="utf-8")
    assert "relation_loop_sec" in stage
    assert "assignment_map_sec" in stage
    assert "input_edges" in relation
    assert "coarse_edges" in relation
    assert "0.01" in stage
    assert "r0" in relation
    assert "0.0" in preservation
