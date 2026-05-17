# Next12 Paper Table Index

## Main HGB Tables

- `outputs/exp_next12_paper_tables_20260517_summary/table1_hgb_main_operator_task.csv`
- `outputs/exp_next12_paper_tables_20260517_summary/table2_flatten_h6_rebuttal_with_metapath.csv`
- `outputs/exp_next12_paper_tables_20260517_summary/table3_external_baselines_with_ahugc_tuned_if_available.csv`
- `outputs/exp_next12_paper_tables_20260517_summary/table4_metapath_diagnostics.csv`
- `outputs/exp_next12_paper_tables_20260517_summary/table5_claim_boundary.csv`

Use Table 2 for the rebuttal-style comparison against flatten-sum and H6-no-spec. It includes explicit `paper_final_dee`, relation metrics, metapath collapse/count diagnostics, and task F1 columns.

## AH-UGC-Style Tuning

- `outputs/exp_next12_ahugc_style_sweep_20260517_summary/ahugc_style_best_overall.csv`
- `outputs/exp_next12_ahugc_style_sweep_20260517_summary/ahugc_style_best_config_by_dataset.csv`
- `outputs/exp_next12_ahugc_style_sweep_20260517_summary/ahugc_style_sweep_by_config.csv`

The best overall protocol-matched AH-UGC-style config is:

```text
hash_bits = 20
bucket_topk = 4
assignment_source = chebheat_sketch
target_hit_all = True
```

## Structure-Sensitive Stress

- `outputs/exp_next12_structure_sensitive_stress_20260517_summary/structure_stress_by_method.csv`
- `outputs/exp_next12_structure_sensitive_stress_20260517_summary/structure_stress_win_rates.csv`

Structure-only stress supports a cautious robustness statement. Feature-mask and feature-noise stress do not support task-dominance claims.

## OGBN System Table

- `outputs/exp_next12_ogbn_aggregation_backend_20260517_summary/aggregation_backend_speedup_summary.csv`

Use this only for system/profiling discussion. A3 packed-key sort passed correctness but is not adopted because full-local speedup did not reach the 1.25x rule.
