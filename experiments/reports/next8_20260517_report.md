# Next8 HeSF-LVC Experiment Report

Environment: local conda env `pytorch` via `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.
GPU task eval used `cuda`. No local run hit OOM, so no server fallback command was needed.

## P0 Final Gap Table

Output: `outputs/exp_next8_final_gap_20260517_summary`

The final table now reports ACM/DBLP/IMDB, 5 seeds, per-dataset rows, explicit checkpoint
metrics, `target_hit`, full graph baselines, and renamed oracle coarse-baseline gaps.

Key rows from `final_gap_main_table.csv`:

| method | DEE | REEmax | SIPE | refined@5 | best |
| --- | ---: | ---: | ---: | ---: | ---: |
| HeSF-LVC-P | 0.0209 | 0.0949 | 0.5390 | 0.7383 | 0.7423 |
| HeSF-LVC-S | 0.0160 | 0.0909 | 0.5361 | 0.7317 | 0.7389 |
| flatten-sum | 0.1792 | 0.2593 | 0.5789 | 0.7397 | 0.7398 |
| H6-no-spec | 0.1775 | 0.1987 | 0.5656 | 0.7437 | 0.7483 |
| full RGCN tuned |  |  |  | 0.7639 | 0.7639 |

Interpretation: P/S give the clean operator-preservation result; flatten-sum and H6 remain
task-competitive but much worse on spectral metrics. Tuned full RGCN remains stronger on task F1,
so the main claim should be quality-cost and preservation, not task-SOTA.

## P1 Flatten-sum Challenge

Output: `outputs/exp_next8_p1_flatten_sum_challenge_20260517`

Generated:

- `checkpoint_comparison.csv`: projected-only, refined@0/@1/@3/@5, best, and spectral metrics.
- `cross_model_transfer.csv`: RGCN-lite / HAN-small / HGT-lite coarse graph task models.
- `low_label_transfer.csv`: train_fraction=0.1, val_fraction=0.1.
- `flatten_sum_failure_by_dataset.csv`: per-dataset P/S vs flatten-sum failure-case view.

Low-label best macro-F1 highlights:

| method | ACM | DBLP | IMDB |
| --- | ---: | ---: | ---: |
| HeSF-LVC-P | 0.9089 | 0.7559 | 0.3850 |
| HeSF-LVC-S | 0.8872 | 0.7612 | 0.3828 |
| flatten-sum | 0.8950 | 0.7627 | 0.3869 |

Interpretation: flatten-sum remains a dangerous task baseline. The stronger paper positioning is
operator-preserving coarsening, with task-recovery evidence treated as supporting analysis rather
than the primary novelty.

## P2 Frozen Main Configs

Updated `docs/paper_configs_next7.md` and `configs/paper/hgb_hesf_lvc_t.yaml`.

Main configs are frozen as:

- HeSF-LVC-P: `lambda_spec=0.25`, `lambda_conv=0`, `lambda_rel=0`
- HeSF-LVC-S: `lambda_spec=0.5`, `lambda_conv=0`, `lambda_rel=0`
- HeSF-LVC-T: retained only as a SIPE/REEmax appendix variant

The novelty wording should be relation-preserving heterogeneous fused operators and operator
preservation, not relation-profile or convolution-aware scoring as the core contribution.

## P3 Source-aware Filtering

Outputs:

- Source-aware runs: `outputs/exp_next8_hgb_lvc_P_sourceaware_20260517`,
  `outputs/exp_next8_hgb_lvc_S_sourceaware_20260517`
- Summary: `outputs/exp_next8_hgb_lvc_sourceaware_5seed_20260517_summary`
- Baseline-vs-source-aware comparison:
  `outputs/exp_next8_p3_source_aware_20260517_summary`

ACM shows the intended source-policy effect:

| policy | method | onehop retained | rejected by spec | best |
| --- | --- | ---: | ---: | ---: |
| baseline | HeSF-LVC-P | 291.0 |  | 0.9233 |
| source-aware | HeSF-LVC-P | 129.2 | 45.2 | 0.9217 |
| baseline | HeSF-LVC-S | 289.8 |  | 0.9123 |
| source-aware | HeSF-LVC-S | 125.4 | 44.0 | 0.9162 |

DBLP/IMDB had no onehop-selected issue under the baseline policy, so source-aware results are
effectively unchanged there. This supports the claim that the new policy reduces onehop spectral
pollution where it appears while maintaining target hit.

## P4 OGBN-MAG System Section

Output: `outputs/exp_next8_ogbn_system_scale_20260517_summary`

This is system/protocol evidence only, not task-quality evidence.

| size | candidate pairs | scored pairs | selected merges | coarse edges | matching sec | aggregation sec | RSS GB | shard GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200k | 375863 | 322022 | 89999 | 911602 | 0.8259 | 1.8852 | 0.9228 | 0.0272 |
| 500k | 1370512 | 854317 | 225000 | 4099440 | 2.1807 | 8.2547 | 1.3757 | 0.1243 |
| 1M | 5308325 | 2061940 | 450000 | 11916388 | 5.0207 | 27.3368 | 2.2567 | 0.3643 |
| full-local-1.94M | 6487947 | 2592982 | 843615 | 17965976 | 9.7819 | 42.6586 | 2.8785 | 0.5533 |

The bottleneck remains aggregation, so future engineering should focus on relation-wise
aggregation, sort-reduce/dedup, per-relation parallel aggregation, uniqueness ratio, and memory
traffic rather than only two-hop candidate generation.

## P5 Quality-cost Pareto

Output: `outputs/exp_next8_final_gap_20260517_summary`

Generated:

- `quality_cost_pareto_points.csv`
- `figures/quality_cost_best_macro_f1_vs_wall_clock.png`
- `figures/quality_cost_best_macro_f1_vs_peak_memory.png`
- `figures/quality_cost_best_macro_f1_vs_train_time.png`

Peak memory is blank for HGB rows where previous runs did not record memory, but wall-clock and
train-time Pareto points are available. Full graph references are included as rows with
`coarse_graph_ratio=1.0`.
