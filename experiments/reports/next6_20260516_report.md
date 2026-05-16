# Next6 Claim-Sealing Experiments - 2026-05-16

## Scope

Local environment: `C:\Users\slian\anaconda3\envs\pytorch\python.exe`, device `cuda` where task eval/coarsening configs requested it. No run hit OOM. The full-local OGBN-MAG 1.94M run completed locally after fixing an O(N * clusters) diagnostics path.

| block | status | artifact |
| --- | --- | --- |
| P0A H2-lite lambda_rel=0 5 seeds | done | outputs/exp_next6_h2_lite_rel0_5seed_20260516_summary |
| P1 lambda grid HGB H2/H3 120 runs | done | outputs/exp_next6_lambda_grid_hgb_20260516_summary |
| P2 selected-source diagnostics | implemented + populated | diagnostics.json / candidate_source_pareto.csv |
| P3 OGBN scalability 200k/500k/1M/full-local | done locally | outputs/ogbn_mag_next6_stage_*_20260516_summary |
| P0D OGBN official split sanity | done | outputs/ogbn_mag_next6_stage_full_local_20260516/task_eval_summary.csv |

## P0A - H2-lite lambda_rel=0, 5 seeds

| dataset | runs | DEE | REEmax | SIPE | projected macro-F1 | refined macro-F1 | best macro-F1 | refine AUC | runtime sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACM | 5 | 0.0909 +/- 0.0066 | 0.1973 +/- 0.0074 | 0.5634 +/- 0.0028 | 0.8937 +/- 0.0300 | 0.9071 +/- 0.0236 | 0.9237 +/- 0.0225 | 0.8710 +/- 0.0391 | 12.0508 +/- 0.1534 |
| DBLP | 5 | 0.0455 +/- 0.0012 | 0.0600 +/- 0.0058 | 0.5217 +/- 0.0006 | 0.6945 +/- 0.0075 | 0.8522 +/- 0.0089 | 0.8522 +/- 0.0090 | 0.8431 +/- 0.0130 | 10.9980 +/- 0.1091 |
| IMDB | 5 | 0.0526 +/- 0.0034 | 0.0746 +/- 0.0041 | 0.5259 +/- 0.0013 | 0.3982 +/- 0.0185 | 0.4482 +/- 0.0146 | 0.4482 +/- 0.0146 | 0.4250 +/- 0.0161 | 4.7923 +/- 0.1100 |

## P1 - Lambda Grid Pareto Knee Candidates

Aggregated over ACM/DBLP/IMDB for each variant/lambda pair. Knee score is normalized distance to low spectral error, high best macro-F1, and low runtime.

| variant | lambda_spec | lambda_conv | lambda_rel | DEE | REEmax | SIPE | best macro-F1 | AUC | runtime sec | knee |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2 | 0.25 | 0 | 0 | 0.0207 | 0.0921 | 0.5403 | 0.7510 | 0.7303 | 9.4 | 0.544 |
| H3 | 0.5 | 0 | 0 | 0.0152 | 0.0906 | 0.5372 | 0.7469 | 0.7114 | 9.4 | 0.555 |
| H2 | 2 | 0.25 | 0 | 0.0319 | 0.0894 | 0.5333 | 0.7466 | 0.7269 | 9.4 | 0.588 |
| H2 | 0.5 | 0.25 | 0 | 0.0469 | 0.1127 | 0.5394 | 0.7460 | 0.7216 | 9.4 | 0.661 |
| H3 | 0.25 | 0 | 0 | 0.0207 | 0.0921 | 0.5403 | 0.7510 | 0.7303 | 9.6 | 0.688 |
| H3 | 1 | 0.25 | 0 | 0.0369 | 0.1015 | 0.5358 | 0.7468 | 0.7225 | 9.6 | 0.705 |
| H3 | 0.5 | 0.25 | 0 | 0.0469 | 0.1127 | 0.5394 | 0.7460 | 0.7216 | 9.5 | 0.712 |
| H3 | 2 | 0.25 | 0 | 0.0319 | 0.0894 | 0.5333 | 0.7466 | 0.7269 | 9.7 | 0.745 |
| H2 | 0.5 | 0 | 0 | 0.0152 | 0.0906 | 0.5372 | 0.7469 | 0.7114 | 9.7 | 0.746 |
| H3 | 1 | 1 | 0 | 0.1070 | 0.1320 | 0.5419 | 0.7463 | 0.7076 | 9.3 | 0.803 |

## P2 - Full-Local Selected Source Analysis

| source | generated | generated share | retained | selected merges | selected share | avg score | avg delta spec | avg delta conv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bucket | 974,688 | 15.0% | 923,810 | 438,560 | 52.0% | 0.299 | 0.495 | 0.0089 |
| fallback | 96,988 | 1.5% | 94,025 | 74,860 | 8.9% | 0.396 | 4.304 | 0.0069 |
| onehop | 5,416,271 | 83.5% | 1,575,147 | 330,195 | 39.1% | 1.060 | 16.024 | 0.0132 |

## P3 - OGBN Matching/Aggregation Scalability

| size | nodes | candidate pairs | scored pairs | selected merges | coarse edges | matching sec | aggregation sec | edges/sec | pairs/sec | RSS GB | shard GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200k | 200,000 | 375,863 | 322,022 | 89,999 | 911,602 | 0.83 | 1.89 | 510082 | 479810 | 0.92 | 0.03 |
| 500k | 500,000 | 1,370,512 | 854,317 | 225,000 | 4,099,440 | 2.18 | 8.25 | 545952 | 371962 | 1.38 | 0.12 |
| 1M | 1,000,000 | 5,308,325 | 2,061,940 | 450,000 | 11,916,388 | 5.02 | 27.34 | 485539 | 319884 | 2.26 | 0.36 |
| 1.94M full-local | 1,939,743 | 6,487,947 | 2,592,982 | 843,615 | 17,965,976 | 9.78 | 42.66 | 494883 | 304739 | 2.88 | 0.55 |

## P0D - OGBN Official Split Sanity

| split policy | consistency | train | valid | test | projected macro-F1 | refined macro-F1 | best macro-F1 | AUC | task sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official | official_target_type | 629,571 | 64,879 | 41,939 | 0.000331 | 0.000596 | 0.000596 | 0.000450 | 6.73 |

## Notes

- P0B/P0C were not re-run as new full mainline baselines in this pass; use the existing next5/next4 summaries for the H2/H3 paired and quality-cost baseline context: `outputs/exp_next5_hgb_final_5seed_20260516_summary`, `outputs/exp_next5_hgb_final_5seed_mainonly_20260516_summary`, and `outputs/exp_next4_mainline_full_20260515_summary`.
- P2 fields now emitted in diagnostics: `generated_candidates_by_source`, `selected_merges_by_source`, `selected_source_avg_score`, `selected_source_avg_delta_spec`, `selected_source_avg_delta_conv`, and `selected_source_cluster_size_hist`.
- Full-local OGBN stage split confirms the bottleneck is aggregation wall time, not matching or GPU memory. For 1.94M, matching was 9.78s and aggregation was 42.66s.
