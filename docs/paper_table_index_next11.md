# Next11 Paper Table Index

This index maps each paper or appendix table to the generated Next11 outputs.
Generated CSVs and figures are kept under `outputs/` and are not committed.

## Main Paper Tables

| paper item | purpose | source output |
| --- | --- | --- |
| E1 HGB main table | P/S, H0, random, GraphZoom-style, ConvMatch-style, flatten-sum, H6, full graph baselines | `outputs/exp_next11_hgb_rebuttal_paper_table_20260517/paper_rebuttal_table_aggregate.csv` |
| E1 per-dataset HGB table | ACM/DBLP/IMDB mean/std split to avoid cross-dataset variance confusion | `outputs/exp_next11_hgb_rebuttal_paper_table_20260517/paper_rebuttal_table_by_dataset.csv` |
| E2 flatten/H6 rebuttal | relation drift, relation energy error, edge collapse, refine/masking evidence | `outputs/exp_next11_hgb_rebuttal_paper_table_20260517/` and `outputs/exp_next11_hgb_task_stress_20260517_summary/` |
| E4 quality-cost | task recovery versus wall-clock/RSS/graph size | `outputs/exp_next11_hgb_external_baselines_20260517_summary/external_baseline_by_method.csv` and Next10 resource logs |
| E5 OGBN system | scale plus aggregation bottleneck breakdown | `outputs/exp_next11_ogbn_aggregation_optimizer_20260517_summary/` |

## Appendix Tables

| appendix item | purpose | source output |
| --- | --- | --- |
| DEE consistency audit | disambiguate paper-final and resource-logged DEE fields | `outputs/exp_next11_dee_consistency_20260517/` |
| Guard ablation | appendix/future safeguard validation | `outputs/exp_next11_guard_appendix_20260517/` |
| HGB stress tests | low-label, early-refine, cross-model, relation-mask stress | `outputs/exp_next11_hgb_task_stress_20260517_summary/` |
| External baseline | protocol-matched AH-UGC-style baseline | `outputs/exp_next11_hgb_external_baselines_20260517_summary/` |
| Bounded metapath sanity | bounded relation-compatible actual-graph samples | `outputs/exp_next11_bounded_metapath_sanity_20260517_summary/` |

## Required Figures

| figure | source |
| --- | --- |
| relation drift / energy rebuttal | `outputs/exp_next11_hgb_rebuttal_paper_table_20260517/figures/` |
| stress deltas and win rates | `outputs/exp_next11_hgb_task_stress_20260517_summary/figures/` |
| guard acceptance | `outputs/exp_next11_guard_appendix_20260517/figures/` |
| external baseline comparison | `outputs/exp_next11_hgb_external_baselines_20260517_summary/figures/` |
| OGBN aggregation stage and speedup | `outputs/exp_next11_ogbn_aggregation_optimizer_20260517_summary/figures/` |
| bounded metapath sanity | `outputs/exp_next11_bounded_metapath_sanity_20260517_summary/figures/` |

## Naming Rules

- Do not publish a bare `DEE` column when mixing Next9 paper-final and Next10
  resource-logged outputs.
- Use `paper_final_dee`, `resource_logged_cumulative_dee`, or
  `resource_logged_final_level_dee`.
- For task results, report `projected`, `refined@0`, `refined@1`,
  `refined@3`, `refined@5`, `best`, `AUC`, runtime, peak memory, and
  `target_hit` when available.
- For OGBN, report system metrics only: candidate pairs, scored pairs,
  selected merges, coarse edges, stage timing, per-relation breakdown,
  edges/sec, pairs/sec, RSS, and shard size.

