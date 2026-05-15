# Next5 HeSF-LVC Experiment Report

Date: 2026-05-16
Environment: local `conda` env `pytorch`, PyTorch `2.8.0+cu129`
GPU: `NVIDIA GeForce RTX 5060 Ti`

All new task/refine evaluations were run with `--device cuda`. OGBN-MAG coarsening and scale runs also record `cuda_available=true` and VRAM fields in diagnostics. No local OOM or VRAM failure occurred, so no server fallback command is required for this round.

## Output Index

- HGB final 5-seed main table: `outputs/exp_next5_hgb_final_5seed_mainonly_20260516_summary`
- Relation fusion final ablation: `outputs/exp_next5_relation_fusion_20260516_summary`
- HGB candidate ablation: `outputs/exp_next5_candidate_hgb_20260516_summary`
- OGBN-MAG 50k/200k protocol and candidate summary: `outputs/ogbn_mag_next5_quality_candidate_20260516_summary`
- OGBN-MAG 1M scale summary: `outputs/ogbn_mag_next5_scale_1m_20260516_summary`
- OGBN-MAG task eval CSVs: `outputs/*_task_eval_gpu_protocol.csv`

## Code Changes

- Added OGBN/HGB task protocol reporting: `target_node_type`, synthetic stratified target-type split, train/val/test label coverage, class-present counts, macro empty-class policy, train-only coarse label source, and leakage check.
- Added `target_node_type=paper` support for OGBN relation schemas such as `paper__cites__paper`.
- Added `macro_empty_class_policy=eval_present` and surfaced micro/macro F1 at `projected`, `refined@0/1/3/5`, `best`, and AUC.
- Added limited two-hop candidate mode: `twohop_mode=capped_sampled`, `twohop_budget_per_node`, and optional wall-clock budget, with diagnostics for skipped pairs and time-budget stop.
- Added protocol OGBN variant names: `H2-fullcand`, `H2-opt`, `H3-opt`, `H4-opt`, `flatten-sum-opt`.
- Added summary outputs for paper-ready mean +/- std tables, dataset/variant tables, GPU marking, candidate substage timings, and task protocol fields.

## Checklist Status

| item | status |
| --- | --- |
| A. HGB final confirmation H0/H2/H3/H4/H6, 5 seeds | done, GPU task eval |
| B. Relation fusion ablation with per-dataset and sampled eigensanity | done, GPU task eval |
| C. OGBN-MAG medium protocol repair | done, GPU task eval; task quality remains weak |
| D. Candidate optimization ablation | done on HGB and OGBN 50k/200k |
| E. Scale path 200k -> 1M | done locally on 1M; 5M unavailable from local source |
| F. Full graph baselines | done for HGB and OGBN task eval |

## A. HGB Final Confirmation [GPU Task Eval]

Runs: ACM/DBLP/IMDB x 5 seeds x H0/H2/H3/H4/H6 = 75. Target hit rate is 1.0 with final ratio 0.5000.

| variant | role | DEE | FSE | REEmax | SIPE | refined@5 | best | AUC | full RGCN tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | old mutual-best baseline | 0.2423 +/- 0.0161 | 0.6297 +/- 0.0030 | 0.2863 +/- 0.0527 | 0.6243 +/- 0.0075 | 0.7162 +/- 0.2217 | 0.7240 +/- 0.2221 | 0.6701 +/- 0.2036 | 0.7639 +/- 0.2002 |
| H2 | task-balanced default | 0.0684 +/- 0.0234 | 0.5337 +/- 0.0150 | 0.1124 +/- 0.0646 | 0.5376 +/- 0.0190 | 0.7349 +/- 0.2122 | 0.7367 +/- 0.2138 | 0.6961 +/- 0.2008 | 0.7639 +/- 0.2002 |
| H3 | spectral-safe | 0.0522 +/- 0.0158 | 0.5343 +/- 0.0137 | 0.1058 +/- 0.0680 | 0.5365 +/- 0.0181 | 0.7410 +/- 0.2132 | 0.7453 +/- 0.2154 | 0.7144 +/- 0.2100 | 0.7639 +/- 0.2002 |
| H4 | spectral-only / no-conv | 0.0140 +/- 0.0049 | 0.5414 +/- 0.0057 | 0.0882 +/- 0.0610 | 0.5339 +/- 0.0099 | 0.7310 +/- 0.2060 | 0.7384 +/- 0.2122 | 0.6984 +/- 0.2019 | 0.7639 +/- 0.2002 |
| H6 | no-spec | 0.1775 +/- 0.0091 | 0.5535 +/- 0.0035 | 0.1987 +/- 0.0121 | 0.5656 +/- 0.0035 | 0.7437 +/- 0.2107 | 0.7483 +/- 0.2142 | 0.7164 +/- 0.2080 | 0.7639 +/- 0.2002 |

Interpretation: H2/H3 clearly dominate H0 on cumulative spectral metrics. H4 has the strongest spectral preservation but slightly lower task recovery than H2/H3. H6 confirms the spectral term is needed because its cumulative spectral metrics are much worse than H2/H3. Tuned full RGCN remains stronger on macro-F1, so the main claim should be quality/cost and preservation, not task-SOTA.

## B. Relation Fusion Final Ablation [GPU Task Eval]

Runs: ACM/DBLP/IMDB x 3 seeds x 4 variants = 36. Target hit rate is 1.0. Sampled eigen sanity exists for final spectral diagnostics, e.g. `status=sampled_subgraph`, `mode=sampled_dense_eigvalsh`.

| variant | setting | DEE | FSE | REEmax | SIPE | refined@5 | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2-full | heterogeneous relation fusion | 0.0678 +/- 0.0238 | 0.5335 +/- 0.0153 | 0.1125 +/- 0.0659 | 0.5374 +/- 0.0194 | 0.7443 +/- 0.2062 | 0.7490 +/- 0.2101 |
| H2-single-relation-sum | flatten/sum adjacency | 0.1795 +/- 0.1057 | 0.5989 +/- 0.0205 | 0.2596 +/- 0.0454 | 0.5792 +/- 0.0079 | 0.7487 +/- 0.2114 | 0.7491 +/- 0.2117 |
| H2-no-rel-term | fused sketch, lambda_rel=0 | 0.0632 +/- 0.0224 | 0.5323 +/- 0.0159 | 0.1101 +/- 0.0631 | 0.5367 +/- 0.0198 | 0.7431 +/- 0.2068 | 0.7480 +/- 0.2114 |
| H2-uniform-fused-only | pure fused spectral | 0.0267 +/- 0.0198 | 0.5309 +/- 0.0045 | n/a | 0.5238 +/- 0.0016 | 0.7252 +/- 0.2137 | 0.7290 +/- 0.2168 |

Interpretation: flatten/sum is the clean negative control for heterogeneous fusion. It gives similar task numbers but much worse DEE/FSE/REE/SIPE, especially on DBLP/IMDB. The relation profile term alone is not the main source of the gain; the heterogeneous fused operator is the stronger novelty point.

## C. OGBN-MAG Medium Protocol [GPU]

Protocol fixes are present in `task_summary.csv`: target node type is `paper`, type id is 3, split policy is `synthetic_stratified`, macro policy is `eval_present`, leakage check is `passed`, and coarse labels are train-only.

| subset | train/val/test labeled papers | classes train/val/test | label coverage | train-only coarse coverage |
| --- | ---: | ---: | ---: | ---: |
| 50k | 9619 / 3145 / 3496 | 348 / 341 / 344 | 1.0 / 1.0 / 1.0 | 0.252 |
| 200k | 54615 / 18109 / 18536 | 349 / 349 / 349 | 1.0 / 1.0 / 1.0 | 0.366 |

Task outcome remains weak on OGBN-MAG: macro-F1 is near zero for both coarsened and full-graph baselines because the subset has 348-349 classes and a sparse synthetic split. Micro-F1 is nonzero but low. Use these OGBN results as protocol/system evidence, not a task-quality claim.

| size | variant/mode | ratio | candidate sec | runtime sec | projected micro/macro | refined@5 micro/macro | full RGCN tuned macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | H2-fullcand/full | 0.5500 | 5.45 | 6.77 | 0.0649 / 0.0008 | 0.0838 / 0.0021 | 0.0067 |
| 50k | H2-opt/optimized | 0.5500 | 2.53 | 3.89 | 0.0655 / 0.0008 | 0.0792 / 0.0021 | 0.0067 |
| 50k | H3-opt/optimized | 0.5500 | 3.20 | 4.52 | 0.0672 / 0.0008 | 0.0844 / 0.0025 | 0.0067 |
| 50k | H4-opt/optimized | 0.5500 | 2.52 | 3.78 | 0.0669 / 0.0008 | 0.0890 / 0.0026 | 0.0067 |
| 50k | flatten-sum-opt | 0.5500 | 2.34 | 3.49 | 0.0578 / 0.0007 | 0.0850 / 0.0020 | 0.0067 |
| 200k | H2/full two-hop | 0.5500 | 95.96 | 107.75 | 0.0607 / 0.0006 | 0.0836 / 0.0016 | 0.0060 |
| 200k | H2/optimized | 0.5500 | 7.53 | 19.56 | 0.0575 / 0.0006 | 0.0824 / 0.0015 | 0.0060 |
| 200k | H3/optimized | 0.5500 | 7.21 | 18.66 | 0.0587 / 0.0006 | 0.0814 / 0.0015 | 0.0060 |
| 200k | H4/optimized | 0.5500 | 7.23 | 18.66 | 0.0557 / 0.0006 | 0.0829 / 0.0016 | 0.0060 |

The one-level OGBN medium protocol ends at ratio 0.55 because it uses the default per-level ratio. That is acceptable for this medium/scale protocol but should be stated as `ratio=0.55`, not exact `target_ratio=0.5`.

## D. Candidate Optimization

HGB one-seed ablation, averaged over ACM/DBLP/IMDB:

| candidate source | candidate sec | two-hop sec | coverage | DEE | refined@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| onehop_twohop_bucket | 5.25 | 4.73 | 0.968 | 0.067 | 0.743 |
| onehop_bucket_limited_twohop | 3.24 | 2.79 | 0.960 | 0.076 | 0.741 |
| onehop_bucket | 0.43 | 0.00 | 0.861 | 0.035 | 0.724 |
| bucket | 0.42 | 0.00 | 0.846 | 0.031 | 0.741 |
| onehop | 0.06 | 0.00 | 0.100 | 0.307 | 0.701 |

OGBN 200k candidate path:

| mode | ratio | candidate sec | two-hop sec | coverage | refined@5 micro/macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| full onehop+twohop+bucket | 0.5500 | 95.96 | 82.89 | 0.946 | 0.0836 / 0.0016 |
| optimized onehop+bucket | 0.5500 | 7.53 | 0.00 | 0.860 | 0.0824 / 0.0015 |
| limited two-hop | 0.5500 | 22.74 | 10.50 | 0.940 | 0.0836 / 0.0016 |
| bucket-only | 0.5500 | 5.80 | 0.00 | 0.785 | 0.0810 / 0.0015 |
| onehop-only | 0.7753 | 1.48 | 0.00 | 0.382 | 0.0870 / 0.0019 |

Interpretation: full two-hop is too costly at 200k. Optimized onehop+bucket is the practical default; limited two-hop is the fallback when coverage close to full mode is needed. Onehop-only fails the compression target and should stay a negative control.

## E. Scale Path [GPU, Local]

The 1M OGBN-MAG full-relation subset has 1,000,000 nodes and 13,273,095 edges. Local source graph has 1,939,743 nodes, so a true 5M-node subset cannot be generated from this local data source.

| variant | nodes -> coarse | candidate sec | total sec | candidate pairs/sec | coverage | peak RSS GB | peak VRAM allocated GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2-opt | 1,000,000 -> 550,000 | 91.46 | 282.26 | 22,545 | 0.895 | 2.25 | 0.022 |
| H3-opt | 1,000,000 -> 550,000 | 92.00 | 289.66 | 22,412 | 0.895 | 2.25 | 0.022 |
| H4-opt | 1,000,000 -> 550,000 | 91.93 | 283.82 | 22,430 | 0.895 | 2.25 | 0.022 |

Stage timing for 1M is roughly: sketch 32.6-32.9s, candidates 91.5-92.0s, scoring 11.8-12.4s, matching/aggregation 146.1-152.7s. Candidate generation is no longer the only bottleneck at 1M; matching/aggregation is now the largest stage.

## Full Graph / Sampling Baseline Fairness

HGB full graph baselines are included in the main 5-seed table: default RGCN, tuned RGCN, HAN-small, and HGT-lite. H2/H3 are competitive with default RGCN/HAN/HGT but not tuned RGCN.

OGBN full graph baselines were run on the same subset/task protocol for every medium run. Tuned full RGCN macro-F1 is around 0.006, HAN/HGT around 0.0015-0.0023. Since all task macro-F1 values are very low, the OGBN claim should be restricted to system scalability and protocol sanity.

## Final Claims For Paper Draft

- Freeze HGB main variants to H0/H2/H3/H4/H6. Do not reintroduce H1/H5 into the main table.
- Use H2 as task-balanced default and H3 as spectral-safe default.
- State that H2/H3 strongly improve cumulative spectral preservation over H0, while staying competitive with common full-graph baselines on HGB.
- Use H4 and H6 as mechanism ablations: H4 shows the cost of removing the task/convolution term; H6 shows the spectral term is necessary.
- Use relation fusion ablation to argue heterogeneous fused operator preservation. The flatten/sum control is clearly worse on spectral/operator metrics.
- Keep OGBN-MAG medium and 1M scale as system/protocol evidence, not task-SOTA evidence.
- Candidate generation optimization is justified: optimized onehop+bucket cuts 200k OGBN candidate time from 95.96s to 7.53s, and limited two-hop cuts it to 22.74s while recovering most coverage.

## Server Commands

No server fallback was required. If a future larger run exceeds local memory, use this command shape on a server with a larger GPU/RAM budget:

```powershell
conda activate pytorch
python -m experiments.scripts.run_ogbn_mag_next4_medium --input data/ogbn_mag_subsets_20260516/subset_1m_fullrels --output outputs/ogbn_mag_next5_scale_server --variants H2-opt H3-opt H4-opt --seeds 12345 --target-ratio 0.5 --max-levels 1 --device cuda --candidate-mode optimized
python -m experiments.scripts.summarize_next4 outputs/ogbn_mag_next5_scale_server --output outputs/ogbn_mag_next5_scale_server_summary
```
