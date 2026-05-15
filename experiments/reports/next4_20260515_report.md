# Next4 HeSF-LVC Experiment Report

Date: 2026-05-15
Environment: local `conda` env `pytorch`, PyTorch `2.8.0+cu129`
GPU: `NVIDIA GeForce RTX 5060 Ti`; all task/refine and OGBN medium/scale runs below are GPU-marked.

## Outputs

- Main confirmation: `outputs/exp_next4_mainline_full_20260515`, summary `outputs/exp_next4_mainline_full_20260515_summary`
- Strong full-graph task baselines: `outputs/exp_next4_full_graph_strong_20260515_task_eval_gpu.csv`
- Relation fusion ablation: `outputs/exp_next4_relation_fusion_20260515`, summary `outputs/exp_next4_relation_fusion_20260515_summary`
- Target-matched baselines: `outputs/exp_next4_mainline_full_20260515_baselines`, `outputs/exp_next4_aggressive_full_20260515_baselines`
- OGBN-MAG 200k medium optimized: `outputs/ogbn_mag_next4_medium_optimized_20260515`, summary `outputs/ogbn_mag_next4_medium_optimized_20260515_summary`
- OGBN-MAG 200k full-candidate control: `outputs/ogbn_mag_next4_medium_fullcand_20260515`, summary `outputs/ogbn_mag_next4_medium_fullcand_20260515_summary`

## Method Status

- Default method: HeSF-LVC with ChebHeat d16/order 5, uniform relation fusion, meta-path disabled, type-compatible greedy clusters, max cluster size 4, `target_ratio=0.5`, `lambda_spec=1.0`, `lambda_conv=0.5`.
- Main matrix is frozen to H0/H2/H3/H4/H6. H1 and H5 are not part of the main table.
- Downgraded/appendix only: meta-path, inverse/capped relation weighting, lazy d32, terminal guard, target 0.25 aggressive stress.
- Heavy-edge is excluded from main baseline comparison unless target-hit; in these runs it stays around ratio 0.9478 and is marked failed target control.

## Main Confirmation [GPU Task Eval]

| variant | role | runs | ratio | DEE | FSE | REEmax | SIPE | projected | refined@0 | refined@1 | refined@3 | refined@5 | best | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | mutual_best baseline | 9 | 0.5000 | 0.2408 | 0.6298 | 0.2892 | 0.6244 | 0.4827 | 0.6533 | 0.6162 | 0.7007 | 0.7191 | 0.7261 | 0.6743 |
| H2 | task-balanced default | 9 | 0.5000 | 0.0678 | 0.5335 | 0.1125 | 0.5374 | 0.6531 | 0.7210 | 0.6717 | 0.7149 | 0.7443 | 0.7490 | 0.7084 |
| H3 | spectral-safe | 9 | 0.5000 | 0.0519 | 0.5340 | 0.1066 | 0.5362 | 0.6410 | 0.7185 | 0.7068 | 0.7167 | 0.7485 | 0.7490 | 0.7203 |
| H4 | no-conv / spectral-only | 9 | 0.5000 | 0.0139 | 0.5416 | 0.0919 | 0.5339 | 0.5716 | 0.7064 | 0.6387 | 0.7162 | 0.7372 | 0.7420 | 0.6962 |
| H6 | no-spec | 9 | 0.5000 | 0.1760 | 0.5535 | 0.1975 | 0.5656 | 0.6977 | 0.7307 | 0.6834 | 0.7295 | 0.7446 | 0.7508 | 0.7188 |

Interpretation: H2/H3 dominate H0 on cumulative spectral metrics. H4 shows the conv term trades spectral preservation for task recovery. H6 is worse than H2 on every cumulative spectral metric, so the spectral term remains necessary.

## Strong Full-Graph Baselines [GPU]

| variant | refined@5 | best refined | full RGCN default | full RGCN tuned | full HAN-small | full HGT-lite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | 0.7443 | 0.7490 | 0.7482 | 0.7627 | 0.7510 | 0.7493 |
| H3 | 0.7485 | 0.7490 | 0.7482 | 0.7627 | 0.7510 | 0.7492 |

H2/H3 still match or nearly match the default RGCN-lite and the HAN/HGT-lite baselines, but the tuned RGCN-lite is stronger by about 0.014-0.018 macro-F1. The paper claim should therefore say H2/H3 are competitive with common full-graph baselines, not that they beat the tuned full graph.

## Relation Fusion Ablation [GPU Task Eval]

| variant | setting | DEE | FSE | REEmax | SIPE | projected | refined@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2-full | current H2 | 0.0678 | 0.5335 | 0.1125 | 0.5374 | 0.6531 | 0.7443 |
| H2-single-relation-sum | flatten/sum adjacency | 0.1795 | 0.5989 | 0.2596 | 0.5792 | 0.6136 | 0.7487 |
| H2-no-rel-term | fused sketch, `lambda_rel=0` | 0.0632 | 0.5323 | 0.1101 | 0.5367 | 0.6625 | 0.7431 |
| H2-uniform-fused-only | fused spectral only | 0.0267 | 0.5309 | n/a | 0.5238 | 0.5516 | 0.7252 |

Flatten/sum hurts spectral/operator preservation substantially. Task accuracy alone is not a decisive separator, so the novelty argument should emphasize heterogeneous operator preservation.

## Target-Matched Baselines

| block | baseline | status | rows | target hit | ratio | DEE | FSE | REEmax | SIPE | refined@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2/H3 target=0.5 | random | included | 18 | 1.0 | 0.5001 | 0.4926 | 0.7566 | 0.5095 | 0.7508 | 0.6632 |
| H2/H3 target=0.5 | graphzoom_style | included | 18 | 1.0 | 0.5001 | 0.4878 | 0.7563 | 0.5004 | 0.7509 | 0.7131 |
| H2/H3 target=0.5 | convmatch_style | included | 18 | 1.0 | 0.5001 | 0.4941 | 0.7578 | 0.5332 | 0.7518 | 0.6837 |
| H2/H3 target=0.5 | heavy_edge | failed target control | 18 | 0.0 | 0.9478 | 0.0955 | 0.0622 | 0.2922 | 0.0587 | n/a |

## OGBN-MAG 200k Medium Scale [GPU]

Input: `data/ogbn_mag_subsets_h2_cuda_smoke/subset_200k`, 200,000 nodes and 961,584 edges.

| variant | candidate mode | ratio | candidate time | two-hop time | bucket emit | retained pairs | runtime | projected | refined@5 | full RGCN default | peak RSS GB | peak VRAM GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | full onehop+twohop+bucket | 0.5500 | 95.96s | 82.89s | 6.49s | 507,623 | 107.75s | 0.000132 | 0.000596 | 0.000069 | 0.94 | 0.0208 / 0.0430 |
| H2 | optimized onehop+bucket | 0.5500 | 7.53s | 0.00s | 5.44s | 322,022 | 19.56s | 0.000165 | 0.000698 | 0.000069 | 0.92 | 0.0208 / 0.0430 |
| H3 | optimized onehop+bucket | 0.5500 | 7.21s | 0.00s | 5.17s | 322,022 | 18.66s | 0.000164 | 0.000307 | 0.000069 | 0.92 | 0.0208 / 0.0430 |
| H4 | optimized onehop+bucket | 0.5500 | 7.23s | 0.00s | 5.24s | 322,022 | 18.66s | 0.000134 | 0.000553 | 0.000069 | 0.92 | 0.0208 / 0.0430 |

Candidate generation improved from 95.96s to 7.53s for H2 on the same 200k subset. The bottleneck is now explicitly localized: capped two-hop expansion was 82.89s. The optimized mode removes that low-yield source for the medium-scale system path; bucket coverage remains 0.7686 and total candidate coverage is 0.8604. No local OOM or VRAM failure occurred, so no server fallback command was needed.

## System Metrics Added

Diagnostics and summaries now expose:

- `candidate_generation_time`, `candidate_pairs_per_sec`
- candidate substage times: `onehop`, `incident_index_build`, `twohop_expansion`, `simhash`, `bucket_emit`, `fallback`, `store_finalize`
- bucket/source coverage, source generation counts, partition imbalance
- candidate buffer memory, peak RSS, peak VRAM allocated/reserved
- run-level resource summary fields for GPU/system reporting

## Conclusions

- Main paper table should use H0/H2/H3/H4/H6 only.
- H2 remains the default task-balanced method; H3 is the spectral-safe variant.
- H4 and H6 stay as necessary ablations.
- Relation fusion should be claimed conservatively around heterogeneous spectral/operator preservation.
- The full-graph tuned baseline is stronger, so H2/H3 should be positioned as competitive under coarsening plus refinement.
- OGBN-MAG medium-scale GPU smoke confirms the candidate bottleneck is explainable and optimizable; the optimized candidate path brings the candidate stage well below the requested 20-30s target.
