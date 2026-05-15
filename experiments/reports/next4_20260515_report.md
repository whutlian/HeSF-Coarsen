# Next4 HeSF-LVC Experiment Report

Date: 2026-05-15
Environment: `conda` env `pytorch`
GPU-marked runs: task/refine eval and scale smoke used `--device cuda` on `NVIDIA GeForce RTX 5060 Ti`, PyTorch `2.8.0+cu129`.

## Outputs

- Main confirmation: `outputs/exp_next4_mainline_full_20260515`, summary `outputs/exp_next4_mainline_full_20260515_summary`
- Relation fusion ablation: `outputs/exp_next4_relation_fusion_20260515`, summary `outputs/exp_next4_relation_fusion_20260515_summary`
- Baselines: `outputs/exp_next4_mainline_full_20260515_baselines`, `outputs/exp_next4_aggressive_full_20260515_baselines`
- GPU refine curves: `outputs/exp_next4_mainline_full_20260515_refine_curve_gpu`, `outputs/exp_next4_relation_fusion_20260515_refine_curve_gpu`, `outputs/exp_next4_aggressive_full_20260515_refine_curve_gpu`
- GPU scale smoke: `outputs/ogbn_mag_H2_cuda_smoke_200k_scaleonly`, summary `outputs/ogbn_mag_H2_cuda_smoke_200k_scaleonly_summary`

## Method Status

- Default method remains HeSF-LVC: target ratio 0.5, ChebHeat d16/order 5, uniform fused relation operator, meta-path disabled, greedy clusters with max size 4, same type/partition, `lambda_spec=1.0`, `lambda_conv=0.5`.
- Main confirmation is now frozen to H0/H2/H3/H4/H6. H1 is not part of the main matrix. H5 was config-equivalent to H4 in the previous full run and is removed from the default run list.
- Retained but downgraded: meta-path optional only, inverse/capped relation weighting appendix only, lazy d32 appendix only, terminal guard/aggressive target 0.25 appendix only.
- Heavy-edge baseline remains excluded from main comparison unless it hits the target ratio; it produced ratio 0.9478 and is marked failed target control.

## Main Confirmation [GPU Task Eval]

| variant | role | runs | ratio | DEE | FSE | REEmax | SIPE | projected | refined@0 | refined@1 | refined@3 | refined@5 | best | AUC | full_graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | mutual_best baseline | 9 | 0.5000 | 0.2408 | 0.6298 | 0.2892 | 0.6244 | 0.4827 | 0.6533 | 0.6162 | 0.7007 | 0.7191 | 0.7261 | 0.6743 | 0.7482 |
| H2 | task-balanced default | 9 | 0.5000 | 0.0678 | 0.5335 | 0.1125 | 0.5374 | 0.6531 | 0.7210 | 0.6717 | 0.7149 | 0.7443 | 0.7490 | 0.7084 | 0.7482 |
| H3 | spectral-safe | 9 | 0.5000 | 0.0519 | 0.5340 | 0.1066 | 0.5362 | 0.6410 | 0.7185 | 0.7068 | 0.7167 | 0.7485 | 0.7490 | 0.7203 | 0.7482 |
| H4 | spectral-only/no-conv | 9 | 0.5000 | 0.0139 | 0.5416 | 0.0919 | 0.5339 | 0.5716 | 0.7064 | 0.6387 | 0.7162 | 0.7372 | 0.7420 | 0.6962 | 0.7482 |
| H6 | no-spec | 9 | 0.5000 | 0.1760 | 0.5535 | 0.1975 | 0.5656 | 0.6977 | 0.7307 | 0.6834 | 0.7295 | 0.7446 | 0.7508 | 0.7188 | 0.7482 |

Interpretation: H2/H3 both dominate H0 on cumulative spectral metrics. H3 is the safer spectral variant and has the best mean refined@5/AUC. H4 proves the conv term trades some spectral preservation for task recovery; its task drop versus H2 is mild but consistent. H6 is worse than H2 on all cumulative spectral metrics, so the spectral term is necessary.

## Per-Dataset Check [GPU Task Eval]

| dataset | variant | refined@5 | full_graph | delta_full | DEE | FSE | REEmax | SIPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | H0 | 0.8896 | 0.9356 | -0.0460 | 0.2444 | 0.6332 | 0.3381 | 0.6346 |
| ACM | H2 | 0.9075 | 0.9356 | -0.0281 | 0.0995 | 0.5537 | 0.1999 | 0.5631 |
| ACM | H3 | 0.9228 | 0.9356 | -0.0128 | 0.0730 | 0.5521 | 0.1960 | 0.5606 |
| ACM | H4 | 0.9024 | 0.9356 | -0.0332 | 0.0189 | 0.5456 | 0.1715 | 0.5472 |
| ACM | H6 | 0.9076 | 0.9356 | -0.0280 | 0.1661 | 0.5563 | 0.2075 | 0.5689 |
| DBLP | H0 | 0.8475 | 0.8608 | -0.0133 | 0.2567 | 0.6266 | 0.2928 | 0.6196 |
| DBLP | H2 | 0.8533 | 0.8608 | -0.0075 | 0.0518 | 0.5244 | 0.0642 | 0.5233 |
| DBLP | H3 | 0.8469 | 0.8608 | -0.0139 | 0.0428 | 0.5264 | 0.0612 | 0.5234 |
| DBLP | H4 | 0.8440 | 0.8608 | -0.0168 | 0.0159 | 0.5450 | 0.0484 | 0.5288 |
| DBLP | H6 | 0.8520 | 0.8608 | -0.0088 | 0.1751 | 0.5544 | 0.1816 | 0.5614 |
| IMDB | H0 | 0.4203 | 0.4483 | -0.0280 | 0.2214 | 0.6297 | 0.2368 | 0.6189 |
| IMDB | H2 | 0.4721 | 0.4483 | +0.0238 | 0.0522 | 0.5223 | 0.0734 | 0.5257 |
| IMDB | H3 | 0.4759 | 0.4483 | +0.0276 | 0.0399 | 0.5235 | 0.0625 | 0.5246 |
| IMDB | H4 | 0.4653 | 0.4483 | +0.0170 | 0.0069 | 0.5341 | 0.0559 | 0.5257 |
| IMDB | H6 | 0.4744 | 0.4483 | +0.0261 | 0.1868 | 0.5498 | 0.2034 | 0.5664 |

Checks: H2 beats H0 on all three datasets for refined@5 and all cumulative spectral metrics. H2 is near full graph on ACM/DBLP and above full graph on IMDB. H4's task drop is not concentrated in one dataset; the H4-vs-H2 gap is ACM -0.0051, DBLP -0.0093, IMDB -0.0068.

## Relation Fusion Ablation [GPU Task Eval]

| variant | setting | DEE | FSE | REEmax | SIPE | projected | refined@5 | best | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2-full | current H2 | 0.0678 | 0.5335 | 0.1125 | 0.5374 | 0.6531 | 0.7443 | 0.7490 | 0.7084 |
| H2-single-relation-sum | flatten/sum adjacency | 0.1795 | 0.5989 | 0.2596 | 0.5792 | 0.6136 | 0.7487 | 0.7491 | 0.7069 |
| H2-no-rel-term | fused sketch, `lambda_rel=0` | 0.0632 | 0.5323 | 0.1101 | 0.5367 | 0.6625 | 0.7431 | 0.7480 | 0.7127 |
| H2-uniform-fused-only | fused spectral only, no per-relation REE detail | 0.0267 | 0.5309 | n/a | 0.5238 | 0.5516 | 0.7252 | 0.7290 | 0.6891 |

The flatten/sum ablation substantially worsens spectral metrics, but its mean refined@5 is close to H2. The no-rel-term ablation is also close to H2. This supports keeping the relation-fusion ablation in the paper, but the novelty claim should be phrased around spectral/operator preservation, not only task accuracy.

## Aggressive Appendix [GPU Task Eval]

| dataset | variant | refined@5 | best | DEE | FSE |
| --- | --- | ---: | ---: | ---: | ---: |
| ACM | A0 | 0.9026 | 0.9136 | 0.4971 | 0.8131 |
| ACM | A1 | 0.9130 | 0.9161 | 0.4322 | 0.8175 |
| ACM | A3 | 0.9049 | 0.9234 | 0.4646 | 0.8032 |
| DBLP | A0 | 0.8359 | 0.8372 | 0.4919 | 0.8206 |
| DBLP | A1 | 0.8375 | 0.8375 | 0.4149 | 0.8247 |
| DBLP | A3 | 0.8513 | 0.8555 | 0.5357 | 0.8197 |
| IMDB | A0 | 0.4218 | 0.4225 | 0.5084 | 0.8309 |
| IMDB | A1 | 0.4206 | 0.4206 | 0.4765 | 0.8349 |
| IMDB | A3 | 0.4868 | 0.4868 | 0.4969 | 0.8319 |

A3's 0.25 task improvement is mainly IMDB-driven, with a smaller DBLP lift and no ACM lift. It should stay as appendix stress evidence, not a main result.

## Baseline Summary

| block | baseline | status | rows | target_hit_rate | ratio | DEE | FSE | REEmax | SIPE | refined@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2/H3 target=0.5 | random | included | 18 | 1 | 0.5001 | 0.4926 | 0.7566 | 0.5095 | 0.7508 | 0.6632 |
| H2/H3 target=0.5 | graphzoom_style | included | 18 | 1 | 0.5001 | 0.4878 | 0.7563 | 0.5004 | 0.7509 | 0.7131 |
| H2/H3 target=0.5 | convmatch_style | included | 18 | 1 | 0.5001 | 0.4941 | 0.7578 | 0.5332 | 0.7518 | 0.6837 |
| H2/H3 target=0.5 | heavy_edge | failed target control | 18 | 0 | 0.9478 | 0.0955 | 0.0622 | 0.2922 | 0.0587 | n/a |
| A0 target=0.25 | random | included | 6 | 1 | 0.2501 | 0.7427 | 0.9399 | 0.7573 | 0.9376 | 0.4541 |
| A0 target=0.25 | graphzoom_style | included | 6 | 1 | 0.2501 | 0.7453 | 0.9403 | 0.7599 | 0.9381 | 0.6582 |
| A0 target=0.25 | convmatch_style | included | 6 | 1 | 0.2501 | 0.7449 | 0.9402 | 0.7770 | 0.9380 | 0.4764 |
| A0 target=0.25 | heavy_edge | failed target control | 6 | 0 | 0.9478 | 0.0949 | 0.0621 | 0.2920 | 0.0586 | n/a |

## Scale/System Smoke [GPU]

Input: OGBN-MAG relation-aware subset, 200,000 nodes and 961,584 edges from `data/ogbn_mag_hesf`.

| run | device | cuda | nodes out | ratio | runtime | sketch | candidates | scoring | match+agg | pairs/sec | edges/sec | peak RSS GB | peak VRAM alloc/reserved GB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2-GPU scale smoke | cuda | true | 110,001 | 0.5500 | 107.48s | 2.57s | 95.53s | 2.14s | 7.25s | 521,393 | 132,698 | 0.94 | 0.0208 / 0.0430 |

This is an independent system smoke only; it does not enter the HGB ablation tables. CUDA allocator diagnostics and the report both mark this run as GPU.

## Conclusions

- Use H2 as the task-balanced default and H3 as the spectral-safe variant.
- Report the main table with projected, refined@0, refined@1, refined@3, refined@5, best, AUC, and full_graph, not only refined@5.
- Keep H4 and H6 in the main ablation; drop H5 from future runs because it was identical to H4.
- Keep relation-fusion ablation, but state the result conservatively: flatten/sum hurts spectral preservation, while task accuracy alone is not a decisive separation.
- Keep 0.25 aggressive and terminal-guard results in appendix/future-work framing.
- Scale/system evidence is now separate and GPU-marked; larger OGBN/Product/Papers runs remain future system work.
