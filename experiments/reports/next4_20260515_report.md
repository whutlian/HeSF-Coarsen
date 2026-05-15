# Next4 HeSF-LVC Experiment Report

Date: 2026-05-15
Code/report base commit before final follow-up: bce1b9f
PyTorch environment: `conda` env `pytorch`, task eval run with `--device cpu`.

## Outputs

- Mainline runs: `outputs/exp_next4_mainline_full_20260515`
- Aggressive runs: `outputs/exp_next4_aggressive_full_20260515`
- Mainline summary: `outputs/exp_next4_mainline_full_20260515_summary`
- Aggressive summary: `outputs/exp_next4_aggressive_full_20260515_summary`
- Combined summary: `outputs/exp_next4_full_20260515_combined_summary`
- Baseline summaries: `outputs/exp_next4_mainline_full_20260515_baselines`, `outputs/exp_next4_aggressive_full_20260515_baselines`
- Refine curves: `outputs/exp_next4_mainline_full_20260515_refine_curve`, `outputs/exp_next4_aggressive_full_20260515_refine_curve`

## Method Status

- Default: HeSF-LVC with ChebHeat d16, uniform relation fusion, meta-path disabled, greedy_cluster small clusters, max_cluster_size=4, target_ratio=0.5, lambda_conv=0.5.
- Optional ablations retained: mutual_best, non-uniform relation weighting, meta-path, source quota diagnostics, 0.25 aggressive terminal guard.
- Primary quality metrics are cumulative spectral metrics. Final-level metrics are diagnostics only.
- Coarsening path remains CPU-only; CUDA was not used to validate sketch/scoring, so no GPU-scale system claim is made.

## Mainline Mean Results

| variant | runs | target_hit_rate | ratio | DEE | FSE | REEmax | SIPE | projected | refined@0 | refined@1 | refined@3 | refined@5 | best | AUC | full_graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | 9 | 1 | 0.5 | 0.2408 | 0.6298 | 0.2892 | 0.6244 | 0.487 | 0.6712 | 0.6299 | 0.6986 | 0.7214 | 0.723 | 0.6798 | 0.7484 |
| H1 | 9 | 1 | 0.5 | 0.05659 | 0.5419 | 0.107 | 0.5438 | 0.6089 | 0.7185 | 0.7055 | 0.7064 | 0.7455 | 0.7488 | 0.7151 | 0.7484 |
| H2 | 9 | 1 | 0.5 | 0.06782 | 0.5335 | 0.1125 | 0.5374 | 0.658 | 0.7271 | 0.6912 | 0.729 | 0.7523 | 0.7545 | 0.7221 | 0.7484 |
| H3 | 9 | 1 | 0.5 | 0.05188 | 0.534 | 0.1066 | 0.5362 | 0.6454 | 0.7221 | 0.6809 | 0.7205 | 0.7439 | 0.7494 | 0.7135 | 0.7484 |
| H4 | 9 | 1 | 0.5 | 0.0139 | 0.5416 | 0.09194 | 0.5339 | 0.5689 | 0.7034 | 0.6612 | 0.7098 | 0.7304 | 0.7396 | 0.6987 | 0.7484 |
| H5 | 9 | 1 | 0.5 | 0.0139 | 0.5416 | 0.09194 | 0.5339 | 0.5689 | 0.7034 | 0.6612 | 0.7098 | 0.7304 | 0.7396 | 0.6987 | 0.7484 |
| H6 | 9 | 1 | 0.5 | 0.176 | 0.5535 | 0.1975 | 0.5656 | 0.6986 | 0.731 | 0.6881 | 0.7228 | 0.7468 | 0.748 | 0.718 | 0.7484 |

## Aggressive 0.25 Mean Results

| variant | runs | target_hit_rate | ratio | DEE | FSE | REEmax | SIPE | refined@5 | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 6 | 1 | 0.25 | 0.4991 | 0.8215 | 0.5455 | 0.8209 | 0.727 | 0.7279 |
| A1 | 6 | 1 | 0.25 | 0.4412 | 0.8257 | 0.4974 | 0.8241 | 0.7327 | 0.7327 |
| A2 | 6 | 1 | 0.25 | 0.4991 | 0.8215 | 0.5455 | 0.8209 | 0.727 | 0.7279 |
| A3 | 6 | 1 | 0.25 | 0.499 | 0.8183 | 0.5429 | 0.817 | 0.7508 | 0.757 |
| A4 | 6 | 1 | 0.25 | 0.4506 | 0.8256 | 0.5079 | 0.8235 | 0.7393 | 0.7464 |

90 percent of H2 target=0.5 refined@5 is 0.6771. All A variants are above this threshold; A0 refined@5 is 0.727.

## Baseline Summary

| block | baseline | status | rows | target_hit_rate | ratio | DEE | FSE | REEmax | SIPE | refined@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2/H3 target=0.5 | convmatch_style | included | 18 | 1 | 0.5001 | 0.4941 | 0.7578 | 0.5332 | 0.7518 | 0.6837 |
| H2/H3 target=0.5 | graphzoom_style | included | 18 | 1 | 0.5001 | 0.4878 | 0.7563 | 0.5004 | 0.7509 | 0.7131 |
| H2/H3 target=0.5 | heavy_edge | failed target control | 18 | 0 | 0.9478 | 0.09552 | 0.06215 | 0.2922 | 0.05866 |  |
| H2/H3 target=0.5 | random | included | 18 | 1 | 0.5001 | 0.4926 | 0.7566 | 0.5095 | 0.7508 | 0.6632 |
| A0 target=0.25 | convmatch_style | included | 6 | 1 | 0.2501 | 0.7449 | 0.9402 | 0.777 | 0.938 | 0.4764 |
| A0 target=0.25 | graphzoom_style | included | 6 | 1 | 0.2501 | 0.7453 | 0.9403 | 0.7599 | 0.9381 | 0.6582 |
| A0 target=0.25 | heavy_edge | failed target control | 6 | 0 | 0.9478 | 0.09488 | 0.06212 | 0.292 | 0.0586 |  |
| A0 target=0.25 | random | included | 6 | 1 | 0.2501 | 0.7427 | 0.9399 | 0.7573 | 0.9376 | 0.4541 |

## Acceptance

| id | criterion | status | evidence |
| --- | --- | --- | --- |
| A | H2/H3 outperform H0 on cumulative spectral metrics | PASS | H2 DEE/FSE/REE/SIPE=0.06782/0.5335/0.1125/0.5374; H0=0.2408/0.6298/0.2892/0.6244 |
| B | H4/H5 no-conv establishes conv necessity | PASS | No-conv improves spectral DEE to 0.0139 but lowers task refined@5 to 0.7304 vs H2 0.7523; conv remains task-balanced, optional for spectral-only. |
| C | H6 no-spec worse than H2 on cumulative spectral metrics | PASS | H6 DEE/FSE/REE/SIPE=0.176/0.5535/0.1975/0.5656; worse than H2 on all four. |
| D | Baselines target-matched before comparison | PASS | random/graphzoom_style/convmatch_style included; heavy_edge marked failed target control. |
| E | Baseline task metrics not missing for target-hit baselines | PASS | Included baseline rows have projected/refined task metrics from the conda `pytorch` environment. |
| F | No unsupported claims | PASS | Report keeps meta-path, non-uniform relation weighting, 0.25 quality, and GPU validation out of main claims. |
| G | HeSF-LVC description | PASS | Main method is heterogeneous fused low-pass sketch + type-compatible small-cluster local variation coarsening + convolution-aware scoring. |

## Conclusions

- H2 is the task-balanced default: strong cumulative spectral improvement over H0 and best mean refined@5 among H0-H6.
- H3 is the spectral-safe conv variant: lower DEE/REE/SIPE than H2, with slightly lower refined@5.
- H4/H5 no-conv improve spectral preservation but reduce task quality, so the conv term remains part of the main task-balanced method and can be disabled for spectral-only ablation.
- H6 no-spec is worse than H2 on cumulative spectral metrics, confirming the spectral term is active.
- A0-A4 hit target 0.25, but cumulative spectral errors remain high. Terminal guards are useful diagnostics/future work, not a main contribution.
- Heavy-edge baseline still fails target control and is excluded from main comparison tables.
