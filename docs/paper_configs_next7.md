# Next7 Paper Configs

These configs freeze the names used in the Next7/Next8 baseline-gap tables. The main paper
configuration is P/S, with `lambda_conv=0` and `lambda_rel=0`; the older T setting is retained
only as a SIPE/REEmax-oriented appendix variant, not as the task-balanced default.

| config | method | purpose |
| --- | --- | --- |
| `configs/paper/hgb_hesf_lvc_p.yaml` | HeSF-LVC-P | Pareto knee P, `lambda_spec=0.25`, `lambda_conv=0`, `lambda_rel=0` |
| `configs/paper/hgb_hesf_lvc_s.yaml` | HeSF-LVC-S | Pareto knee S, `lambda_spec=0.5`, `lambda_conv=0`, `lambda_rel=0` |
| `configs/paper/hgb_hesf_lvc_t.yaml` | HeSF-LVC-T | Appendix variant, `lambda_spec=2.0`, `lambda_conv=0.25`, `lambda_rel=0`; not the main claim |
| `configs/paper/hgb_h0_mutual_best.yaml` | H0-mutual-best | mutual-best two-node control |
| `configs/paper/hgb_h6_no_spec.yaml` | H6-no-spec | spectral-term ablation |
| `configs/paper/hgb_flatten_sum.yaml` | flatten-sum | flattened single-relation-sum control |
| `configs/paper/hgb_random_target_matched.yaml` | random | target-matched random baseline marker |
| `configs/paper/hgb_graphzoom_style.yaml` | GraphZoom-style | target-matched GraphZoom-style baseline marker |
| `configs/paper/hgb_convmatch_style.yaml` | ConvMatch-style | target-matched ConvMatch-style baseline marker |

All configs keep same-type and same-partition coarsening, Chebyshev-heat sketches, uniform relation fusion unless explicitly testing flattening, and bounded `onehop_twohop_bucket` candidates.

The novelty wording should emphasize relation-preserving heterogeneous fused operators and
operator preservation. It should not describe the relation-profile term or `lambda_conv` scoring
as the core contribution.

Tiny synthetic smoke tests load each config, override only size/runtime knobs, run one coarsening level, and assert the `paper` block plus scoring lambdas are echoed into diagnostics.
