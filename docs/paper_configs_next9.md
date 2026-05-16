# Next9 Paper Configs

Next9 freezes the main HGB paper configs around two HeSF-LVC variants:

- `configs/paper/hgb_hesf_lvc_p.yaml`: default Pareto knee, `lambda_spec=0.25`, `lambda_conv=0.0`, `lambda_rel=0.0`.
- `configs/paper/hgb_hesf_lvc_s.yaml`: spectral-safe variant, `lambda_spec=0.5`, `lambda_conv=0.0`, `lambda_rel=0.0`.
- `configs/paper/hgb_hesf_lvc_t_appendix.yaml`: appendix-only SIPE/REEmax-oriented variant, not a main default.

Baselines and controls:

- `configs/paper/hgb_h0_mutual_best.yaml`
- `configs/paper/hgb_h6_no_spec.yaml`
- `configs/paper/hgb_flatten_sum.yaml`
- `configs/paper/hgb_graphzoom_style.yaml`
- `configs/paper/hgb_convmatch_style.yaml`
- `configs/paper/hgb_random_target_matched.yaml`

Optional robustness configs:

- `configs/paper/hgb_hesf_lvc_p_spectral_guard.yaml`
- `configs/paper/hgb_hesf_lvc_s_spectral_guard.yaml`
- `configs/paper/hgb_hesf_lvc_p_sourceaware_auto.yaml`
- `configs/paper/hgb_hesf_lvc_s_sourceaware_auto.yaml`

System/protocol config:

- `configs/paper/ogbn_mag_next9_opt_aggregation.yaml`

The method story is relation-preserving heterogeneous fused operators, randomized low-pass sketches, and type-compatible small-cluster LVC. `lambda_conv` and `lambda_rel` are not core contributions in the main configs.
