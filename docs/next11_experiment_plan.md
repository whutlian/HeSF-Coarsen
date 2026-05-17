# Next11 Experiment Plan

Goal: close the remaining paper risks after Next10 without adding a new main method.

Execution order:

1. Audit DEE consistency and rename incompatible metric definitions.
2. Package the Next10 rebuttal evidence into paper-ready tables and figures.
3. Run HGB task stress tests for low-label, early-refine, cross-model, and relation-mask settings.
4. Convert guard ablation into appendix-only evidence.
5. Benchmark OGBN aggregation optimizer variants as system evidence only.
6. Add an AH-UGC-style protocol-matched external baseline.
7. Run bounded metapath sanity for appendix consideration.
8. Write the final claim boundary and table index.

Method boundary:

- Main: `HeSF-LVC-P` and `HeSF-LVC-S`.
- Appendix/future safeguards: `lambda_conv`, `lambda_rel`, spectral guard, source-aware guard.
- OGBN-MAG: scalability and aggregation profiling only, not task quality.

Executed output roots:

- `outputs/exp_next11_dee_consistency_20260517/`
- `outputs/exp_next11_hgb_rebuttal_paper_table_20260517/`
- `outputs/exp_next11_hgb_task_stress_20260517/`
- `outputs/exp_next11_guard_appendix_20260517/`
- `outputs/exp_next11_ogbn_aggregation_optimizer_20260517/`
- `outputs/exp_next11_hgb_external_baselines_20260517/`
- `outputs/exp_next11_bounded_metapath_sanity_20260517/`

Final documentation:

- `docs/claim_boundary_next11.md`
- `docs/paper_table_index_next11.md`
