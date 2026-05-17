# Next13 Claim Boundary

## Main Claim

HeSF-LVC-P/S are preservation-first heterogeneous graph coarsening methods.
They preserve typed fused-operator and relation-energy structure better than
task-competitive negative controls while maintaining competitive task recovery.
They are not task-F1 SOTA and do not beat full tuned RGCN on pure task F1.

## Main Method Variants

| method | role | lambda_spec | lambda_conv | lambda_rel |
| --- | --- | ---: | ---: | ---: |
| HeSF-LVC-P | default / Pareto variant | 0.25 | 0.0 | 0.0 |
| HeSF-LVC-S | spectral-safe variant | 0.50 | 0.0 | 0.0 |

`lambda_conv`, `lambda_rel`, spectral guard, source-aware guard, and aggressive
target-ratio variants belong in appendix ablations or future safeguards.

## Supported Wording

- HeSF-LVC-P/S strongly preserve operator and relation-energy structure under
  HGB coarsening.
- HeSF-LVC-P/S are task-competitive under compression.
- Flatten-sum and H6-no-spec can remain task-competitive after refinement, but
  their operator preservation is substantially weaker.
- AH-UGC-style tuned-global is a protocol-matched external-style baseline, not
  an official AH-UGC reproduction.
- OGBN-MAG is used for scalability and aggregation profiling only.

## Unsupported Wording

- HeSF-LVC beats full tuned RGCN on task F1.
- HeSF-LVC dominates flatten-sum or H6-no-spec on task F1.
- `lambda_conv` or `lambda_rel` is a core contribution.
- Metapath survival alone proves relation-preserving quality.
- Source-aware filtering or guard variants are universally beneficial.
- OGBN-MAG proves task quality.

## Paper Table Boundary

Paper-facing tables should use explicit DEE names only:

- `paper_final_dee`
- `resource_logged_cumulative_dee`
- `resource_logged_final_level_dee`

Bare `DEE` is ambiguous and should stay out of paper-facing table headers.
