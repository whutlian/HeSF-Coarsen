# Next14 HeSF-LVC Experiment Plan

## Goal

Next14 is a paper-cleanup, objective-aligned diagnostic, and system-bottleneck
iteration. It does not add a new main method heuristic. The main variants stay
fixed:

| method | role | lambda_spec | lambda_conv | lambda_rel |
| --- | --- | ---: | ---: | ---: |
| HeSF-LVC-P | default / Pareto variant | 0.25 | 0.0 | 0.0 |
| HeSF-LVC-S | spectral-safe variant | 0.50 | 0.0 | 0.0 |

## Required Outputs

| block | output |
| --- | --- |
| P0 paper tables | `outputs/exp_next14_paper_tables_20260518_summary` |
| P1 held-out fused operator | `outputs/exp_next14_operator_holdout_20260518_summary` |
| P2 metapath appendix | `outputs/exp_next14_metapath_appendix_20260518_summary` |
| P3 TypedHash fairness | `outputs/exp_next14_typedhash_fair_baseline_20260518_summary` |
| P4 OGBN output/merge backend | `outputs/exp_next14_ogbn_output_merge_backend_20260518_summary` |
| P6 final report | `outputs/exp_next14_final_report_20260518.md` |

## Task Boundary

- P0 cleans paper-facing tables, explicit metric names, main-vs-appendix rows,
  and claim-safe wording.
- P1 replaces weak path-count claims with held-out fused-operator probes that
  share deterministic probes across methods for each dataset and seed.
- P2 keeps Next13 path-mass as appendix evidence only.
- P3 renames the old AH-UGC-style row into TypedHash rows and keeps validation
  selected and oracle rows out of main tables.
- P4 benchmarks output/merge backend labels against A0 and adopts none unless
  the full-local correctness, speedup, RSS, and timing residual gates all pass.
- P5 is optional and is not required unless P0-P4 are complete and a new
  objective-aligned probe is needed.

## Hard Constraints

- No dense full adjacency construction.
- No explicit relation-product adjacency materialization.
- No full two-hop neighborhood materialization.
- No large eigendecomposition.
- No full relation arrays moved to GPU.
- No task-SOTA claim.
- No claim that TypedHash-ChebHeat is official AH-UGC.
- No bare `DEE` column in paper-facing outputs.

## Verification

The required local verification commands are:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_next14_paper_tables.py -q
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_holdout_operator_probes.py -q
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_next14_metapath_position.py -q
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_next14_typedhash_baseline.py -q
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_aggregation_output_merge_backend.py tests/test_aggregation_exclusive_timing.py -q
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest -q
```
