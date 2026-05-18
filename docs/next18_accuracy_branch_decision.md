# Next18 Accuracy Branch Decision

Final verdict: `DROP_ENTIRE_ACCURACY_BRANCH`

## Evidence Used

- Eligible official/faithful real full-target rows: 0.
- Wins vs internal keep-target comparator: 0.
- Reason: No official or faithful real full-target inference rows are available.
- Stage 0 audit keeps only A1/A2 in serious scope and deprecates A3/A4/A5.
- Stage 1 separates `coarse_transfer`, `approx_full_target_adapter`, and `real_full_target_inference`.
- Stage 2 records that official SeHGNN/HETTREE/FreeHGC are not integrated locally.
- Stage 3 local A1/A2 rows are lite-adapter diagnostics only.
- Stage 4 literature alignment marks all direct comparisons as non-comparable.

## Carry Forward

- Carry forward: preservation-first HeSF-LVC-P/S mainline.
- Carry forward only as experimental utility: target-preserve support-only coarsening helper.
- Do not carry forward: Hybrid-B target selection, meta-reconstruction proxy, deterministic distillation proxy, or task-aligned score as method claims.