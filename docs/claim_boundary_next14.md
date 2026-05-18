# Next14 Claim Boundary

## Main Claim

HeSF-LVC-P/S are preservation-first heterogeneous graph coarsening methods.
They strongly preserve typed fused-operator and relation-energy structure under
HGB coarsening while maintaining competitive task recovery under compression.

They are not task-F1 SOTA, do not beat full tuned RGCN on pure task F1, and do
not dominate flatten-sum or H6-no-spec on task F1.

## Supported Wording

- HeSF-LVC-P/S strongly preserve typed fused-operator and relation-energy
  structure under HGB coarsening.
- HeSF-LVC-P/S maintain competitive task recovery under compression.
- Flatten-sum and H6-no-spec can be task-competitive while damaging
  operator/relation preservation.
- TypedHash-ChebHeat is a strong protocol-matched type-isolated hash baseline;
  HeSF-LVC-P/S improve preservation and task recovery at higher coarsening
  cost.
- OGBN-MAG is system/profiling evidence only.

## Unsupported Wording

- HeSF-LVC beats full tuned RGCN.
- HeSF-LVC dominates flatten-sum or H6-no-spec on task F1.
- HeSF-LVC preserves metapath/path-mass better than flatten-sum/H6.
- TypedHash-ChebHeat is official AH-UGC.
- A4/A6/A7/A8 improves aggregation unless the adoption criteria are met.
- `lambda_conv`, `lambda_rel`, guard, or source-aware filtering is core.
- OGBN-MAG proves task quality.

## Paper Table Boundary

Paper-facing tables use explicit DEE names:

- `paper_final_dee`
- `resource_logged_cumulative_dee`
- `resource_logged_final_level_dee`

Bare `DEE` is ambiguous and should not appear in paper-facing table headers.
Validation-selected and oracle rows are appendix-only. Full tuned RGCN is a
task reference, not a coarsening oracle baseline.
