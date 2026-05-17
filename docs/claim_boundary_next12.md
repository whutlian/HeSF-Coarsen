# Next12 Claim Boundary

## Supported Scope

Next12 supports a preservation-first positioning for HeSF-LVC-P and HeSF-LVC-S:

- HeSF-LVC-P and HeSF-LVC-S preserve fused-operator and relation/path structure better than flatten-sum and H6-no-spec on the audited HGB operator tables.
- Task recovery is competitive under 50% HGB compression, but it is not consistently dominant.
- The tuned AH-UGC-style baseline is a protocol-matched type-isolated hash/LSH baseline, not an official AH-UGC reproduction.
- OGBN-MAG aggregation results are system/profiling evidence only.

## Unsupported Scope

Do not claim:

- HeSF-LVC beats full tuned RGCN on task F1.
- HeSF-LVC dominates flatten-sum or H6-no-spec on task F1.
- `lambda_conv` or `lambda_rel` is the core contribution.
- spectral guard or source-aware guard is part of the main method.
- bounded metapath sanity alone proves the mechanism.
- OGBN-MAG proves task quality.

## Next12 Evidence Notes

The Next12 metapath retention protocol is method-sensitive because it maps shared sampled original typed paths through method-specific assignments and evaluates the resulting coarse path. In this run, typed exact and untyped survival were both 1.0 for all main methods, so the typed-vs-untyped gap is not the separating evidence. The useful metapath evidence is collapse/count distortion, especially `log_path_count_error`.

The structure-sensitive stress results should be described carefully. Structure-only stress gives high P/S win rates against flatten-sum and H6, but the mean deltas are near zero. Feature-mask and feature-noise settings do not establish stable task superiority.

A3 packed-key sort passed correctness and weight checks, but it did not meet the full-local adoption rule. Keep A0 as the default OGBN aggregation backend.
