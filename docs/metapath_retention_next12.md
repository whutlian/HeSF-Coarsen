# Next12 Metapath Retention

## Why Next11 Was Non-Diagnostic

The Next11 bounded metapath sanity sampled relation-compatible paths in the original graph and checked only original-graph validity. Because it did not map the sampled path through each method's assignment or inspect the method-specific coarse graph, all coarsening methods could receive identical values within a dataset.

## Next12 Protocol

Next12 uses the same sampled original typed paths for every method at a fixed dataset, seed, schema path, and sample seed. For each sampled path:

```text
v0 -[r1]-> v1 -[r2]-> ... -[rk]-> vk
```

the evaluator maps original nodes to method-specific clusters:

```text
c_i = assignment[v_i]
```

It then checks whether each coarse step exists under the same typed relation sequence:

```text
(c_{i-1}, c_i) in coarse relation r_i
```

The evaluator also checks untyped coarse connectivity, endpoint/intermediate collapse, bounded path-count preservation, and available coarse edge weights.

## Typed vs Untyped Survival

Typed exact survival requires each coarse edge to preserve the same relation id as the sampled original path. Untyped survival ignores relation ids and checks whether any coarse edge connects the mapped cluster pair. This distinction is meant to expose cases where a method preserves coarse connectivity while losing relation semantics.

In the completed Next12 HGB run, typed exact survival and untyped survival were both 1.0 for the main methods, so the survival gap was 0.0. The diagnostic signal came from collapse/count metrics rather than typed-vs-untyped survival.

## Boundedness

The implementation uses sparse relation arrays and capped frontier expansion. It does not construct dense adjacency matrices, explicit relation-product matrices, full two-hop materializations, or large eigendecompositions. Path-count preservation stops at the configured cap and marks capped counts explicitly.

## Main Text vs Appendix

The metric is suitable for a main-text paper table as a method-sensitive collapse/count diagnostic. The typed-vs-untyped survival gap should be treated as non-separating in this run and can be discussed as an appendix diagnostic or limitation.

Key outputs:

- `outputs/exp_next12_metapath_retention_20260517_summary/paper_metapath_rebuttal_table.csv`
- `outputs/exp_next12_metapath_retention_20260517_summary/metapath_path_count_drift_table.csv`
- `outputs/exp_next12_metapath_retention_20260517_summary/metapath_endpoint_collapse_table.csv`
