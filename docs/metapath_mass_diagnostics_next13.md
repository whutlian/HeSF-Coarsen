# Next13 Path-Mass Metapath Diagnostics

## Purpose

Next12 metapath survival was not informative because typed and untyped step
survival were 1.0 for every method. Next13 replaces that check with path-mass
preservation: bounded terminal probes are propagated through typed schema paths
with sequential sparse relation transitions, then coarse/lifted path mass is
compared with original path mass.

## Constraints

The diagnostic intentionally avoids scalability traps:

- no dense adjacency matrices;
- no explicit relation-product materialization;
- no full two-hop or three-hop path enumeration;
- no large eigendecomposition;
- no full relation arrays moved to GPU.

## Outputs

Primary output root:

`outputs/exp_next13_metapath_mass_20260517_summary`

Paper-use files:

- `metapath_mass_by_method.csv`
- `metapath_mass_by_dataset.csv`
- `metapath_mass_by_schema_path.csv`
- `metapath_mass_gap_vs_flatten_h6.csv`
- `metapath_collapse_count_secondary.csv`
- `summary.md`

## Interpretation

The diagnostic is method-sensitive and no longer tautological. In the completed
Next13 run, HeSF-LVC-P/S improve path-mass error over H0, AH-UGC-style,
GraphZoom-style, ConvMatch-style, and random, but do not clearly beat
flatten-sum or H6-no-spec. Therefore the metric belongs in the appendix as a
diagnostic, not as a main claim.
