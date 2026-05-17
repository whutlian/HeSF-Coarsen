# Next11 Claim Boundary

This document is the paper-facing boundary after the Next11 experiments. It is
intended to keep the method claim aligned with the evidence that was actually
generated.

## Main Method

Use two named variants as the main method family:

| variant | role | lambda_spec | lambda_conv | lambda_rel |
| --- | --- | --- | --- | --- |
| HeSF-LVC-P | default / Pareto variant | 0.25 | 0 | 0 |
| HeSF-LVC-S | spectral-safe variant | 0.5 | 0 | 0 |

The main scoring formula should be described as:

```text
core score = normalized delta_spec
           + typed feature / boundary constraints

optional diagnostics / ablations:
           + lambda_conv * delta_conv
           + lambda_rel * delta_rel
```

Do not present `lambda_conv`, `lambda_rel`, spectral guard, or source-aware
guard as the core contribution. They belong in appendix ablations or future
safeguards.

## Supported Claims

- HeSF-LVC-P/S preserve heterogeneous relation/operator structure more strongly
  than flatten-sum and H6-no-spec on the Next11 rebuttal diagnostics.
- HeSF-LVC-P/S maintain competitive task recovery under HGB compression, but do
  not consistently dominate flatten-sum or H6-no-spec on task F1.
- The task value should be framed as preservation with competitive recovery, not
  as task-SOTA or task dominance.
- Full tuned RGCN can remain stronger on task F1; the paper should answer this
  with quality-cost tradeoff, not with a stronger accuracy claim.
- OGBN-MAG is system/protocol evidence for scaling and aggregation profiling,
  not task-quality evidence.
- The AH-UGC-style baseline is protocol matched and type isolated, but it is not
  an official AH-UGC reproduction.
- Guard experiments can be reported as appendix validation only. P-side spectral
  guard rows passed the strict task-drop threshold, while S-side rows did not.
- Bounded metapath sanity is appendix-only evidence that the sampled metapaths
  are relation-compatible and bounded on actual graph structure.

## Not Supported Claims

Do not write:

- HeSF-LVC beats full tuned RGCN.
- HeSF-LVC dominates flatten-sum or H6-no-spec on task F1.
- `lambda_conv` or `lambda_rel` is the core contribution.
- Source-aware filtering is universally beneficial.
- Spectral guard is part of the main method.
- OGBN-MAG proves task quality.
- Bare `DEE` is directly comparable across Next9 and Next10 outputs.
- The bounded metapath sanity proves a main-result mechanism.
- The current OGBN optimizer variants should replace A0 by default.

## Metric Naming

The Next11 DEE audit concluded `different_metric_renamed`. Paper-facing tables
must avoid bare `DEE` when comparing Next9 and Next10 sources. Use explicit
metric names:

- `paper_final_dee`
- `resource_logged_cumulative_dee`
- `resource_logged_final_level_dee`

## Safe Paper Wording

Recommended core claim:

```text
HeSF-LVC-P/S strongly preserve heterogeneous fused operator structure under
HGB compression. Compared with H0, random, GraphZoom-style, ConvMatch-style,
flatten-sum, H6-no-spec, and protocol-matched AH-UGC-style baselines, they
reduce relation drift and relation energy error while retaining competitive
refined task recovery. Full tuned RGCN remains a stronger task model in some
settings, so the result is best framed as a preservation-quality and
quality-cost tradeoff rather than task dominance.
```

