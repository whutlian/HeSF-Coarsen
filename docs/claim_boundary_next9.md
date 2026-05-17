# Claim Boundary After Next9

Do claim:

- Relation-preserving heterogeneous fused operator preservation.
- Strong spectral/operator preservation vs target-matched coarse baselines.
- Competitive task recovery under compression.
- Quality-cost tradeoff.

Do not claim:

- Task-SOTA.
- Beating full tuned RGCN.
- Flatten-sum task dominance.
- `lambda_conv` as a core contribution.
- `lambda_rel` or a relation-profile scoring term as essential.
- OGBN-MAG task-quality validation.
- GPU-scale or billion-edge validation.

Required wording:

> HeSF-LVC-P/S are relation-preserving heterogeneous graph coarsening methods that strongly preserve heterogeneous fused operators under compression, while maintaining competitive task recovery. They do not dominate full-graph tuned RGCN or all task baselines on F1.

Novelty wording:

> relation-preserving heterogeneous fused operators + randomized low-pass sketches + type-compatible small-cluster LVC.

Paper-ready final claim:

> HeSF-LVC-P/S strongly preserve heterogeneous fused operator structure under 50% HGB coarsening. Compared with H0, random, GraphZoom-style, ConvMatch-style, flatten-sum, and H6-no-spec, P/S reduce operator and relation-structure distortion. Task recovery is competitive under compression, but full tuned RGCN remains stronger on pure task F1. flatten-sum and H6 can remain task-competitive after refinement, but their operator distortion is substantially larger. OGBN-MAG is used for scalability and profiling, not task-quality claims.

Method boundary:

- Main method: `HeSF-LVC-P`, default/Pareto variant, `lambda_spec=0.25`, `lambda_conv=0`, `lambda_rel=0`.
- Spectral-safe main variant: `HeSF-LVC-S`, `lambda_spec=0.5`, `lambda_conv=0`, `lambda_rel=0`.
- Appendix / future safeguards only: `lambda_conv`, `lambda_rel`, spectral guard, and source-aware guard.

Formula wording:

```text
core score = normalized Delta_spec
           + typed feature / boundary constraints

optional diagnostics / ablations:
           + lambda_conv Delta_conv
           + lambda_rel Delta_rel
```
