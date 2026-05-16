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
