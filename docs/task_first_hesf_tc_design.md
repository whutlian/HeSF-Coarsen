# HeSF-TC Task-First Design

HeSF-TC is a new experimental branch. It does not modify the preservation-first HeSF-LVC-P/S mainline and does not reuse deprecated accuracy-branch proxy methods as method logic.

## Objective

The branch treats task behavior on target nodes as primary and uses spectrum as a target-conditioned regularizer:

```text
L_total = L_task
        + lambda_target_spec L_target_spec
        + lambda_rel_response L_rel_response
        + lambda_support_coverage L_support_coverage
        + lambda_support_purity L_support_purity
        + lambda_feat L_feat
```

In v1 the greedy matcher uses local support-node merge deltas for the regularizer terms. The implemented score is `score_task_first`, with optional p95 normalization over candidate-local deltas.

## Hard Constraints

Target nodes of `target_node_type` are never merged. The full assignment is block structured: target nodes are singleton identity rows and only support nodes may be assigned into non-singleton clusters. The constraint filter rejects target-target, target-support, cross-type, cross-partition, and high-JS support-purity merges.

## Implemented In V1

- `hesf_coarsen/task_first/` new package.
- Target-label seed matrix from train target nodes only.
- Sparse target-conditioned low-pass response bank.
- Target-relevant relation response regularizer.
- Anchor support-neighborhood coverage delta.
- One-hop train-label support class footprints.
- Mandatory JS threshold purity block.
- Support-only coarsening pipeline using existing `Assignment` and `coarsen_graph`.
- Strict `real_full_target_inference` protocol gate that rejects `lite` backbones.

## Deferred To V2

- Learned `L_task` optimization inside coarsening.
- Official or faithful HGNN training integration.
- Teacher/student distillation. No deterministic or random proxy teacher is used in this branch.
- Streaming task-first cluster matching for very large candidate sets.

## Difference From HeSF-LVC-P/S

HeSF-LVC-P/S preserves heterogeneous fused low-frequency geometry for the whole graph. HeSF-TC preserves target-conditioned support-to-target discriminative propagation. This means the compression domain is support nodes, while the prediction domain remains the full target-node set.
