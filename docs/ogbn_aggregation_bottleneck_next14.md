# Next14 OGBN Aggregation Bottleneck

## Scope

OGBN-MAG is system/profiling evidence only. It is not task-quality evidence for
the HGB claim.

## Backend Boundary

A0 remains the default aggregation backend unless another backend meets all
adoption gates:

- full-local speedup at least 1.25x for both HeSF-LVC-P and HeSF-LVC-S;
- correctness checks pass;
- edge-weight preservation checks pass;
- RSS increase is at most 0.2 GB versus A0;
- exclusive timing residual is at most 5%;
- coarse edge counts and weights match A0 within documented tolerance.

## Next14 Backends

| backend | role |
| --- | --- |
| A0_current_sort_reducer | baseline and default |
| A6_direct_relation_writer | output/merge experiment label |
| A7_parallel_relation_output_writer | output/merge experiment label |
| A8_shard_count_chunk_sweep | shard/chunk sweep experiment label |

The Next14 code keeps A0 valid and unchanged as the default. A6/A7/A8 are
reported only when their measured full-local results satisfy the adoption
criteria.

## Next14 Result

The completed local run covered 32/32 size/method/backend combinations. All
runs were available and passed edge-weight preservation checks. No A6/A7/A8
backend met the adoption gate because none reached full-local speedup >= 1.25x
for both HeSF-LVC-P and HeSF-LVC-S:

| backend | HeSF-LVC-P full-local speedup | HeSF-LVC-S full-local speedup | adopted |
| --- | ---: | ---: | --- |
| A6_direct_relation_writer | 1.060 | 0.988 | no |
| A7_parallel_relation_output_writer | 1.033 | 0.982 | no |
| A8_shard_count_chunk_sweep | 1.063 | 0.994 | no |

A0 remains the default. The system conclusion is that output/merge profiling is
useful, but this Next14 implementation does not justify adopting a new backend.

## Output

The backend benchmark summary is stored at:

`outputs/exp_next14_ogbn_output_merge_backend_20260518_summary`
