# Next14 TypedHash Baseline

## Scope

TypedHash-ChebHeat is a protocol-matched type-isolated hash baseline using
ChebHeat sketches. It is inspired by type-isolated hash/LSH coarsening ideas,
but it is not official AH-UGC.

## Naming

| row | paper use |
| --- | --- |
| TypedHash-ChebHeat tuned-global | main fixed global row |
| TypedHash-raw global | main fixed global row when available |
| TypedHash validation-selected-by-dataset | appendix-only |
| TypedHash oracle-max | appendix-only |

## Main Result Boundary

TypedHash-ChebHeat tuned-global is a strong external-style baseline:

- `hash_bits = 20`
- `bucket_topk = 4`
- `assignment_source = chebheat_sketch`
- `target_hit_rate = 1.0`
- mean best macro-F1 is about 0.7236 in the Next14 tables.

The main paper use is a fairness and cost-quality comparison. It should not be
used to claim an official AH-UGC result.

## Output

The cleaned TypedHash summary is stored at:

`outputs/exp_next14_typedhash_fair_baseline_20260518_summary`
