# Next13 AH-UGC-Style Fair Baseline

## Scope

The AH-UGC-style baseline is a protocol-matched type-isolated hash/LSH baseline.
It is not an official AH-UGC reproduction and should not be named as one.

## Result Classes

| result_class | paper use |
| --- | --- |
| global_fixed | Main external baseline row |
| validation_selected_by_dataset | Diagnostic row |
| oracle_appendix_only | Appendix-only upper-bound row |

## Main Row

The paper-facing external baseline row is:

| field | value |
| --- | --- |
| method | AH-UGC-style tuned-global |
| hash_bits | 20 |
| bucket_topk | 4 |
| assignment_source | chebheat_sketch |
| target_hit_rate | 1.0 |

The mean best macro-F1 is approximately 0.7236. The relation and operator
preservation metrics remain substantially weaker than HeSF-LVC-P/S, which keeps
the claim boundary focused on preservation rather than task-F1 dominance.

## Outputs

Primary output root:

`outputs/exp_next13_ahugc_fair_baseline_20260517_summary`

Paper-use files:

- `ahugc_global_fixed.csv`
- `ahugc_validation_selected_by_dataset.csv`
- `ahugc_oracle_appendix_only.csv`
- `external_baseline_main_table.csv`
- `summary.md`
