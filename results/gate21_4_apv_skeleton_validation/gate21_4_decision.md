# Gate21.4 APV Skeleton Decision

## Official Unmodified SeHGNN Main Result
- APV skeleton method: `H6-APV-skeleton`
- APV success rows: `25`
- APV mean micro: `0.9481691999999999`
- APV mean macro: `0.9444910000000002`
- APV structural ratio: `0.198818942091278`

## Adapter/Deployment Result
- feature_adapter_byte50_pass: `False`
- feature_adapter_byte30_pass: `False`
- feature_adapter_accuracy_validated: `False`

## Unsupported Or Not Claimed
- generic coarse graph for arbitrary HGNNs is not claimed.
- weighted superedge in unmodified official SeHGNN is not claimed.
- raw HGB 20% storage is not claimed from structural ratio alone.

## Decision Flags
- `NATIVE_FULL_REPRO_PASS`
- `EXPORT_FULL_FIDELITY_PASS`
- `APV_SKELETON_CACHE_CLEAN_PASS`
- `APV_SKELETON_5X5_NOT_CONFIRMED`
- `STRUCTURAL_STORAGE20_NOT_VALIDATED`
- `RAW_HGB_BYTE50_FAIL`
- `RAW_HGB_BYTE30_FAIL`
- `FEATURE_ADAPTER_BYTE50_FAIL`
- `FEATURE_ADAPTER_BYTE30_FAIL`
- `FEATURE_ADAPTER_ACCURACY_NOT_VALIDATED`
- `PAPER_FEATURE_TERM_REDUNDANCY_NOT_VALIDATED`
- `DIRECTIONALITY_ABLATION_NOT_RUN`
- `PATHAWARE_V2_GAIN_NOT_VALIDATED`
- `WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN`
- `TARGET_ONLY_SCHEMA_STUB_DIAGNOSTIC_ONLY`
- `GENERIC_COARSE_GRAPH_NOT_VALIDATED`
