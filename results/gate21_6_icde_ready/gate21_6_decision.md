# Gate21.6 ICDE-Ready Decision

- native_full_micro: `0.9533802`
- native_full_macro: `0.9498198`

## Pass Flags
- `NATIVE_EXPORT_FIDELITY_PASS`
- `OFFICIAL_STRUCTURAL_APV12_PASS`
- `OFFICIAL_STRUCTURAL_APV16_PASS`
- `OFFICIAL_STRUCTURAL30_PASS`
- `OFFICIAL_STRUCTURAL20_PASS`
- `OFFICIAL_STRUCTURAL12_PASS`
- `ADAPTER_PACKAGE10_PASS`
- `ADAPTER_PACKAGE05_PASS`
- `FEATURE_ABLATION_SHAPE_SAFE_PASS`
- `COVERAGE_DIAGNOSTICS_PASS`
- `EXTERNAL_TP_BASELINES_READY`
- `CROSS_DATASET_AUTO_CHANNEL_READY`
- `ICDE_MAIN_TABLE_READY`
- `STORAGE_ONLY_BASELINES_READY`
- `SYSTEM_RESOURCE_TABLE_READY`

## Fail Or Not Ready Flags
- `RAW_HGB_BYTE50_PASS`
- `RAW_HGB_BYTE30_PASS`
- `CACHE_HYGIENE_PASS`
- `METAPATH_INTROSPECTION_PASS`
- `FREEHGC_TP_READY`
- `STANDARD_CONDENSATION_BASELINES_READY`

## Counts
```json
{
  "adapter_rows": 21,
  "coverage_rows": 4,
  "cross_dataset_failure_rows": 0,
  "cross_dataset_rows": 2,
  "external_tp_rows": 24,
  "feature_ablation_rows": 80,
  "metapath_rows": 45,
  "official_rows": 15,
  "standard_condensation_rows": 7,
  "storage_only_rows": 6,
  "system_resource_rows": 3
}
```

## Claim Boundaries
- Feature adapters are not official-unmodified SeHGNN main-table methods.
- Structural storage ratio, raw HGB text byte ratio, cache ratio, and adapter package ratio are separate metrics.
- Missing FreeHGC/HGCond dependencies are recorded as failure rows rather than silently skipped.
- Standard condensation and schema-preserving TP workload protocols are reported separately.
