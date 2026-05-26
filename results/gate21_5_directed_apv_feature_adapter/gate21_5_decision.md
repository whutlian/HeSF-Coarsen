# Gate21.5 Directed APV + Feature Adapter Decision

## Best Official-Unmodified Structural Method
- method: `H6-dirskel-AP100-PA00-PV75-VP00-PTTP00`
- micro: `0.9310562000000001`
- macro: `0.9259472000000001`
- structural ratio: `0.11612983356793907`

## Best Feature Adapter Method
- method: `random_projection_dim64`
- micro: `0.9483804`
- effective byte ratio: `0.004462669911368823`

## Decision Flags
- `OFFICIAL_STRUCTURAL12_PASS`
- `ADAPTER_EFFECTIVE_BYTE10_PASS`
- `FEATURE_ADAPTER_NOT_OFFICIAL_UNMODIFIED`
- `WEIGHTED_EDGE_UNSUPPORTED_FOR_UNMODIFIED_SEHGNN`
- `GENERIC_COARSE_GRAPH_NOT_CLAIMED`
- `STRUCTURAL_RATIO_IS_NOT_RAW_HGB_BYTE_RATIO`

## Structural Outcome
- directed structural12 flag: `True`
- directed structural10 flag: `False`
- directed primary success: `False`
- directed strong success: `False`
- APV fallback structural20 success: `True`
- best official raw HGB text ratio: `0.5299044736341588`
- raw HGB text bytes remain above 50% for best official method: `True`

## Audit Status
- directed rows: `95/95` success
- feature adapter rows: `70/70` success
- feature channel rows: `324/324` success
- feature loader audit failures: `0`
- loaded relation audit failures: `0`
- cache sanity pass: `True`
- path-aware AP/PV pruning status: `diagnostic_only_no_success_rows`

## Feature Channel Answers
- directed APV raw PTTP00 micro: `0.9464786666666667`
- directed APV zero-paper PTTP00 micro: `0.9456573333333333`
- directed APV zero-target-author-only PTTP00 micro: `0.946244`
- directed APV zero-venue PTTP00 micro: `0.906103`
- directed APV zero-term PTTP00 micro: `0.9460093333333334`
- zero-paper with PTTP30 micro: `0.9461269999999999`
- zero-term with PTTP30 micro: `0.9460093333333334`
- pca-paper-128 loaded PTTP00 micro: `0.9478873333333333`
- random-projection-paper-128 loaded PTTP00 micro: `0.947418`

## Unsupported Or Not Claimed
- Generic coarse graph for arbitrary HGNNs is not claimed.
- Weighted superedges in unmodified official SeHGNN are not claimed.
- Feature adapter results are not official-unmodified SeHGNN main-table results.
- Structural ratio is not raw HGB byte ratio.
- Target-only schema-stub, if present, is diagnostic only.
