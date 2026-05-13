# Coarsening

The multilevel pipeline keeps type constraints enabled by default:

```yaml
coarsening:
  same_type_only: true
  same_partition_only: true
  matching_method: mutual_best
```

Matched nodes must share type and partition, and unmatched nodes become singleton supernodes. Relation aggregation uses the chunked reducer in the main pipeline and preserves relation specs.

## Feature Aggregation

Coarse features are weighted means within each type-safe cluster:

```text
X_I^c = sum_{i in C_I} w_i X_i / sum_{i in C_I} w_i
```

Configure the weights with:

```yaml
coarsening:
  feature_aggregation: mean
```

Supported methods:

- `mean`: unweighted average by cluster count.
- `degree_weighted`: weights each node by total incident edge weight mass.
- `pagerank_weighted`: computes a matrix-free PageRank score on the symmetrized relation graph and uses it as the node weight.
- `custom_weight`: uses `coarsening.feature_aggregation_weights` or a NumPy `.npy` array at `coarsening.feature_aggregation_weight_path`.

If all weights in a cluster are zero, feature aggregation falls back to the unweighted mean for that cluster. Each level records the selected strategy under `feature_aggregation` in `diagnostics.json`.
