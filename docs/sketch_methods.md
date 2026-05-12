# Sketch Methods

HeSF-Coarsen supports two low-pass sketch methods.

## Lazy Smoothing

`sketch.method: lazy` is the fast baseline. It generates a random probe, repeatedly applies relation-weighted smoothing, averages the last configured scales, centers columns, and row-normalizes. The old `repeated_smoothing` name remains accepted as an alias.

## Chebyshev Heat-Kernel Sketch

`sketch.method: chebyshev_heat` applies a Chebyshev approximation to `exp(-t L_F)` for each configured heat time, then concatenates the components with optional meta-path sketch channels. The recurrence uses the normalized Laplacian scaling `lambda_max = 2`, so the scaled operator is `L_F - I = -S_F`.

Chebyshev coefficients are computed with numerical Chebyshev quadrature, not SciPy:

```text
theta_j = pi * (j + 0.5) / M
x_j = cos(theta_j)
lambda_j = x_j + 1
c_0 = mean(exp(-t lambda_j))
c_k = 2 * mean(exp(-t lambda_j) cos(k theta_j))
```

The recurrence and accumulation use `float32`; final output follows `sketch.dtype`.

## Relation-Weighted Fused Operator

The fused propagation operator is applied relation by relation:

```text
S_F H = sum_r alpha_r S_r H
L_F H = H - S_F H
```

No dense adjacency, fused sparse matrix, or relation-product matrix is built. By default `fusion.symmetric_relation_operator: true` applies each directed relation as a symmetric block operator by passing messages forward and backward. `fusion.reverse_relation_policy: include_all` keeps every relation. `drop_detected_reverse_for_spectral_operator` detects exact explicit reverse relation arrays, drops the higher relation id from the spectral operator, and renormalizes the remaining relation weights.

Relation weighting supports:

- `uniform`: equal normalized weights.
- `volume`: weights proportional to `(vol(r) + epsilon)^eta`.
- `inverse_energy`: weights proportional to `(vol(r) + epsilon)^eta / (E_r + epsilon)^gamma`.
- `feature_smoothness`: uses type-wise features when available, otherwise falls back to a random basis.

The energy estimator loops over relation edges, optionally sampled per relation, and uses normalized edge differences. It does not materialize relation matrices.

## Meta-Path Chained Sketch

`metapath_sketch.enabled: true` adds channels computed by chaining normalized relation steps:

```text
Z_P = S_step_l ... S_step_1 Omega
```

The implementation supports integer relation IDs and accepts either integer type IDs or type names for `start_type` / `end_type`. Type names are inferred from relation schema names such as `author__writes__paper` or `user_to_item`, and can also be supplied through `metapath_sketch.type_names`. The initial basis is nonzero only on `start_type`; after each step, rows outside the current type are zeroed. This preserves type restrictions and avoids constructing `A_r1 @ A_r2` or any two-hop candidate set. Guardrails cap paths and path length unless `allow_large_metapath_sketch` is enabled.

## Diagnostics

Each coarsening level writes sketch metadata into `diagnostics.json`:

- `sketch`: method, dimension, dtype, Chebyshev order, heat times, runtime, NaN/Inf counts, row norm stats.
- `fusion`: relation weighting method, normalized weights, weight stats, energy estimates, volume estimates.
- `metapath_sketch`: enabled flag, path count, per-path dimension, length, runtime, final nonzero rows.

## Config Examples

Use `configs/sketch_chebheat_metapath.yaml` for the default Chebyshev heat setup and `configs/sketch_chebheat_metapath_tiny.yaml` for a tiny synthetic smoke configuration with one `user-item-user` meta-path channel.

## Guardrails

Production code does not build dense adjacency, full `S_F`, relation-product adjacency, or eigendecompositions. Dense eigendecomposition is used only by tiny unit tests for validation. Torch acceleration remains limited to dense sketch or candidate-scoring blocks; graph structure stays CPU-resident.
