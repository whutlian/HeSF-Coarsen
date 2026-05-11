# agent.md — Codex Direction Constraints for HeSF-Coarsen

## Project Identity

This project implements **HeSF-Coarsen**: a scalable heterogeneous graph coarsening system based on heterogeneous spectral fusion, randomized low-pass spectral sketches, type-constrained local matching, and bounded candidate generation.

Target research positioning:

> Spectrum-preserving graph coarsening methods are theoretically strong but hard to scale; practical graph condensation methods scale better but often lack heterogeneous spectral structure. This project bridges the gap through a type-compatible heterogeneous spectral-fusion coarsening pipeline designed for single-machine billion-edge-class graphs under commodity GPU constraints.

Primary hardware target:

- **GPU**: single NVIDIA RTX 4090-class GPU, assume 24GB VRAM unless explicitly configured otherwise.
- **CPU RAM**: 256GB.
- **Storage**: NVMe scratch / mmap-compatible local storage.

The implementation must be designed so the full graph structure is **never loaded into GPU memory**. The GPU may be used for dense block operations, sketch blocks, candidate scoring, and coarse-graph training only.

---

## Non-Negotiable Design Constraints

### 1. Never materialize full two-hop neighborhoods or `A^2`

Do **not** implement full two-hop expansion such as:

```text
for i in V:
    for u in N(i):
        for j in N(u):
            emit(i, j)
```

without strict caps.

Do **not** materialize:

```text
A @ A
A_r1 @ A_r2
full two-hop neighbor list per node
full all-pairs candidate table
```

Allowed alternatives:

- degree-capped wedge sampling;
- per-middle-node pair emission cap;
- meta-path filtering with hub truncation;
- two-hop graph filtering sketch via repeated sparse-dense multiplication;
- SimHash / LSH buckets over spectral sketches;
- partition-local ANN or bucket search.

Every candidate path must enforce a per-node top-`K` budget.

Default candidate constraints:

```yaml
candidate_budget_total_K: 16       # acceptable range: 16-32
candidate_budget_twohop_K2: 8      # acceptable range: 4-12
middle_degree_cap_policy: p99      # relation/type-specific percentile
per_middle_pair_cap_Pmax: 64       # acceptable range: 32-128
partition_local_only: true
same_type_merge_only: true
```

### 2. Never do explicit large-scale eigendecomposition

Do **not** compute top-`k` Laplacian eigenvectors for large graphs.

Allowed spectral approximations:

- Chebyshev low-pass filtering;
- block-Krylov / Lanczos-style filtering only if streaming-safe;
- heat-kernel-style low-pass approximation;
- random sketching using Rademacher or Gaussian probes.

Default spectral sketch:

```yaml
sketch_dim_q: 32
chebyshev_order_p: 5
num_filter_scales: 2
sketch_dtype: float16
random_probe: rademacher
```

### 3. Preserve heterogeneous schema

Coarsening must be **type-compatible**.

Only merge nodes with the same node type unless a future experiment explicitly enables otherwise.

For relation `r: source_type -> destination_type`, the coarse relation must be constructed as:

```text
A_c[r] = P_source.T @ A[r] @ P_destination
```

Implementation must preserve:

- node type IDs;
- edge / relation type IDs;
- relation direction;
- per-type feature spaces;
- optional labels and train/validation/test masks.

### 4. Candidate generation must be local, bounded, and streamable

Candidate generation must satisfy:

```text
Total retained candidates <= O(n * K)
```

and preferably operate in block-local buffers that are discarded after local matching.

Allowed candidate sources:

1. one-hop local candidates;
2. degree-capped meta-path two-hop candidates;
3. same-type spectral sketch buckets;
4. same-partition ANN / bucket candidates;
5. boundary repair candidates after local matching.

Forbidden candidate sources:

- global all-pairs search;
- global exact kNN over all nodes;
- full two-hop materialization;
- Python object-per-edge or object-per-candidate storage at scale.

### 5. Use streaming / mmap-friendly data structures

Avoid in-memory Python lists of edges for large graphs.

Use compact array-based formats:

```text
rowptr: int64
colidx: int32 or int64 depending on n
edge_weight: float16 or float32
relation_id: uint16 or uint32
node_type: uint16 or uint32
assignment: int32 or int64 depending on coarse size
```

For large data, prefer:

- NumPy memmap;
- PyArrow / Parquet for chunked edge lists;
- CSR / CSC stored as raw arrays;
- chunked sort-reduce for coarse edge aggregation.

### 6. Keep GPU usage block-local

The full graph must not be transferred to GPU.

GPU may be used for:

- dense sketch blocks `Z_block`;
- block SpMM if the block fits;
- candidate pair scoring;
- SimHash / bucket scoring;
- local ANN inside a partition;
- coarse-graph GNN training.

VRAM guardrail:

```text
No single tensor allocation should exceed 50% of configured VRAM unless explicitly justified.
```

For a 24GB RTX 4090, keep working tensors comfortably below 12GB.

### 7. Determinism and reproducibility

All randomized components must accept a seed:

- random probes;
- candidate sampling;
- bucket tie-breaking;
- partition shuffling;
- greedy matching order.

Every CLI command should log:

- random seed;
- input graph statistics;
- node / edge type counts;
- sketch parameters;
- candidate parameters;
- compression ratio by level;
- peak RAM if available;
- peak VRAM if available;
- wall-clock time by stage.

---

## Core Mathematical Objects

### Heterogeneous graph

The input graph is:

```text
G = (V, E, tau, rho, {X_t})
```

where:

- `tau(v)` is node type;
- `rho(e)` is relation type;
- each relation `r: s -> d` has sparse adjacency `A_r`;
- features may be type-specific.

### Relation-normalized operator

For relation `r`, use normalized message passing:

```text
S_r = D_src^{-1/2} A_r D_dst^{-1/2}
```

or a symmetric block operator when a PSD Laplacian is needed.

Do not explicitly construct huge block matrices. Implement apply-functions:

```python
apply_relation(r, X_src) -> X_dst
apply_relation_transpose(r, X_dst) -> X_src
apply_fused_operator(X) -> X
```

### Fused heterogeneous Laplacian / smoothing operator

Use relation weights:

```text
L_F = sum_r alpha_r L_r + sum_m beta_m L_m
```

or equivalently a smoothing operator built from weighted normalized relation operators.

Default relation weights can be uniform. Optional reliability weighting:

```text
alpha_r proportional to volume(r)^eta / (estimated_energy_r + epsilon)
```

Meta-path terms must be applied implicitly through repeated SpMM, never materialized as dense or explicit `A_r1 A_r2` adjacency.

### Low-pass spectral sketch

Compute:

```text
Z = g(L_F) Omega
```

where `Omega` is a random probe matrix and `g` is a low-pass filter approximated with Chebyshev recurrence or repeated smoothing.

Default implementation may start with repeated smoothing:

```text
Z_0 = Omega
Z_{k+1} = normalized_fused_smoothing(Z_k)
Z = concat_or_average({Z_k at selected scales})
```

A later implementation can replace this with a Chebyshev heat-kernel approximation.

---

## Candidate Generation Policy

Candidate generation must be implemented as a set of bounded proposal modules.

### Candidate source 1: one-hop candidates

For each relation, propose same-type pairs when schema permits. For bipartite relations, one-hop may not produce same-type pairs; use it mainly for direct same-type relations or as boundary / relation-profile evidence.

### Candidate source 2: capped meta-path two-hop candidates

Implement by middle-node wedge generation:

```text
for middle node u:
    L = eligible neighbors around u with target endpoint type t
    L = filter_same_partition(L)
    L = optionally_filter_by_degree_and_relation(L)

    if degree(u) > D_max:
        skip or sample with strong down-weight

    emit at most P_max candidate pairs from L
```

The cost must be bounded by:

```text
sum_u min(deg(u)^2, P_max)
```

not by unbounded `sum_u deg(u)^2`.

### Candidate source 3: spectral sketch buckets

Use `Z` to compute SimHash / LSH keys:

```text
bucket_id = sign(R @ Z_i)
```

Only sample candidates within:

- same node type;
- same partition;
- same or nearby bucket.

### Candidate source 4: partition-local ANN / bucket fallback

Use only when a node has too few candidates.

Do not perform global exact ANN over all nodes. ANN must be restricted to partition and node type.

### Candidate retention

Each node maintains a bounded candidate heap / fixed-size candidate array:

```text
candidate_ids[node, K]
candidate_pre_scores[node, K]
```

For large graphs, use block-local buffers and flush only retained candidates.

---

## Merge Scoring Objective

For same-type nodes `i, j`, compute a merge cost:

```text
cost(i, j) =
    lambda_spec     * delta_spec(i, j)
  + lambda_rel      * delta_rel(i, j)
  + lambda_feat     * delta_feat(i, j)
  + lambda_conv     * delta_conv(i, j)
  + lambda_boundary * delta_boundary(i, j)
```

### Spectral term

```text
delta_spec(i, j) = vol_i * vol_j / (vol_i + vol_j) * ||Z_i - Z_j||_2^2
```

### Relation-profile term

For each node, build a relation-degree profile:

```text
p_i[r] = deg_r(i) / deg_total(i)
```

Use Jensen-Shannon divergence or squared distance.

### Feature term

If features are available, project per-type features to a compact dimension first:

```text
phi_t(X_t) -> R^{d_proj}, default d_proj = 32 or 64
```

Do not keep full high-dimensional features in the coarsening core when scale is large.

### Convolution-response term

Use a lightweight graph-convolution sketch:

```text
C = sum_r alpha_r S_r H
H = concat(Z, projected_features)
```

Then:

```text
delta_conv(i, j) = ||C_i - C_j||_2^2
```

### Boundary penalty

Penalize merging partition-boundary nodes or hub-like nodes unless explicitly configured.

---

## Matching / Clustering Policy

Default merge operator:

- greedy maximal matching by ascending cost;
- same-type only;
- same-partition by default;
- optional small clusters with maximum cluster size, but matching pairs are the safest first implementation.

Every level produces an assignment vector:

```text
assignment_level_l[node_id] -> supernode_id
```

The multi-level assignment chain must be stored compactly.

---

## Coarse Graph Construction

For each original or current-level edge:

```text
(src, dst, relation, weight)
    ->
(super_src, super_dst, relation, weight)
```

Then aggregate by key:

```text
(super_src, super_dst, relation)
```

using chunked sort-reduce or hash aggregation with memory caps.

Self-loops:

- keep relation-specific self-loops if configured for GNN training;
- optionally exclude or fold them into degree for spectral diagnostics.

Coarse features:

```text
X_c[supernode] = weighted_average({X_i in cluster})
```

Coarse labels:

- majority vote for classification;
- label distribution if soft labels are needed;
- track cluster label entropy for diagnostics.

---

## Evaluation Requirements

At minimum, implement these diagnostics:

### Structural / compression diagnostics

- original and coarse node counts by type;
- original and coarse edge counts by relation;
- compression ratio by level;
- relation distribution drift;
- degree distribution summary;
- number of singleton nodes;
- average and max cluster size.

### Spectral diagnostics

For small graphs or sampled subgraphs:

- approximate Dirichlet energy preservation;
- sketch inner-product preservation;
- relation-wise energy error.

Do not require full eigendecomposition for large graphs.

### Task diagnostics, optional first version

- node classification transfer from coarse graph to original graph;
- link prediction / recommendation metrics if corresponding data exists;
- runtime and memory by stage.

---

## Repository Structure Guidance

Preferred package layout:

```text
hesf_coarsen/
  __init__.py
  config.py
  io/
    edge_list.py
    memmap_csr.py
    schema.py
  ops/
    relation_ops.py
    fused_operator.py
    normalization.py
  sketch/
    random_probe.py
    lowpass.py
    simhash.py
  partition/
    type_partition.py
  candidates/
    onehop.py
    capped_twohop.py
    bucket.py
    ann_local.py
    bounded_heap.py
  scoring/
    merge_cost.py
    relation_profile.py
    conv_response.py
  matching/
    greedy.py
  coarsen/
    assignment.py
    aggregate_edges.py
    multilevel.py
  eval/
    diagnostics.py
    spectral.py
  cli/
    main.py
  tests/
```

CLI examples:

```bash
python -m hesf_coarsen.cli.main sketch --config configs/default.yaml
python -m hesf_coarsen.cli.main coarsen --config configs/default.yaml
python -m hesf_coarsen.cli.main diagnose --graph outputs/coarse_level_3
```

---

## Coding Standards

- Prefer clear, typed Python for the first prototype.
- Use NumPy / PyTorch for dense block operations.
- Use SciPy sparse only for small and medium graphs; for large graphs, use custom array-backed CSR / memmap.
- Avoid per-edge Python objects.
- Avoid nested unbounded loops over neighbors.
- Every expensive routine must expose budget parameters.
- Every stage should be unit-testable on tiny heterogeneous graphs.
- Include docstrings explaining asymptotic cost and memory behavior.
- Fail loudly if a requested operation would materialize a forbidden object such as full `A^2`.

---

## Correctness Invariants

Every implementation should check:

1. No cross-type merges unless explicitly enabled.
2. Every original node maps to exactly one supernode at each level.
3. Coarse edge relation IDs are preserved.
4. Coarse node type equals source node type of its cluster.
5. Candidate count per node never exceeds configured `K` after retention.
6. Two-hop generator never emits more than configured per-middle cap unless explicitly disabled for tiny tests.
7. Compression ratio is monotonic across levels.
8. Aggregated edge weight sum is preserved per relation unless self-loop filtering is enabled.
9. Random seed controls all stochastic decisions.

---

## What Not To Optimize Prematurely

Do not start with a highly optimized CUDA kernel.

First deliver a correct, bounded, testable CPU / NumPy / PyTorch prototype that works on:

1. a synthetic tiny heterogeneous graph;
2. a medium graph with millions of edges if available;
3. a stress-test synthetic graph with power-law degree distribution.

Only after the algorithmic budgets are verified should GPU acceleration, C++ extension, Rust sort-reduce, or CUDA kernels be introduced.

---

## Research Narrative To Preserve

Keep the implementation aligned with this narrative:

1. Build a heterogeneous fused spectral operator without flattening away schema.
2. Approximate low-frequency geometry using randomized graph filtering, not eigendecomposition.
3. Generate only local, bounded, streamable candidates.
4. Merge only same-type nodes using spectral, relation, feature, convolution, and boundary-aware costs.
5. Construct the coarse heterogeneous graph by relation-preserving aggregation.
6. Evaluate not only downstream accuracy, but also spectral / energy preservation and resource use.

