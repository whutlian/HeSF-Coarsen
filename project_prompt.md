# Project Prompt for Codex — Implement HeSF-Coarsen Research Prototype

You are Codex. Implement a research prototype for **HeSF-Coarsen**, a scalable heterogeneous graph coarsening system based on heterogeneous spectral fusion, randomized low-pass spectral sketches, type-constrained local candidate generation, and bounded multi-level matching.

Read `agent.md` first and treat it as the project-wide constraint document. In particular, obey these hard constraints:

1. Do not materialize full two-hop neighborhoods.
2. Do not compute or store `A^2` or `A_r1 @ A_r2` as explicit adjacency.
3. Do not run large-scale eigendecomposition.
4. Do not move the full graph to GPU.
5. Do not merge nodes of different types by default.
6. Every candidate-generation path must enforce per-node and per-middle-node budgets.
7. Preserve node types, relation types, relation direction, and per-type features in the coarse graph.

---

## Goal

Build a working Python package and CLI that can:

1. load a heterogeneous graph from compact edge-list / array files;
2. build relation-aware normalized operators;
3. compute a randomized low-pass spectral sketch;
4. generate bounded local merge candidates;
5. score same-type candidate pairs;
6. perform greedy type-constrained matching;
7. construct a relation-preserving coarse heterogeneous graph;
8. run multiple coarsening levels until a target ratio is reached;
9. emit diagnostics on compression, schema preservation, candidate counts, and approximate spectral quality.

The first version should prioritize correctness, bounded complexity, and testability over peak performance.

---

## Expected Repository Layout

Create or adapt the repository to roughly follow this structure:

```text
hesf_coarsen/
  __init__.py
  config.py
  io/
    __init__.py
    edge_list.py
    memmap_csr.py
    schema.py
  ops/
    __init__.py
    relation_ops.py
    fused_operator.py
    normalization.py
  sketch/
    __init__.py
    random_probe.py
    lowpass.py
    simhash.py
  partition/
    __init__.py
    type_partition.py
  candidates/
    __init__.py
    onehop.py
    capped_twohop.py
    bucket.py
    bounded_heap.py
  scoring/
    __init__.py
    relation_profile.py
    conv_response.py
    merge_cost.py
  matching/
    __init__.py
    greedy.py
  coarsen/
    __init__.py
    assignment.py
    aggregate_edges.py
    multilevel.py
  eval/
    __init__.py
    diagnostics.py
    spectral.py
  cli/
    __init__.py
    main.py
  tests/
    test_tiny_hetero.py
    test_capped_twohop.py
    test_type_safe_matching.py
    test_edge_aggregation.py
configs/
  default.yaml
README.md
```

Use a different layout only if the existing repository already has a clear structure. Preserve all existing working tests.

---

## Core Data Model

Implement a minimal heterogeneous graph representation:

```python
@dataclass
class RelationSpec:
    relation_id: int
    name: str
    src_type: int
    dst_type: int

@dataclass
class HeteroGraph:
    num_nodes: int
    node_type: np.ndarray        # shape [num_nodes]
    relations: dict[int, RelationAdj]
    features: dict[int, np.ndarray] | None
    labels: np.ndarray | None
```

Each `RelationAdj` should store sparse directed edges using compact arrays:

```python
src: np.ndarray
dst: np.ndarray
weight: np.ndarray | None
src_type: int
dst_type: int
relation_id: int
```

For the first prototype, edge-list arrays are acceptable. Add CSR helpers for efficient relation-level message passing. For large-graph paths, design the API so arrays can later be replaced by memmap arrays.

---

## Configuration

Create `configs/default.yaml` with defaults similar to:

```yaml
seed: 12345
hardware:
  gpu: optional
  max_vram_gb: 24
  max_ram_gb: 256
coarsening:
  target_ratio: 0.1
  max_levels: 6
  per_level_ratio: 0.55
  same_type_only: true
  same_partition_only: true
sketch:
  dim: 32
  order: 5
  num_scales: 2
  dtype: float16
  probe: rademacher
  method: repeated_smoothing
fusion:
  relation_weighting: uniform
  include_metapath_filters: false
candidates:
  total_budget_K: 16
  twohop_budget_K2: 8
  middle_degree_cap_policy: p99
  per_middle_pair_cap: 64
  enable_onehop: true
  enable_capped_twohop: true
  enable_bucket: true
  enable_partition_ann: false
  simhash_bits: 16
scoring:
  lambda_spec: 1.0
  lambda_rel: 0.2
  lambda_feat: 0.1
  lambda_conv: 0.3
  lambda_boundary: 0.1
features:
  projected_dim: 32
output:
  dir: outputs/default_run
```

---

## Implementation Tasks

### Task 1 — Tiny graph loader and schema validation

Implement utilities to create and load tiny heterogeneous graphs for tests.

Include a synthetic graph generator with at least these node types:

```text
0 = user
1 = item
2 = tag
```

and relation types:

```text
0 = user -> item
1 = item -> user
2 = item -> tag
3 = tag -> item
4 = user -> user
```

Validation must check:

- every relation edge endpoint matches the relation schema;
- every node has exactly one node type;
- feature arrays, if present, match type counts or global node indexing;
- relation IDs are unique.

### Task 2 — Relation operator and fused smoothing

Implement relation-level normalized message passing.

For relation `r: s -> d`, implement:

```python
apply_relation(graph, relation_id, X_src, normalize=True) -> X_dst
```

Then implement a fused smoothing operator:

```python
apply_fused_smoothing(graph, X, relation_weights) -> X_smooth
```

where `X` is a global dense matrix `[num_nodes, q]`. This function must process relation by relation and accumulate messages into the destination rows. It must not construct a dense adjacency matrix.

### Task 3 — Randomized low-pass spectral sketch

Implement:

```python
compute_lowpass_sketch(graph, config) -> np.ndarray
```

Default method:

1. Generate a seeded Rademacher probe matrix `Omega` with shape `[n, q]`.
2. Repeatedly apply fused smoothing for `order` iterations.
3. Optionally concatenate or average selected scales.
4. Store the result as `float16` if configured.
5. Row-normalize or standardize the sketch if useful for candidate generation.

No eigenvectors.

### Task 4 — SimHash bucket generation

Implement:

```python
compute_simhash_buckets(Z, node_type, partition_id, bits, seed) -> np.ndarray
```

Buckets must be type-aware and partition-aware. Bucket candidates may only be proposed among nodes with the same type and same partition by default.

### Task 5 — Bounded candidate container

Implement a per-node bounded candidate structure.

For small and medium graphs, a Python heap implementation is acceptable. However, document that the large-scale path should use fixed-size array buffers.

API:

```python
class BoundedCandidateStore:
    def add(i: int, j: int, score: float, source: str) -> None
    def to_pairs() -> np.ndarray  # shape [num_pairs, 2 or more]
    def counts() -> np.ndarray
```

Requirements:

- never retain more than `K` candidates per node;
- ignore self-candidates;
- ignore cross-type candidates by default;
- deduplicate candidate pairs;
- track source counts: onehop, capped_twohop, bucket, fallback.

### Task 6 — One-hop candidate generator

Implement a simple bounded one-hop candidate generator.

It should propose same-type neighboring pairs only when relation schema allows same-type endpoints, or when configured to use selected direct relations.

### Task 7 — Capped two-hop meta-path candidate generator

Implement the most important safety-sensitive module:

```python
generate_capped_twohop_candidates(graph, Z, partition_id, config, store)
```

Use middle-node wedge generation:

```text
for each middle node u:
    collect eligible endpoint nodes L by allowed meta-path schema
    keep only same endpoint type
    keep only same partition unless disabled
    if len(L) exceeds degree cap, skip or sample
    emit at most P_max pairs from L
    insert into the bounded candidate store
```

The implementation must expose and enforce:

- `middle_degree_cap_policy`;
- `per_middle_pair_cap`;
- `twohop_budget_K2` or source-specific candidate cap if feasible;
- same-type endpoint filtering;
- partition filtering.

Add a unit test proving that a high-degree middle node does not emit `O(d^2)` pairs when the cap is active.

### Task 8 — Bucket candidate generator

Implement same-type, same-partition candidate proposals from SimHash buckets.

For large buckets, sample or cap pairs. Do not generate all bucket pairs if bucket size is large.

### Task 9 — Relation profile and convolution-response sketch

Implement relation-degree profiles:

```python
compute_relation_profiles(graph) -> np.ndarray
```

Implement convolution-response sketch:

```python
compute_conv_response_sketch(graph, H, relation_weights) -> np.ndarray
```

where:

```text
H = concat(Z, projected_features) if features exist else Z
```

### Task 10 — Merge cost scoring

Implement:

```python
score_candidate_pairs(graph, pairs, Z, relation_profiles, conv_sketch, features, config) -> np.ndarray
```

Use:

```text
cost = lambda_spec * spectral_distance
     + lambda_rel * relation_profile_distance
     + lambda_feat * feature_distance
     + lambda_conv * conv_response_distance
     + lambda_boundary * boundary_penalty
```

For the first version, use squared Euclidean distances and simple boundary flags.

### Task 11 — Greedy same-type matching

Implement:

```python
run_greedy_matching(graph, scored_pairs, config) -> Assignment
```

Requirements:

- sort pairs by ascending cost;
- merge a node at most once per level;
- same-type only;
- same-partition only by default;
- unmerged nodes become singleton supernodes;
- output assignment vector and supernode type array.

### Task 12 — Coarse graph aggregation

Implement:

```python
coarsen_graph(graph, assignment) -> HeteroGraph
```

For every relation edge:

```text
(src, dst, relation, weight) -> (assignment[src], assignment[dst], relation, weight)
```

Aggregate duplicate coarse edges by `(coarse_src, coarse_dst, relation_id)` using sum of weights.

Requirements:

- preserve relation IDs;
- preserve relation source/destination types;
- preserve total relation weight unless self-loop filtering is configured;
- aggregate features by mean or weighted mean;
- aggregate labels by majority vote or label distribution if labels exist.

### Task 13 — Multi-level pipeline

Implement:

```python
run_multilevel_coarsening(graph, config) -> list[LevelResult]
```

Each level should:

1. compute or update sketch;
2. partition nodes if needed;
3. generate bounded candidates;
4. score candidates;
5. run matching;
6. aggregate graph;
7. record diagnostics;
8. stop if target ratio or max levels is reached.

### Task 14 — Diagnostics and tests

Implement diagnostics:

- node count by type;
- edge count by relation;
- compression ratio;
- candidate count distribution;
- source contribution of candidates;
- number of matched pairs;
- singleton ratio;
- relation weight preservation;
- runtime by stage.

For small graphs, implement approximate energy diagnostics:

```text
DirichletEnergy(Z, L_F) before and after coarsening
```

using relation-level smoothing / Laplacian apply-functions, not dense matrices unless the graph is tiny.

Create tests for:

1. schema validation;
2. no cross-type merge;
3. capped two-hop does not explode on a hub;
4. candidate count per node <= K;
5. edge aggregation preserves relation weights;
6. multi-level pipeline runs on synthetic graph;
7. deterministic output under fixed seed.

---

## CLI Requirements

Implement a CLI with at least:

```bash
python -m hesf_coarsen.cli.main generate-synthetic --output data/tiny --num-users 1000 --num-items 500 --num-tags 100
python -m hesf_coarsen.cli.main coarsen --config configs/default.yaml --input data/tiny --output outputs/tiny_run
python -m hesf_coarsen.cli.main diagnose --input outputs/tiny_run/level_1
```

The CLI should print and save a JSON diagnostics file.

---

## README Requirements

Write a concise README explaining:

1. what HeSF-Coarsen does;
2. why it avoids full two-hop expansion and eigendecomposition;
3. how to run the synthetic example;
4. how to inspect diagnostics;
5. current limitations;
6. next steps for large-scale memmap and GPU acceleration.

---

## Important Algorithmic Guardrails

When implementing, actively avoid these failure modes:

### Failure mode 1 — hidden `O(d^2)` from two-hop generation

Any loop that generates pairs inside a neighbor list must enforce `per_middle_pair_cap` before emitting pairs. Large middle nodes must be skipped, sampled, or bucketed.

### Failure mode 2 — all-pairs bucket explosion

Large SimHash buckets must be sampled or capped. Do not emit all pairs from a large bucket.

### Failure mode 3 — cross-type merging

Merging a user with an item, paper with author, or any other cross-type pair is invalid unless an explicit config flag enables an experimental mode.

### Failure mode 4 — dense adjacency construction

Do not build `n x n` dense arrays. Dense arrays are allowed only for node sketches, features, or tiny test graphs.

### Failure mode 5 — candidate storage as Python objects at scale

The prototype may use Python heaps for tiny tests, but design APIs so the candidate store can later be replaced with fixed-size arrays or memmap buffers.

---

## Acceptance Criteria for First Working Version

The first acceptable implementation should satisfy:

1. `pytest` passes.
2. The synthetic graph pipeline runs end to end.
3. The output coarse graph has fewer nodes than the input graph.
4. No merged supernode contains multiple node types.
5. Relation IDs and relation schemas are preserved after coarsening.
6. Candidate counts are bounded by config.
7. A synthetic hub test proves capped two-hop generation does not emit quadratic candidates.
8. Diagnostics are saved as JSON.
9. README includes runnable commands.

---

## Suggested Implementation Order

Use this sequence:

1. Create package skeleton, config loader, and synthetic graph generator.
2. Implement schema validation and relation adjacency representation.
3. Implement fused smoothing and random sketch.
4. Implement bounded candidate store.
5. Implement one-hop and capped two-hop candidates.
6. Implement SimHash bucket candidates.
7. Implement relation profiles, convolution-response sketch, and merge scoring.
8. Implement greedy matching and assignment construction.
9. Implement edge aggregation and feature aggregation.
10. Implement multilevel pipeline.
11. Add diagnostics, CLI, tests, and README.
12. Run synthetic smoke tests and fix invariants.

Do not jump to CUDA or large-scale optimization before the above is correct.

---

## Final Output Expected From Codex

When finished, provide:

1. a summary of implemented modules;
2. commands to run tests;
3. commands to run a synthetic coarsening example;
4. known limitations;
5. next recommended engineering steps for 100M-node / billion-edge scale.

