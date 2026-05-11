# HeSF-Coarsen

HeSF-Coarsen is a NumPy-first research prototype for heterogeneous graph coarsening. It preserves node types, relation IDs, relation directions, and per-type features while using randomized low-pass spectral sketches and bounded local candidate generation.

The prototype avoids the main scalability traps by design:

- no dense adjacency matrix construction;
- no explicit `A^2` or relation product adjacency;
- no full two-hop neighborhood materialization;
- no large eigendecomposition;
- no full-graph GPU transfer.

## Environment

Use the local conda environment requested for this workspace:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest
```

Torch/CUDA are available in that environment. The default coarsening core keeps graph structure on CPU with NumPy; optional Torch support is limited to dense sketch/feature blocks and never moves full relation arrays to GPU.

## Synthetic Example

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main generate-synthetic --output data/tiny --num-users 1000 --num-items 500 --num-tags 100
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main coarsen --config configs/default.yaml --input data/tiny --output outputs/tiny_run --progress
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main diagnose --input outputs/tiny_run/level_1
```

Each graph directory uses `schema.json`, `nodes.npz`, one `relation_<id>.npz` per relation, optional per-type feature arrays, and `diagnostics.json` for coarsened levels.

## Progress Feedback

Progress is disabled by default in the library config and writes only to stderr when enabled, so the final CLI JSON remains on stdout. Enable it from the command line:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main coarsen --config configs/default.yaml --input data/tiny --output outputs/tiny_run --progress
```

For server logs, use the plain backend and combine stderr/stdout into a log:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main coarsen --config configs/default.yaml --input data/ogbn_mag_hesf --output outputs/ogbn_mag_full --progress --progress-backend plain 2>&1 | Tee-Object outputs/ogbn_mag_full.log
```

The equivalent YAML config is:

```yaml
progress:
  enabled: true
  backend: auto
  min_interval_seconds: 1.0
```

`backend: auto` uses `tqdm` if it is installed and otherwise falls back to plain progress lines.

## Real Dataset Imports

Import HGB datasets through PyG. Use `--root data` so local `data/acm` and `data/dblp` caches are reused when present:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main import-hgb --name ACM --root data --output data/acm_hesf
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main import-hgb --name DBLP --root data --output data/dblp_hesf
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main import-hgb --name IMDB --root data --output data/imdb_hesf
```

Import OGB MAG and optionally export a memmap copy for large-graph experiments:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main import-ogbn-mag --root data/ogb --output data/ogbn_mag_hesf
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main export-memmap --input data/ogbn_mag_hesf --output data/ogbn_mag_mmap --chunk-size 1000000
```

Current imported dataset directories:

- `data/acm_hesf`
- `data/dblp_hesf`
- `data/imdb_hesf`
- `data/ogbn_mag_hesf`
- `data/ogbn_mag_mmap`

## Explicit Large-Graph Utilities

The memmap and chunked paths are explicit CLI utilities. They are not used by the default small/medium pipeline.

Export a graph directory to mmap-loadable `.npy` arrays:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main export-memmap --input data/tiny --output data/tiny_mmap --chunk-size 1000000
```

Run chunked edge aggregation from a saved assignment file:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m hesf_coarsen.cli.main chunked-aggregate --input data/tiny_mmap --memmap-input --assignment assignment.npz --output outputs/chunked_level --chunk-size 1000000 --reducer sort
```

The assignment file is an `.npz` with `assignment` and `supernode_type` arrays.

`--reducer sort` is the default large-graph reducer. It performs vectorized per-chunk sort-reduce and a final reduced-key merge. `--reducer hash` keeps the earlier Python dictionary path for small debugging cases.

Enable fixed-size array or memmap-backed candidate storage inside the multilevel pipeline with config. This is intended for large-graph experiments; the default remains the simpler heap store.

```yaml
progress:
  enabled: true
  backend: plain
  min_interval_seconds: 5.0
candidates:
  store_backend: array
  use_chunked_generation: true
  mmap_dir: outputs/candidate_mmap
  incident_index_mmap_dir: outputs/incident_index_mmap
  edge_chunk_size: 1000000
  middle_chunk_size: 100000
  node_chunk_size: 1000000
  enable_partition_ann: true
  ann_num_projections: 4
  ann_window_size: 8
  ann_budget_K: 8
```

With `mmap_dir` set, each level writes `candidate_ids.npy`, `candidate_scores.npy`, `candidate_sources.npy`, and `candidate_counts.npy` under `mmap_dir/level_<n>`. With `incident_index_mmap_dir` set, capped two-hop also writes `incident_middle.npy`, `incident_endpoint_type.npy`, `incident_endpoints.npy`, and `incident_indptr.npy` under `incident_index_mmap_dir/level_<n>`. Chunked one-hop, capped two-hop, and SimHash bucket generation keep per-node budgets in the same store API used by the default path. Chunked capped two-hop builds an incident index once per level, keyed by `(middle_node, endpoint_type)`, then slices that index per middle-node chunk instead of rescanning all relation edges. In memmap mode, the index builder writes sorted temporary edge chunks and merges them into the final mmap arrays.

`enable_partition_ann` adds an optional ANN-style source. It is deterministic and dependency-free: for each same-type, same-partition group, it sorts nodes by several seeded random projections of the low-pass sketch and proposes only small sliding-window neighbors. `ann_budget_K` limits proposals per node from this source before the shared candidate store applies the global per-node budget.

## Optional Torch Dense Blocks

Set this in config to use Torch for dense row normalization in low-pass sketching:

```yaml
acceleration:
  dense_backend: torch
  device: auto
  fallback_to_numpy: true
  max_dense_bytes:
  scoring_batch_size: 65536
```

This path only handles dense blocks such as sketches and candidate scoring matrices. Relation arrays, candidate generation, and graph structure stay CPU-resident. Candidate scoring uses block-local Torch batches: each batch copies only the unique rows touched by candidate pairs for sketch, relation-profile, convolution-response, and feature distance terms. `max_dense_bytes` applies to the batch-local dense block, and `scoring_batch_size` controls candidate pairs per scoring batch.

## Diagnostics

Diagnostics include node counts by type, edge counts by relation, compression ratio, candidate count distribution, candidate source counts, matched-pair count, singleton ratio, relation weight preservation, and per-stage runtime.

Enable sampled large-graph envelopes with config:

```yaml
diagnostics:
  enable_large_graph_envelope: true
  edge_sample_size: 1024
```

When enabled, `diagnostics.json` includes `large_graph_envelope` with exact graph array bytes, current process RSS when available, runtime totals and the slowest stage, candidate-store byte estimates, candidate count quantiles, artifact directory sizes for candidate/incident mmap outputs, and bounded per-relation edge samples. Edge samples are deterministic, capped by `edge_sample_size` per relation, and report sample weight statistics, self-loop counts, and sampled unique endpoint counts without materializing two-hop neighborhoods.

## Current Limitations

- The default candidate store uses Python dictionaries and is intended for small and medium prototype runs. Large-graph experiments can opt into fixed-size array or memmap-backed candidate storage.
- Memmap export and chunked aggregation are available as explicit utilities, not the default pipeline. The default chunked reducer is NumPy sort-reduce; the Python hash reducer is retained only as a debug fallback.
- Memmap-backed capped two-hop indexing uses temporary sorted chunk files during construction, so large runs need enough disk headroom for those intermediate chunks.
- Partition-local ANN is a projection-window candidate source, not an HNSW/FAISS index. It is deterministic and lightweight, but not a high-recall ANN implementation.
- Torch acceleration covers dense helper kernels, low-pass sketch normalization, and block-local dense candidate scoring only.
- Large-graph diagnostics are sampled envelopes. They are intended for memory and runtime sanity checks, not exact distributional profiling.
- Spectral diagnostics use relation-wise edge energy approximations, not eigendecomposition.

## Next Engineering Steps

- Run calibrated full OGB MAG experiments with mmap candidates, memmap incident indexes, sampled diagnostics, and Torch dense scoring.
- Add optional external ANN backends after baseline projection-window ANN is profiled.
