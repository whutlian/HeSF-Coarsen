# Next8 Final Gap Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the attachment's full P0-P5 experiment outputs without omitting required final-table, flatten-sum, source-policy, OGBN-system, and quality-cost Pareto evidence.

**Architecture:** Reuse completed Next7 HGB and Next6 OGBN outputs whenever they already contain the required 5-seed or scale evidence. Add narrow reporting and experiment switches for missing table columns, cross-model/low-label task probes, and source-aware candidate filtering; keep generated outputs out of git.

**Tech Stack:** Python 3.9 in `C:\Users\slian\anaconda3\envs\pytorch`, pytest, CSV/Markdown reports, matplotlib figures, existing HeSF coarsening scripts.

---

### Task 1: Final Gap Table Reporter

**Files:**
- Modify: `experiments/scripts/summarize_next7_baseline_gap.py`
- Modify: `tests/test_next7_baseline_gap.py`

- [ ] **Step 1: Write failing tests**

Add assertions that the reporter emits `final_gap_main_table.csv`, `final_gap_per_dataset_table.csv`, comparator columns named `delta_best_vs_oracle_coarse_baseline`, `delta_vs_H0`, `delta_vs_flatten_sum`, `delta_vs_GraphZoom_style`, `delta_vs_ConvMatch_style`, `delta_vs_random`, and rows for `full RGCN default`, `full RGCN tuned`, `HAN-small`, and `HGT-lite`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_next7_baseline_gap.py -q
```

Expected: fails because the new files/columns/full-graph rows do not exist yet.

- [ ] **Step 3: Implement reporter**

Extend the existing reporter to carry FSE, target hit, peak memory, all refine checkpoints, explicit comparator deltas, full-graph task rows, per-dataset tables, win-rate by dataset/seed, and quality-cost input columns.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command and expect all tests in `tests/test_next7_baseline_gap.py` to pass.

### Task 2: Cross-Model And Low-Label Task Eval

**Files:**
- Modify: `hesf_coarsen/eval/task_gnn.py`
- Modify: `experiments/scripts/run_hgb_task_eval.py`
- Modify: `tests/test_task_gnn.py`
- Modify: `tests/test_experiment_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add tests for `coarse_model=rgcn_lite|han_small|hgt_lite`, CSV rows containing `coarse_model`, and `--no-write-run-json` preserving existing run `task_eval.json`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_task_gnn.py tests/test_experiment_pipeline.py -q
```

Expected: fails on missing coarse-model and no-write flags.

- [ ] **Step 3: Implement task eval switches**

Add model factory selection for coarse/refine models, CLI `--coarse-models`, and `--no-write-run-json` so challenge runs can be written only to separate output CSVs.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command and expect it to pass.

### Task 3: Source-Aware Candidate Filtering

**Files:**
- Modify: `hesf_coarsen/candidates/bounded_heap.py`
- Modify: `hesf_coarsen/candidates/array_store.py`
- Modify: `hesf_coarsen/coarsen/multilevel.py`
- Modify: `hesf_coarsen/matching/greedy.py`
- Modify: `tests/test_array_candidate_store.py`
- Modify: `tests/test_multilevel_pipeline.py`

- [ ] **Step 1: Write failing tests**

Add tests for source policy config: bucket priority high, bucket top-k 8, onehop top-k 2, onehop rejected when raw spec delta exceeds bucket q95, and fallback selected share capped at 0.05.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_array_candidate_store.py tests/test_multilevel_pipeline.py -q
```

Expected: fails because source policy is not implemented.

- [ ] **Step 3: Implement source policy**

Thread `candidates.source_policy` into candidate stores and non-streaming scoring/matching diagnostics. Apply per-source local caps and priority only for candidate retention, then apply spec-threshold filtering before greedy matching.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command and expect it to pass.

### Task 4: OGBN System Summary

**Files:**
- Create: `experiments/scripts/summarize_ogbn_system_scale.py`
- Create: `tests/test_ogbn_system_summary.py`

- [ ] **Step 1: Write failing tests**

Create fixture summaries with 200k/500k/1M/1.94M rows and assert the script emits candidate pairs, scored pairs, selected merges, coarse edges, matching sec, aggregation sec, edges/sec, pairs/sec, RSS, and shard GB.

- [ ] **Step 2: Verify RED**

Run:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_ogbn_system_summary.py -q
```

Expected: fails because the script does not exist.

- [ ] **Step 3: Implement OGBN system summary**

Read the existing Next6 OGBN summaries and write `ogbn_system_scale_table.csv` plus a Markdown report for the system section.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command and expect it to pass.

### Task 5: Run Required Experiments

**Files:**
- Generated only under `outputs/`, not committed.

- [ ] **Step 1: P0/P5 final table and Pareto**

Run the enhanced Next7 reporter using existing P/S/H0/H4/H6/flatten/random/GraphZoom/ConvMatch/full-graph summaries.

- [ ] **Step 2: P1 flatten-sum challenge**

Run projected/refine checkpoint comparison, coarse-model transfer comparison, low-label comparison, relation-wise metric extraction, and per-dataset failure-case tables for ACM/DBLP/IMDB.

- [ ] **Step 3: P3 source-aware filtering**

Run baseline policy vs source-aware policy for HeSF-LVC-P/S on ACM/DBLP/IMDB, 5 seeds, using local `pytorch`.

- [ ] **Step 4: P4 OGBN system summary**

Generate the 200k/500k/1M/1.94M system table from local outputs; rerun only if a required field is missing.

### Task 6: Verification And Commit

**Files:**
- Commit code/config/test/docs only.

- [ ] **Step 1: Run focused tests**

Run all tests touched in Tasks 1-4.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest -q
```

- [ ] **Step 3: Commit**

Stage only code/config/test/docs and commit to `main`.
