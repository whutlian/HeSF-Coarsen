# Next12 HeSF-LVC Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the Next12 experiment/code iteration: method-sensitive metapath retention, paper table refresh, tuned AH-UGC-style baseline, structure-sensitive stress, one real OGBN aggregation backend, and final claim-boundary docs.

**Architecture:** Keep generated outputs under `outputs/` and code/config/docs/tests in git. Add a bounded sparse metapath-retention evaluator under `hesf_coarsen/eval/`, extend the type-isolated LSH baseline for tuning, add a packed-key aggregation backend under the existing chunked aggregation path, and implement Next12 runner/summarizer scripts around existing HGB/OGBN run utilities.

**Tech Stack:** Python, NumPy sparse-style array operations, existing `HeteroGraph` / `Assignment` data structures, pytest, local conda `pytorch`.

---

### Task 0: Preflight

- [x] Read `C:/Users/slian/Desktop/codex_next12_experiment_plan_prompt.md`.
- [x] Confirm branch is `main` and record unrelated untracked files.
- [x] Confirm existing Next10/Next11 output roots needed by Next12.

### Task 1: P0 Method-Sensitive Metapath Retention

**Files:**
- Create: `hesf_coarsen/eval/metapath_retention.py`
- Create: `tests/test_metapath_retention.py`
- Create: `experiments/scripts/run_next12_metapath_retention.py`
- Create: `experiments/scripts/summarize_next12_metapath_retention.py`

- [x] Write failing tests for identity, relation removal, flatten relation control, method sensitivity, bounded count caps, invalid schema paths, and no dense/product construction.
- [x] Implement sparse bounded path inference/sampling/evaluation.
- [x] Implement runner that reuses the same original path samples across methods.
- [x] Implement summarizer and required paper/figure outputs.
- [x] Run `pytest tests/test_metapath_retention.py -q`.

### Task 2: P1 Paper Table Refresh

**Files:**
- Create: `experiments/scripts/summarize_next12_paper_tables.py`
- Create: `tests/test_next12_metapath_summarizer.py`

- [x] Write failing summarizer tests for metapath columns and non-diagnostic note.
- [x] Implement table1-table5 outputs with explicit DEE names.
- [x] Run summarizer against Next11 + Next12 metapath outputs.

### Task 3: P2 AH-UGC-Style Tuning Mini-Grid

**Files:**
- Modify: `hesf_coarsen/baselines/type_isolated_lsh.py`
- Create: `configs/paper/hgb_ahugc_style_sweep.yaml`
- Create: `experiments/scripts/run_next12_ahugc_style_sweep.py`
- Create: `experiments/scripts/summarize_next12_ahugc_style_sweep.py`
- Extend: `tests/test_type_isolated_lsh_baseline.py`

- [x] Write failing baseline tests for assignment sources and tuned config diagnostics.
- [x] Add `hash_bits`, `bucket_topk`, and `assignment_source` support.
- [x] Implement full mini-grid runner and summarizer.
- [x] Run all ACM/DBLP/IMDB seeds/configs locally.

### Task 4: P3 Structure-Sensitive Task Stress

**Files:**
- Create: `experiments/scripts/run_next12_structure_sensitive_stress.py`
- Create: `experiments/scripts/summarize_next12_structure_sensitive_stress.py`
- Create: `tests/test_next12_structure_sensitive_stress.py`

- [x] Write failing tests for feature mask/noise/structure-only graph transforms and summary win rates.
- [x] Implement stress runner for all requested methods/datasets/seeds/settings.
- [x] Include relation energy and metapath columns when available.
- [x] Run local stress experiment.

### Task 5: P4 OGBN Aggregation Backend

**Files:**
- Modify: `hesf_coarsen/coarsen/aggregate_edges.py`
- Create: `experiments/scripts/run_next12_ogbn_aggregation_backend.py`
- Create: `experiments/scripts/summarize_next12_ogbn_aggregation_backend.py`
- Create: `tests/test_aggregation_packed_key_backend.py`

- [x] Write failing packed-key reducer tests.
- [x] Implement `packed_key_sort` backend with rectangular type-local key space and overflow guard.
- [x] Implement benchmark runner/summarizer for A0 vs A3.
- [x] Run OGBN 200k/500k/1m/full-local P/S benchmark.

### Task 6: P5 Docs, Final Report, Verification, Commit

**Files:**
- Create: `docs/claim_boundary_next12.md`
- Create: `docs/metapath_retention_next12.md`
- Create: `docs/paper_table_index_next12.md`
- Create: `outputs/exp_next12_final_report_20260517.md`

- [x] Write docs with bounded claim language.
- [x] Run targeted pytest command from the prompt.
- [x] Run full pytest suite.
- [x] Stage only code/config/docs/tests and commit to `main`.
