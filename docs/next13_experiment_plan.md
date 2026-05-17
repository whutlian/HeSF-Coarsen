# Next13 HeSF-LVC Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the Next13 iteration: paper table fixes, path-mass metapath diagnostics, structure-critical validation, AH-UGC-style fair tables, OGBN exclusive timing plus A4 backend, and final bounded reporting.

**Architecture:** Keep outputs under `outputs/` and commit only code/config/docs/tests. Reuse Next12 coarse graph loading helpers for HGB methods, add a separate sparse metapath transition evaluator, add lightweight structure-critical diagnostics that do not claim official HGB performance, and keep OGBN results system-only.

**Tech Stack:** Python, NumPy, existing `HeteroGraph` / `Assignment` structures, existing CSV/plot helpers, pytest, local conda `pytorch`.

---

### Task 0: Preflight

- [x] Read `C:/Users/slian/Desktop/codex_next13_experiment_plan_prompt.md`.
- [x] Confirm branch is `main` and record unrelated untracked files.
- [x] Confirm Next12 output roots exist.

### Task 1: P0 Paper Table Fixes

**Files:**
- Create: `experiments/scripts/summarize_next13_paper_tables.py`
- Create: `tests/test_next13_paper_tables.py`
- Create: `docs/claim_boundary_next13.md`

- [x] Write tests that fail unless tuned global AH-UGC-style is represented as `AH-UGC-style tuned-global`.
- [x] Write tests that distinguish global-fixed, validation-selected, and oracle appendix AH-UGC-style rows.
- [x] Write tests that reject paper-facing bare `DEE` columns.
- [x] Implement paper table summarizer using Next12 paper tables plus Next12 AH-UGC sweep summary.
- [x] Generate `outputs/exp_next13_paper_tables_20260517_summary`.

### Task 2: P1/P2 Path-Mass Metapath Diagnostics

**Files:**
- Create: `hesf_coarsen/eval/metapath_mass.py`
- Create: `tests/test_metapath_mass.py`
- Create: `experiments/scripts/run_next13_metapath_mass.py`
- Create: `experiments/scripts/summarize_next13_metapath_mass.py`

- [x] Write tests for identity near-zero error, collapsed assignment higher error, typed vs untyped relation handling, 2-hop/3-hop shapes, deterministic shared probes, and rejection of survival-only positive claims.
- [x] Implement row-normalized sparse relation transition application with sequential matvec only.
- [x] Implement original/coarse/lifted path-mass comparison metrics and collapse/count secondary merge.
- [x] Run all requested datasets, seeds, schema path lengths, probes, and methods.
- [x] Generate summary CSVs and figures under `outputs/exp_next13_metapath_mass_20260517_summary`.

### Task 3: P3 Structure-Critical Validation

**Files:**
- Create: `hesf_coarsen/eval/structure_tasks.py`
- Create: `experiments/scripts/run_next13_structure_critical_tasks.py`
- Create: `experiments/scripts/summarize_next13_structure_critical_tasks.py`
- Create: `tests/test_next13_structure_tasks.py`

- [x] Write tests for low-pass signal reconstruction metrics and feature-free label propagation metrics.
- [x] Implement diffusion-based low-pass signal generation, coarse projection/lift, and reconstruction scoring.
- [x] Implement feature-free label propagation diagnostic with projected/refined/best/AUC outputs.
- [x] Run all requested methods, datasets, seeds, and both diagnostic tasks.
- [x] Generate summary outputs under `outputs/exp_next13_structure_critical_20260517_summary`.

### Task 4: P4 AH-UGC-Style Fair Baseline

**Files:**
- Create: `experiments/scripts/summarize_next13_ahugc_fair_baseline.py`
- Create: `tests/test_next13_ahugc_table.py`
- Create: `docs/ahugc_style_baseline_next13.md`

- [x] Write tests for global fixed, validation-selected, oracle appendix-only, and main external baseline rows.
- [x] Implement mean+/-std formatting and target-hit aggregation without single-seed or max leakage into main rows.
- [x] Generate `outputs/exp_next13_ahugc_fair_baseline_20260517_summary`.

### Task 5: P5 OGBN Exclusive Timing and A4 Backend

**Files:**
- Modify: `hesf_coarsen/coarsen/aggregate_edges.py`
- Create: `tests/test_aggregation_exclusive_timing.py`
- Create: `experiments/scripts/run_next13_ogbn_aggregation_backend.py`
- Create: `experiments/scripts/summarize_next13_ogbn_aggregation_backend.py`

- [x] Write tests for exclusive timing keys, timing sum/residual invariants, and A4 correctness vs A0.
- [x] Add exclusive timing diagnostics while preserving existing inclusive timing fields.
- [x] Add `local_prededup_sort` reducer exposed as `A4_local_prededup_sort_reducer`.
- [x] Run all requested OGBN sizes/methods/backends locally unless OOM occurs.
- [x] Generate summary outputs under `outputs/exp_next13_ogbn_aggregation_backend_20260517_summary`.

### Task 6: P6 Docs, Final Report, Verification, Commit

**Files:**
- Create: `docs/metapath_mass_diagnostics_next13.md`
- Create: `outputs/exp_next13_final_report_20260517.md`

- [x] Write docs with bounded claim language.
- [x] Generate final report with exact commands, outputs, limitations, and claim boundary.
- [x] Run targeted pytest from the prompt.
- [x] Run full pytest or document failures with exact reasons.
- [x] Stage only code/config/docs/tests and commit to `main`.
