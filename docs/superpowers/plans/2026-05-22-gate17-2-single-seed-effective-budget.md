# Gate17.2 Single-Seed Effective Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Gate17.2 on one best historical seed per dataset to verify effective support budget, candidate semantic sensitivity, real feedback nondegeneracy, and prototype saturation before any Gate18 run.

**Architecture:** Reuse Gate17.1 semantic-tree and task-evaluation helpers, but add Gate17.2-specific method names, seed policy, effective-budget accounting, no-free-raw selector configs, no-fallback real feedback diagnostics, and fail-fast summary decisions. Generated outputs live under `outputs/gate17_2_single_seed/`.

**Tech Stack:** Python dataclasses, NumPy, existing HETTREE-lite evaluator, local conda env `pytorch`.

---

### Task 1: Effective Budget And Raw Bridge Discipline

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/condensation.py`
- Create: `experiments/scripts/gate17_2_effective_budget.py`
- Test: `tests/test_gate17_2_effective_budget.py`

- [ ] **Step 1: Add failing tests**
  Assert no-free-raw configs emit `forced_raw_support_count=0`, compute effective budget fields, and fail when prototype or raw context exceeds requested ratio by more than `0.02`.

- [ ] **Step 2: Add config controls**
  Add `raw_bridge_mode`, `allow_proxy_fill`, and keep `force_raw_bridge_nodes=false` / `force_raw_keep_high_degree_bridges=false` as Gate17.2-safe defaults.

- [ ] **Step 3: Implement effective budget helper**
  Compute selected, forced raw, prototype, effective support node, represented context, leak ratios, and `effective_budget_exact_match`.

- [ ] **Step 4: Run focused tests**
  Run `conda run -n pytorch python -m pytest tests/test_gate17_2_effective_budget.py -q`.

### Task 2: No-Fallback Real Feedback Diagnostics

**Files:**
- Modify: `hesf_coarsen/task_first/selection/validation_selector.py`
- Test: `tests/test_gate17_2_feedback_no_fallback.py`

- [ ] **Step 1: Add failing validation no-fallback test**
  A zero-gain validation callback must accept no block, set `real_validation_degenerate=true`, and keep `proxy_fallback_fill_count=0`.

- [ ] **Step 2: Add failing occlusion completeness test**
  Real occlusion no-fallback must emit complete non-NaN CE/margin/KL/tree-delta metrics and report nonzero delta rates.

- [ ] **Step 3: Implement selector diagnostics**
  Enforce positive `min_gain`, `allow_proxy_fill=false`, occlusion completeness, max/mean deltas, and no proxy fallback for no-fallback method configs.

- [ ] **Step 4: Run feedback tests**
  Run `conda run -n pytorch python -m pytest tests/test_gate17_2_feedback_no_fallback.py tests/test_gate17_selector_callbacks.py -q`.

### Task 3: Gate17.2 Runner And Summary

**Files:**
- Create: `experiments/scripts/run_gate17_2_single_seed_effective_budget.py`
- Create: `experiments/scripts/summarize_gate17_2.py`
- Test: `tests/test_gate17_2_runner_summary.py`

- [ ] **Step 1: Add failing runner/summary tests**
  Assert default seed policy is `ACM:23456, DBLP:23456, IMDB:45678`, no accidental 5-seed loop, required output names, and failure reasons disable best-method reporting under degeneracy.

- [ ] **Step 2: Implement runner**
  Support `--seed-policy best_single`, `--dataset-seed-map`, required methods/ratios/settings, target-only, baselines, no-free-raw candidates, semantic-tree deltas, effective budget CSVs, and diagnostics.

- [ ] **Step 3: Implement summary**
  Write required result/final/decision files and fail-fast labels: budget leak, full-graph equivalent, validation degenerate, occlusion degenerate, DBLP prototype saturation, all-tie, no-test-leakage.

- [ ] **Step 4: Run runner/summary tests**
  Run `conda run -n pytorch python -m pytest tests/test_gate17_2_runner_summary.py -q`.

### Task 4: Local Gate17.2 Experiment And Verification

**Files:**
- Output: `outputs/gate17_2_single_seed/`

- [ ] **Step 1: Run all Gate17.2 tests and compile checks**
  Run the Gate17.2 tests, relevant Gate17 tests, `py_compile`, and `git diff --check`.

- [ ] **Step 2: Run Gate17.2 single-seed command**
  Run the required command under `conda run -n pytorch`, CPU device unless explicitly overridden.

- [ ] **Step 3: Handle OOM only if observed**
  If OOM/GPU OOM occurs, stop and return a server command.

- [ ] **Step 4: Verify outputs**
  Confirm all required main and diagnostics files exist and inspect `result.json` decision/failure reasons.

- [ ] **Step 5: Commit and push main**
  Stage only Gate17.2-owned files, commit, and push `origin main`.
