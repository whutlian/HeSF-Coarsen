# Gate17.1 Support Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run Gate17.1 support-sensitivity and real-feedback diagnostics before any Gate18 paper-level run.

**Architecture:** Keep the existing Gate17 runner and selection pipeline, but add explicit Gate17.1 tests, summary logic, and a small diagnostic runner that writes `outputs/gate17_1/main` and `outputs/gate17_1/diag`. Selector callbacks must drive real validation/occlusion branches; semantic-tree deltas must be measured before any best-method claim.

**Tech Stack:** Python dataclasses, NumPy, PyTorch via local conda env `pytorch`, existing HGB loader and HETTREE-lite evaluator.

---

### Task 1: Main-Branch Compatibility

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/selector.py`
- Modify: `hesf_coarsen/task_first/selection/validation_selector.py`
- Test: `tests/test_gate17_config_compat.py`
- Test: `tests/test_gate17_selector_callbacks.py`

- [ ] **Step 1: Add failing config compatibility tests**
  Assert every Gate17/Gate17.1 selector, background strategy, and required config field can instantiate `SupportSelectorConfig`.

- [ ] **Step 2: Add failing callback tests**
  Assert real validation calls the validation callback more than once, real occlusion calls the occlusion callback for multiple blocks, and zero occlusion deltas are labeled degenerate with proxy fallback diagnostics.

- [ ] **Step 3: Update config and selector helpers**
  Support required `block_key_mode` values, field names from the prompt, callback types, trial row fields, and explicit real-feedback diagnostics.

- [ ] **Step 4: Run focused selector/config tests**
  Run `conda run -n pytorch python -m pytest tests/test_gate17_config_compat.py tests/test_gate17_selector_callbacks.py -q`.

### Task 2: Prototype Splitting And Raw Bridge Diagnostics

**Files:**
- Modify: `hesf_coarsen/task_first/selection/condensation.py`
- Test: `tests/test_gate17_prototype_splitting.py`

- [ ] **Step 1: Add failing prototype splitting test**
  Construct a giant DBLP-aware bucket and assert `prototype_member_count_max <= max_members_per_prototype`, split counts are positive, bridge raw preservation is positive, and rare-class fallback is blocked.

- [ ] **Step 2: Implement deterministic DBLP-aware splitting and bridge preservation**
  Split by degree/anchor/relation/stable id; force raw bridge nodes when configured; report fallback and saturation diagnostics.

- [ ] **Step 3: Run prototype tests**
  Run `conda run -n pytorch python -m pytest tests/test_gate17_prototype_splitting.py tests/test_gate17_dblp_prototype_condensation.py tests/test_gate16_prototype_condensation.py -q`.

### Task 3: Gate17.1 Summary Degeneracy

**Files:**
- Create: `experiments/scripts/summarize_gate17_1.py`
- Test: `tests/test_gate17_summary_degeneracy.py`

- [ ] **Step 1: Add failing summary degeneracy test**
  Synthetic tied rows must produce `all_methods_tied=true`, `best_validation_selected_method=None`, and no `PARTIAL_DBLP_BLOCKER`.

- [ ] **Step 2: Implement Gate17.1 summary**
  Write `result.json`, `final_report.md`, `gate17_1_decision.md`, validation-selected by method/dataset, exact-only paired gaps, metric nunique, and pass/fail booleans.

- [ ] **Step 3: Run summary tests**
  Run `conda run -n pytorch python -m pytest tests/test_gate17_summary_degeneracy.py -q`.

### Task 4: Support-Sensitivity Runner

**Files:**
- Create: `experiments/scripts/run_gate17_1_support_sensitivity.py`
- Modify: `hesf_coarsen/eval/hettree_task.py` only if semantic-tree access cannot be done through existing helpers
- Test: `tests/test_gate17_support_sensitivity_probe.py`

- [ ] **Step 1: Add failing tiny support-sensitivity test**
  On a tiny heterogeneous graph, removing support with `max_paths >= 2` must change at least one non-self target path feature.

- [ ] **Step 2: Implement semantic-tree delta helpers**
  Compare compressed, full, and target-only semantic-tree tensors over aligned target nodes and write all required delta columns.

- [ ] **Step 3: Implement Gate17.1 runner**
  Support required CLI options, methods including `target-only-empty-support`, guard against `task_epochs=0`/`max_paths<=1` quality claims, and write every required main/diag file.

- [ ] **Step 4: Run Gate17.1 tests**
  Run the full required test command from the prompt under conda env `pytorch`.

### Task 5: Local Experiments And Git

**Files:**
- Output: `outputs/gate17_main_smoke/`
- Output: `outputs/gate17_1/main/`
- Output: `outputs/gate17_1/diag/`

- [ ] **Step 1: Run main-branch smoke**
  Run `conda run -n pytorch python -m experiments.scripts.run_gate17_support_selection --smoke --output-dir outputs/gate17_main_smoke --data-root data --device cpu`.

- [ ] **Step 2: Run Gate17.1 sanity command**
  Run the exact Gate17.1 command from the prompt with local `pytorch` env and CPU.

- [ ] **Step 3: Handle OOM only if observed**
  If local OOM or GPU OOM occurs, stop the run and return a server command instead of fabricating results.

- [ ] **Step 4: Verify deliverables**
  Check all required files exist and have nonzero size; inspect `result.json` and decision report.

- [ ] **Step 5: Commit and push main if code changed**
  Stage only Gate17.1-owned files, commit, and push to `origin main`.
