# Next18 Accuracy Final Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a final high-fidelity validation cycle for the experimental accuracy-first branch and make a hard keep/drop decision.

**Architecture:** Keep the preservation-first HeSF-LVC-P/S path untouched. Add a clean Next18 protocol layer that separates coarse transfer, approximate full-target adapters, and real full-target inference, then run only A1/A2 plus keep-target comparators with explicit model-fidelity metadata and decision rules.

**Tech Stack:** Python, NumPy, PyTorch via local conda env `pytorch`, existing HGB graph IO/coarsening helpers, CSV/Markdown output scripts.

---

### Task 1: Protocol And Fidelity Tests

**Files:**
- Create: `tests/test_accuracy_full_target_protocol.py`
- Create: `tests/test_model_fidelity_metadata.py`
- Create: `tests/test_keep_target_assignment_integrity.py`
- Create: `tests/test_accuracy_branch_decision_rules.py`

- [x] **Step 1: Write failing tests**

Tests must require `real_full_target_inference` to use real target-domain metrics, not projected coarse-transfer metrics; require fidelity metadata fields; require keep-target assignment integrity; and require deterministic decision categories.

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n pytorch python -m pytest tests/test_accuracy_full_target_protocol.py tests/test_model_fidelity_metadata.py tests/test_keep_target_assignment_integrity.py tests/test_accuracy_branch_decision_rules.py -q`

Expected: FAIL due missing Next18 modules.

### Task 2: Protocol Layer

**Files:**
- Create: `hesf_coarsen/accuracy/full_target_protocol.py`
- Modify: `hesf_coarsen/accuracy/full_target_inference.py`

- [ ] **Step 1: Implement protocol layer**

Add protocol rows for `coarse_transfer`, `approx_full_target_adapter`, and `real_full_target_inference`, with provenance fields `target_domain`, `support_domain`, and `inference_domain`.

- [ ] **Step 2: Preserve backward compatibility**

Keep `evaluate_full_target_inference` available but make its metadata clearly approximate.

### Task 3: Model Fidelity And Decision Rules

**Files:**
- Create: `hesf_coarsen/accuracy/model_fidelity_registry.py`
- Create: `hesf_coarsen/accuracy/accuracy_branch_decision.py`

- [ ] **Step 1: Implement metadata registry**

Record official availability and local faithful-reproduction limitations for SeHGNN, HETTREE, FreeHGC, and lite adapters.

- [ ] **Step 2: Implement kill criteria**

Return exactly one of `KEEP_ACCURACY_BRANCH_MINIMAL`, `DROP_HYBRID_B_KEEP_A1_A2_EXPLORATORY`, or `DROP_ENTIRE_ACCURACY_BRANCH`.

### Task 4: Next18 Runners And Outputs

**Files:**
- Create: `experiments/scripts/run_next18_accuracy_protocol_audit.py`
- Create: `experiments/scripts/run_next18_accuracy_keep_target_final.py`
- Create: `experiments/scripts/run_next18_literature_alignment.py`
- Create: `experiments/scripts/summarize_next18_accuracy_decision.py`
- Create outputs under `outputs/exp_next18_*`

- [ ] **Step 1: Generate implementation audit**

Write `outputs/exp_next18_accuracy_protocol_audit/implementation_audit.md`.

- [ ] **Step 2: Run A1/A2 and keep-target comparators**

Use local `pytorch`, 3 datasets, 3 seeds, A1/A2 only, plus comparator rows. If OOM occurs, emit server commands.

- [ ] **Step 3: Generate literature alignment**

Write direct-comparison notes and table with comparability flags.

- [ ] **Step 4: Generate decision docs**

Write `docs/next18_accuracy_branch_decision.md` and either postmortem or survivor spec.

### Task 5: Verification And Push

**Files:**
- Stage only Next18 code/docs/tests/config outputs needed for reproducibility.

- [ ] **Step 1: Run focused tests**

Run all Next18 tests plus relevant Next17 regression tests.

- [ ] **Step 2: Run experiment smoke/full commands**

Use `conda run -n pytorch` for local commands.

- [ ] **Step 3: Commit and push**

Commit to `main` and push to `origin/main`.
