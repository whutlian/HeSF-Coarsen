# Gate14 Task-First Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair and fairly evaluate the task-first HeSF-TC branch under realized-ratio-matched downstream node classification.

**Architecture:** Extend the existing Gate13 task-first modules instead of replacing the pipeline. Keep target nodes singleton, add explicit v2 coverage/purity modes, add a bounded stateful cluster-signature matcher, and create Gate14 scripts that emit the exact required CSV/Markdown/figure artifacts.

**Tech Stack:** Python, NumPy, PyTorch through local conda env `pytorch`, existing `hettree_lite` diagnostic evaluator, matplotlib for figures, pytest for Stage 0 tests.

---

### Task 1: Stage 0 Tests And Audit

**Files:**
- Create: `tests/test_task_first_gate14_coverage_v2.py`
- Create: `tests/test_task_first_gate14_purity_v2.py`
- Create: `tests/test_task_first_gate14_stateful_matching.py`
- Create: `tests/test_task_first_gate14_ratio_matching.py`
- Create: `tests/test_task_first_gate14_candidate_sources.py`
- Create: `tests/test_task_first_gate14_evaluator_ceiling.py`
- Create: `experiments/scripts/audit_task_first_gate14_code.py`

- [ ] Write failing tests for all Gate14 Stage 0 requirements.
- [ ] Run the new tests and confirm the missing APIs fail.
- [ ] Implement only the minimal APIs needed for these tests.
- [ ] Run the new tests again and confirm they pass.
- [ ] Generate `code_audit.md` in the Gate14 output directory.

### Task 2: Task-First Repair Primitives

**Files:**
- Modify: `hesf_coarsen/task_first/config.py`
- Modify: `hesf_coarsen/task_first/support_coverage.py`
- Modify: `hesf_coarsen/task_first/support_purity.py`
- Modify: `hesf_coarsen/task_first/candidates.py`
- Create: `hesf_coarsen/task_first/stateful_matching.py`
- Modify: `hesf_coarsen/task_first/pipeline.py`

- [ ] Add explicit `coverage_v1` legacy mode and `coverage_v2` components.
- [ ] Add `support_footprint_mode` and v2 known/unknown policy.
- [ ] Add `relation_response_knn`, `target_response_signature_knn`, and `hybrid_task_aware` candidate sources.
- [ ] Add bounded stateful matching diagnostics and integrate it when `pair_delta_mode=stateful_signature`.
- [ ] Keep dense adjacency, A^2, full two-hop materialization, and large eigendecomposition out of the implementation.

### Task 3: Gate14 Scripts

**Files:**
- Create: `experiments/scripts/gate14_task_first_common.py`
- Create: `experiments/scripts/run_task_first_gate14_full_graph_lite_tuning.py`
- Create: `experiments/scripts/run_task_first_gate14_ratio_matched_baselines.py`
- Create: `experiments/scripts/run_task_first_gate14_candidate_sources.py`
- Create: `experiments/scripts/run_task_first_gate14_repairs.py`
- Create: `experiments/scripts/run_task_first_gate14_final.py`
- Create: `experiments/scripts/summarize_task_first_gate14.py`

- [ ] Reuse Gate13 loading, coarsening, baseline, and hettree_lite evaluation helpers where possible.
- [ ] Emit required full graph tuning tables.
- [ ] Emit requested-ratio, realized-ratio, and nearest-ratio baseline tables.
- [ ] Emit candidate source diagnostics for all required sources.
- [ ] Emit Gate14 final run table, validation selection, oracle appendix, gaps, recovery, diagnostics, and figures.

### Task 4: Local Experiments

**Files:**
- Output: `outputs/exp_task_first_gate14_hgb_20260521/`

- [ ] Run Stage 0 tests under `conda run -n pytorch`.
- [ ] Run full graph lite tuning first.
- [ ] Run ratio-matched baselines.
- [ ] Run coverage_v2, purity_v2, candidate-source, and stateful comparisons.
- [ ] Run the Gate14 main matrix.
- [ ] Summarize into all required MD/CSV/PNG files.

### Task 5: Verification And Commit

**Files:**
- Verify all changed Python files.
- Verify required output artifacts exist.

- [ ] Run focused pytest suite.
- [ ] Run `py_compile` and `git diff --check`.
- [ ] Commit changes to local `main`.
