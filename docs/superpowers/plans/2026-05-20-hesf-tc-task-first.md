# HeSF-TC Task-First Target-Conditioned Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a new experimental `HeSF-TC` branch with support-only target-conditioned coarsening, strict protocol gating, tests, docs, config, and a smoke runner.

**Architecture:** Add a separate `hesf_coarsen/task_first/` package so preservation-first HeSF-LVC-P/S and deprecated accuracy branch code remain untouched. The v1 implementation uses sparse relation arrays, target-label seed propagation, support-only candidate filtering, local task-first merge deltas, and the existing assignment/aggregation stack.

**Tech Stack:** Python dataclasses, NumPy sparse edge-array loops, existing `HeteroGraph`, `Assignment`, `coarsen_graph`, and pytest under conda env `pytorch`.

---

### Task 1: Core Task-First Data Structures

**Files:**
- Create: `hesf_coarsen/task_first/__init__.py`
- Create: `hesf_coarsen/task_first/config.py`
- Create: `hesf_coarsen/task_first/state.py`
- Create: `hesf_coarsen/task_first/probes.py`
- Test: `tests/test_task_first_state.py`

- [ ] Write failing tests for seed matrix, target/support split, identity response sanity.
- [ ] Implement config dataclasses and state builder.
- [ ] Run targeted tests.

### Task 2: Target-Conditioned Terms and Constraints

**Files:**
- Create: `hesf_coarsen/task_first/relation_response.py`
- Create: `hesf_coarsen/task_first/support_coverage.py`
- Create: `hesf_coarsen/task_first/support_purity.py`
- Create: `hesf_coarsen/task_first/constraints.py`
- Create: `hesf_coarsen/task_first/scoring.py`
- Test: `tests/test_task_first_scoring.py`

- [ ] Write failing tests for JS purity block, coverage delta ordering, merge constraints, and score fields.
- [ ] Implement sparse per-merge approximations without dense adjacency/products.
- [ ] Run targeted tests.

### Task 3: Pipeline, Protocol, Config, Docs, Smoke

**Files:**
- Create: `hesf_coarsen/task_first/pipeline.py`
- Create: `hesf_coarsen/task_first/eval_protocol.py`
- Create: `configs/task_first/hgb_hesf_tc_v1.yaml`
- Create: `docs/task_first_hesf_tc_design.md`
- Create: `experiments/scripts/run_task_first_smoke.py`
- Test: `tests/test_task_first_pipeline.py`

- [ ] Write failing integration tests for support-only coarsening and protocol rejection of lite backbones.
- [ ] Implement support-only greedy matching, aggregate graph, diagnostics, strict fidelity gate.
- [ ] Add config, design doc, and smoke runner.
- [ ] Run tests and smoke command.

### Task 4: Verification and Publish

- [ ] Run targeted pytest in conda env `pytorch`.
- [ ] Run smoke runner in conda env `pytorch`.
- [ ] Run `git diff --check`.
- [ ] Stage only HeSF-TC files, commit, and push to GitHub `main`.
