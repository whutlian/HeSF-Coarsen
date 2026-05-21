# Gate15 Supervised Support Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Gate15 supervised support selection experiments that replace Gate14 static pairwise coarsening with task-first support selection.

**Architecture:** Add a new `hesf_coarsen.task_first.selection` submodule for teacher metrics, support features, contribution scores, selectors, condensation, pipeline diagnostics, and Gate15 experiment scripts. Keep Gate14 code as reproducible references and deprecate its handcrafted variants in audit outputs.

**Tech Stack:** Python, NumPy, existing HeSF `HeteroGraph`/`Assignment`, existing `hettree_lite` evaluator, pytest, matplotlib for figures.

---

### Task 1: Gate15 Audit And Tests

**Files:**
- Create: `tests/test_gate15_support_features.py`
- Create: `tests/test_gate15_teacher_outputs.py`
- Create: `tests/test_gate15_support_selector.py`
- Create: `tests/test_gate15_condensation.py`
- Create: `tests/test_gate15_pipeline_outputs.py`
- Create: `tests/test_gate15_no_test_leakage.py`
- Create: `experiments/scripts/audit_gate15_code.py`

- [ ] **Step 1: Write failing tests for Gate15 public APIs**

Add tests that import the new `hesf_coarsen.task_first.selection` APIs and assert target preservation, finite feature matrices, budget matching, typed background condensation, no test-label leakage metadata, and required metric columns.

- [ ] **Step 2: Run the Gate15 tests and verify they fail because APIs do not exist**

Run: `conda run -n pytorch python -m pytest tests/test_gate15_support_features.py tests/test_gate15_teacher_outputs.py tests/test_gate15_support_selector.py tests/test_gate15_condensation.py tests/test_gate15_pipeline_outputs.py tests/test_gate15_no_test_leakage.py -q`

Expected: import errors for the new selection module.

- [ ] **Step 3: Implement code audit script**

Create `experiments/scripts/audit_gate15_code.py` to write `code_sync_report.md` and `method_to_code_path.csv`, mapping Gate14 method names to committed code paths and recording coverage/purity/scoring/pipeline/evaluator caveats.

### Task 2: Selection Module

**Files:**
- Create: `hesf_coarsen/task_first/selection/__init__.py`
- Create: `hesf_coarsen/task_first/selection/config.py`
- Create: `hesf_coarsen/task_first/selection/support_features.py`
- Create: `hesf_coarsen/task_first/selection/teacher.py`
- Create: `hesf_coarsen/task_first/selection/contribution.py`
- Create: `hesf_coarsen/task_first/selection/selector.py`
- Create: `hesf_coarsen/task_first/selection/condensation.py`
- Create: `hesf_coarsen/task_first/selection/pipeline.py`
- Create: `hesf_coarsen/task_first/selection/diagnostics.py`

- [ ] **Step 1: Implement dataclass configs from the Gate15 prompt**

Use frozen dataclasses for feature, teacher, selector, regularizer, and top-level `Gate15Config`.

- [ ] **Step 2: Implement teacher metric/cache helper**

Wrap the existing `hettree_lite` full-graph evaluation, return metrics and deterministic teacher logits/predictions suitable for selector features, and write required teacher artifacts when an output directory is provided.

- [ ] **Step 3: Implement support feature construction**

Build raw/type-padded features, degree profiles, relation footprints, class footprints, anchor summaries, target-response signatures, relation-response signatures, and required diagnostics without using test labels.

- [ ] **Step 4: Implement contribution and selector variants**

Implement teacher top-k, teacher diverse top-k, hybrid teacher-response, validation-greedy proxy, and selected-background variants as deterministic budgeted selectors.

- [ ] **Step 5: Implement condensation and pipeline**

Build target-singleton assignments with selected support singletons and typed background buckets; run the existing `hettree_lite` evaluator and report recovery vs teacher/full-graph ceiling.

### Task 3: Gate15 Experiment And Summary Scripts

**Files:**
- Create: `experiments/scripts/run_gate15_supervised_support_selection.py`
- Create: `experiments/scripts/summarize_gate15_supervised_support_selection.py`

- [ ] **Step 1: Implement Gate15 runner**

Run ACM/DBLP/IMDB, five seeds, six support ratios, five HeSF-SS methods, Gate14 references, support-only baselines, and full-graph ceilings. Write the required `teacher/`, `selection/`, `graphs/`, `runs/`, `summary/`, and `figures/` outputs.

- [ ] **Step 2: Implement summary/decision**

Aggregate by method/ratio/dataset, compute ratio-matched gaps, recovery vs ceiling, validation-selected test rows, accuracy-budget curve, figures, and a clear Gate15 decision.

### Task 4: Verification And Commit

**Files:**
- All Gate15 files above.

- [ ] **Step 1: Run required pytest command**

Run the exact Gate15 pytest command from the prompt.

- [ ] **Step 2: Run py_compile and diff checks**

Run `py_compile` on new and modified Python files and `git diff --check`.

- [ ] **Step 3: Run the full local Gate15 experiment**

Run in conda env `pytorch`, stopping only if local OOM occurs. If OOM occurs, capture the local command and produce the server command.

- [ ] **Step 4: Commit Gate15 code changes to `main`**

Stage only Gate15 code/test/plan files and commit with a concise Gate15 message.
