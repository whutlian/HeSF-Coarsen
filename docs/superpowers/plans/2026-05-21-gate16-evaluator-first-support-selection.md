# Gate16 Evaluator-First Support Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Gate16 evaluator-first support compression so the primary metric is compressed/projected target prediction, then rerun teacher, selector, prototype, and exact-budget diagnostics.

**Architecture:** Patch `evaluate_hettree_task()` first because every downstream Gate16 result depends on its primary metric. Then update the Gate15 selection stack into Gate16 by adding credible teacher metadata, explicit selector diagnostics, prototype residual background condensation, exact budget accounting, and scripts that produce the required output tables.

**Tech Stack:** Python, NumPy, PyTorch through local conda env `pytorch`, existing `HeteroGraph`/`Assignment` data model, pytest, CSV/JSON/Markdown outputs.

---

### Task 1: Code Audit Outputs

**Files:**
- Create: `experiments/scripts/audit_gate16_code.py`
- Output: `outputs/gate16_code_audit/code_sync_report.md`
- Output: `outputs/gate16_code_audit/method_to_code_path.csv`

- [ ] **Step 1: Implement audit script**

Create `write_audit(output: Path)` that writes the required git state, path existence checks, and method-to-code rows for full-graph, H6, flatten, TypedHash, random, and Gate15 HeSF-SS selectors.

- [ ] **Step 2: Run audit script**

Run: `conda run -n pytorch python experiments/scripts/audit_gate16_code.py --output outputs/gate16_code_audit`

Expected: both required files exist.

### Task 2: Evaluator Primary Mode

**Files:**
- Modify: `hesf_coarsen/eval/hettree_task.py`
- Test: `tests/test_hettree_primary_eval_mode.py`

- [ ] **Step 1: Write failing tests**

Add tests that build a tiny hetero graph, call `evaluate_hettree_task(..., primary_eval_mode="compressed_projected")`, assert `macro_f1 == projected_original_macro_f1`, then call `primary_eval_mode="original_transfer"` and assert `macro_f1 == transfer_original_macro_f1`. Also assert `validation_macro_f1`, `primary_eval_mode`, and gap fields exist.

- [ ] **Step 2: Verify RED**

Run: `conda run -n pytorch python -m pytest tests/test_hettree_primary_eval_mode.py -q`

Expected: fails because `primary_eval_mode` is unsupported.

- [ ] **Step 3: Implement evaluator patch**

Add `primary_eval_mode`, validation projected/transfer/hybrid metrics, `projected_vs_transfer_*_gap`, and early stopping with `monitor="projected_val_macro_f1"`. Preserve old metric fields and make `compressed_projected` the default primary path.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n pytorch python -m pytest tests/test_hettree_primary_eval_mode.py tests/test_hettree_task.py -q`

Expected: all tests pass.

### Task 3: Gate16 Selection Stack

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/teacher.py`
- Modify: `hesf_coarsen/task_first/selection/contribution.py`
- Modify: `hesf_coarsen/task_first/selection/selector.py`
- Modify: `hesf_coarsen/task_first/selection/condensation.py`
- Modify: `hesf_coarsen/task_first/selection/pipeline.py`
- Create: `hesf_coarsen/task_first/selection/budget.py`
- Test: `tests/test_gate16_selection_budget.py`
- Test: `tests/test_gate16_prototype_condensation.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert exact budget accounting, `validation_proxy_diverse` is not true validation feedback, `sensitivity_block_selector` returns budget-exact selections, and `class_anchor_relation_prototype` produces more granular background diagnostics than `typed_background`.

- [ ] **Step 2: Verify RED**

Run: `conda run -n pytorch python -m pytest tests/test_gate16_selection_budget.py tests/test_gate16_prototype_condensation.py -q`

Expected: fails because Gate16 selectors/budget/prototype fields do not exist.

- [ ] **Step 3: Implement minimal Gate16 stack**

Extend configs, add budget utilities, rename fake `validation_greedy` output to `validation_proxy_diverse`, add deterministic block sensitivity selector, add prototype residual condensation and diagnostics, and force `primary_eval_mode="compressed_projected"` in the selection pipeline.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n pytorch python -m pytest tests/test_gate15_*.py tests/test_gate16_selection_budget.py tests/test_gate16_prototype_condensation.py -q`

Expected: Gate15 compatibility and Gate16 behavior tests pass.

### Task 4: Gate16 Scripts And Summaries

**Files:**
- Create: `experiments/scripts/run_gate16_evaluator_patch.py`
- Create: `experiments/scripts/run_gate16_teacher_stability.py`
- Create: `experiments/scripts/run_gate16_support_selection.py`
- Create: `experiments/scripts/summarize_gate16.py`

- [ ] **Step 1: Implement scripts**

Scripts must generate the required `outputs/gate16_evaluator`, `outputs/gate16_teacher`, `outputs/gate16_tables`, `outputs/gate16_diag`, and `outputs/gate16_smoke` files, with no test leakage flags.

- [ ] **Step 2: Smoke run**

Run: `conda run -n pytorch python experiments/scripts/run_gate16_support_selection.py --smoke --output-root outputs`

Expected: `outputs/gate16_smoke/gate16_smoke_all_runs.csv` and `gate16_smoke_report.md` exist and have no unexplained skipped rows.

- [ ] **Step 3: Full Gate16 run**

Run the full ACM/DBLP/IMDB, 5 seeds, 6 ratios matrix with local conda env `pytorch`. If an OOM/GPU OOM occurs, stop the local run and report a server command instead.

### Task 5: Verification, Commit, Push

**Files:**
- All Gate16 code, tests, and required outputs.

- [ ] **Step 1: Run focused tests**

Run the Gate16 and regression pytest commands under `conda run -n pytorch`.

- [ ] **Step 2: Verify required outputs**

Check all required files listed in the prompt exist and have nonzero size.

- [ ] **Step 3: Stage only Gate16-owned files**

Do not stage unrelated existing untracked files like `build/`, `exports/`, `session.md`, or old docs artifacts.

- [ ] **Step 4: Commit and push main**

Commit with a Gate16 message and push `main` to `https://github.com/whutlian/HeSF-Coarsen.git`.

---

**Spec coverage check:** This plan covers audit, evaluator patch, pipeline metric closure, teacher metadata/stability, selector rename/sensitivity hooks, prototype condensation, exact budgets, Gate16 scripts, summaries, tests, smoke/full runs, and GitHub push. The full teacher is implemented as local `hettree_lite` tuning/logit export, with official evaluator status explicitly marked unavailable unless later integrated.
