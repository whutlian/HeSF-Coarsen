# Gate17.3 Lossy Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the single-seed Gate17.3 lossy-prototype / feedback disentanglement gate without full residual shortcuts entering the main decision.

**Architecture:** Reuse the Gate17.2 runner/summary shape, but add explicit residual prototype modes and Gate17.3 budget fields. Selection-only uses an induced target+selected-support graph; lossy prototype keeps only bounded residual prototypes; full residual prototypes are diagnostic upper bounds only.

**Tech Stack:** Python, NumPy, pytest, local conda environment `pytorch`, existing HeSF `HeteroGraph` / task-first selection pipeline.

---

### Task 1: Gate17.3 Tests and Contracts

**Files:**
- Create: `tests/test_gate17_3_budget_modes.py`
- Create: `tests/test_gate17_3_runner_summary.py`
- Modify: `tests/test_gate17_2_feedback_no_fallback.py` if occlusion task-signal fields need compatibility assertions.

- [ ] **Step 1: Write failing tests for explicit prototype modes**

Test that `SupportSelectorConfig` exposes `residual_prototype_mode`, selection-only drops unselected support, lossy mode bounds represented context, and full-upperbound is marked diagnostic-only.

- [ ] **Step 2: Write failing tests for runner seed parsing and method eligibility**

Test `parse_dataset_seeds("ACM:23456 DBLP:23456 IMDB:45678")`, exact non-Cartesian pairs, `DEFAULT_METHODS`, diagnostic-only methods, and `eligible_for_main_decision` rules.

- [ ] **Step 3: Write failing tests for summary failure reasons**

Synthetic rows must trigger `FAIL_REPRESENTED_CONTEXT_BUDGET`, exclude full-residual upperbound from best method, compute DBLP gap vs H6/flatten, and write all required Gate17.3 files.

- [ ] **Step 4: Verify tests fail before implementation**

Run:

```powershell
conda run -n pytorch python -m pytest tests/test_gate17_3_budget_modes.py tests/test_gate17_3_runner_summary.py -q
```

Expected: failures from missing Gate17.3 modules/config/fields.

### Task 2: Config, Selection Modes, and Budget Diagnostics

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/condensation.py`
- Modify: `hesf_coarsen/task_first/selection/pipeline.py`
- Create: `experiments/scripts/gate17_3_budget.py`

- [ ] **Step 1: Add config fields**

Add residual mode, lossy prototype budget fields, min occlusion/validation fields, neutral fill fields, and meta-path channel source fields without breaking existing Gate17.2 defaults.

- [ ] **Step 2: Implement selection-only induced graph helper**

Add `build_induced_target_support_graph(...)` that preserves all target nodes, keeps selected support only, filters relations, reindexes nodes, and returns assignment/mapping diagnostics compatible with semantic-tree evaluation.

- [ ] **Step 3: Implement lossy/full prototype mode selection**

In `build_selected_support_graph`, keep current prototype behavior for `full_upperbound`, drop all residual support for `none`, and retain only top bounded prototype blocks for `lossy_topk`.

- [ ] **Step 4: Add Gate17.3 budget split helper**

Compute `node_budget_count`, `node_budget_ratio`, `node_budget_exact_match`, `represented_context_count`, `represented_context_ratio`, `represented_context_exact_or_bounded`, `prototype_member_budget_total`, `full_residual_upperbound`, and `eligible_for_main_decision`.

### Task 3: Feedback and H6 Diagnostics

**Files:**
- Modify: `hesf_coarsen/task_first/selection/validation_selector.py`
- Modify: `experiments/scripts/run_gate17_support_selection.py`
- Modify: `hesf_coarsen/task_first/selection/pipeline.py`
- Modify: `experiments/scripts/gate13_task_first_common.py` only if H6 selected nodes are not exposed by baseline diagnostics.

- [ ] **Step 1: Split occlusion task vs tree signal**

Add per-block `occlusion_task_nonzero_delta`, `occlusion_tree_nonzero_delta` and aggregate rates/pass flags; main summaries must use task signal, not tree-only signal.

- [ ] **Step 2: Add no-negative / neutral-fill support**

For no-fallback occlusion, only select positive blocks and allow underfill; for neutral-fill, fill exact budget from neutral or weakly positive blocks without negative contribution.

- [ ] **Step 3: Add validation neutral-fill support**

Positive validation-gain blocks first; then exact-budget neutral fill by validation score drop threshold, without arbitrary proxy fill.

- [ ] **Step 4: Export H6 overlap diagnostics**

For every dataset/seed/ratio, record H6 selected support nodes and overlap metrics for H6-seeded methods.

### Task 4: Gate17.3 Runner

**Files:**
- Create: `experiments/scripts/run_gate17_3_lossy_prototype_feedback.py`

- [ ] **Step 1: Reuse Gate17.2 loop shape**

Support `--dataset-seeds ACM:23456 DBLP:23456 IMDB:45678`, `--support-ratios`, `--task-epochs`, `--max-paths`, `--feature-mode full`, and write under `outputs/gate17_3_single_seed`.

- [ ] **Step 2: Implement required methods**

Baselines: full graph, target-only, H6, flatten, random, optional TypedHash. Main candidates: selection-only, neutral-fill, lossy prototype, H6-seeded variants. Diagnostics: validation no-fallback and full residual upperbound.

- [ ] **Step 3: Write required raw and diagnostic files**

Write `gate17_3_raw_rows.csv` and every `diagnostics/gate17_3_*.csv` required by the prompt, including ACM saturation, H6 overlap, and meta-path audit.

### Task 5: Gate17.3 Summary

**Files:**
- Create: `experiments/scripts/summarize_gate17_3.py`

- [ ] **Step 1: Implement eligibility and paired gaps**

Exclude upperbound/full-residual/no-fallback/non-budget rows from best main method and exact-budget paired gaps.

- [ ] **Step 2: Implement failure priority and decisions**

Use prompt priority order and allowed decisions; include all explicit `result.json` fields and single-seed diagnostic warning in reports.

- [ ] **Step 3: Verify synthetic summary tests pass**

Run the Gate17.3 summary tests and confirm exact failure labels.

### Task 6: Verification, Experiment, Commit

**Files:**
- Outputs: `outputs/gate17_3_single_seed/**` (local only unless intentionally force-added; `outputs/` is ignored).

- [ ] **Step 1: Run focused and full tests**

Run:

```powershell
conda run -n pytorch python -m pytest tests/test_gate17_3_budget_modes.py tests/test_gate17_3_runner_summary.py -q
conda run -n pytorch python -m pytest -q
```

- [ ] **Step 2: Run the required local experiment**

Run:

```powershell
conda run -n pytorch python -m experiments.scripts.run_gate17_3_lossy_prototype_feedback --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --support-ratios 0.03 0.10 0.30 0.70 --task-epochs 5 --max-paths 2 --feature-mode full --primary-eval-mode compressed_projected --output-dir outputs/gate17_3_single_seed
```

If local OOM/GPU OOM occurs, stop local execution and return the equivalent server command.

- [ ] **Step 3: Summarize and verify required files**

Run:

```powershell
conda run -n pytorch python -m experiments.scripts.summarize_gate17_3 --input-dir outputs/gate17_3_single_seed --output-dir outputs/gate17_3_single_seed
git diff --check
```

- [ ] **Step 4: Commit and push code to main**

Stage Gate17.3 code/tests/plan only, commit, and push `origin main`.

---

Self-review:

- Spec coverage: Covers explicit residual modes, budget split, real feedback task signal, validation neutral fill, H6-seeded variants, ACM saturation curve, required outputs, summary decisions, local experiment, and main push.
- Placeholder scan: No TBD/TODO/fill-later placeholders.
- Type consistency: Uses prompt field names in runner/summary/tests; helper names are introduced before use.
