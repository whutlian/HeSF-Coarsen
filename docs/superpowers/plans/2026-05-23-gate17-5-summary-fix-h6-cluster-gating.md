# Gate17.5 Summary Fix H6 Cluster Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Gate17 CSV header normalization and run a narrow Gate17.5 single-seed diagnostic with real H6-cluster gating and corrected best-eligible decision logic.

**Architecture:** Keep Gate17.4 H6 artifact/equivalence helpers, add shared normalized CSV helpers in the Gate17 summarizer base, and implement Gate17.5 as a new runner/summarizer pair. Gate17.5 H6-cluster methods must select H6 coarse support units, rebuild an induced H6-style coarse graph, and evaluate it; only controls may reuse H6 task metrics directly.

**Tech Stack:** Python, pytest, NumPy, existing `hesf_coarsen` task-first selection/evaluation utilities, local conda env `pytorch`.

---

### Task 1: Header Normalization And Corrected Gate17.4 Summary

**Files:**
- Modify: `experiments/scripts/summarize_gate17.py`
- Modify: `experiments/scripts/summarize_gate17_1.py`
- Modify: `experiments/scripts/summarize_gate17_2.py`
- Modify: `experiments/scripts/summarize_gate17_3.py`
- Modify: `experiments/scripts/summarize_gate17_4.py`
- Test: `tests/test_gate17_5_header_normalization.py`

- [ ] **Step 1: Write failing header-normalization tests**

Create tests that write a temporary raw CSV with a BOM/quoted dataset header, same seed for ACM/DBLP, and assert normalized `dataset` grouping, nonblank exact-gap dataset, and the diagnostic check output.

- [ ] **Step 2: Verify the tests fail**

Run: `conda run -n pytorch python -m pytest tests/test_gate17_5_header_normalization.py -q`

- [ ] **Step 3: Implement shared helpers**

Add `normalize_header`, `normalize_dataset_value`, `normalize_row`, `read_csv_normalized`, and `assert_dataset_integrity` in `summarize_gate17.py`; make `read_csv` return normalized rows and raise the CSV field size limit.

- [ ] **Step 4: Patch Gate17.x summarizers**

Ensure Gate17.1/2/3/4 raw row paths use normalized rows and call dataset integrity assertions for their expected raw rows.

- [ ] **Step 5: Add corrected Gate17.4 output mode**

When Gate17.4 summarizes into a corrected output directory, write `*_corrected.csv`, `gate17_4_result_corrected.json`, and `gate17_4_decision_corrected.md` in addition to the standard summary files.

### Task 2: Gate17.5 H6 Cluster Gating Helpers

**Files:**
- Create: `hesf_coarsen/task_first/selection/h6_cluster_gating.py`
- Test: `tests/test_gate17_5_h6_cluster_gating.py`

- [ ] **Step 1: Write failing helper tests**

Test that H6 support clusters are extracted from an assignment, budget selection is member-weighted, target coarse nodes are always retained, and the selected cluster graph changes when clusters are gated.

- [ ] **Step 2: Verify the tests fail**

Run: `conda run -n pytorch python -m pytest tests/test_gate17_5_h6_cluster_gating.py -q`

- [ ] **Step 3: Implement helpers**

Implement cluster descriptor extraction, member-weighted greedy cluster selection, H6-derived support-node fill, and induced H6 graph construction wrappers over the existing `induced_coarse_graph`.

### Task 3: Gate17.5 Runner And Real-Validation Fill Variants

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/validation_selector.py`
- Create: `experiments/scripts/run_gate17_5_h6_cluster_gating.py`
- Test: `tests/test_gate17_5_runner_summary.py`

- [ ] **Step 1: Write failing runner/config tests**

Assert default Gate17.5 methods include required baselines, eligible real-validation variants, H6 cluster gated methods, and diagnostic controls; assert H6 controls are diagnostic-only and cluster methods are not copy-metric controls.

- [ ] **Step 2: Verify the tests fail**

Run: `conda run -n pytorch python -m pytest tests/test_gate17_5_runner_summary.py -q`

- [ ] **Step 3: Add fill config fields**

Add `allow_negative_fill`, `negative_fill_max_drop`, `budget_penalty_lambda`, and `underfill_penalty_lambda` to `SupportSelectorConfig`; update real validation block selection diagnostics to log positive/neutral/negative/proxy fill counts and penalty values.

- [ ] **Step 4: Implement Gate17.5 runner**

Reuse Gate17.4 data loading, H6 baseline, H6 equivalence diagnostics, semantic/edge/feature diagnostics, and budget accounting. Add true H6-cluster validation gated and budget-penalty methods that rebuild and evaluate induced H6 graphs.

### Task 4: Gate17.5 Summarizer And Reports

**Files:**
- Create: `experiments/scripts/summarize_gate17_5.py`
- Test: `tests/test_gate17_5_runner_summary.py`

- [ ] **Step 1: Add failing summary tests**

Assert best eligible logic excludes diagnostics and underfilled rows, reports DBLP 0.30/0.70 gaps separately, records missing exact ratios, and never uses ACM saturation as success evidence.

- [ ] **Step 2: Implement summarizer**

Use normalized CSV reading and dataset integrity assertions. Write all required Gate17.5 main outputs and diagnostics, including header normalization check, exact-budget gaps, result JSON, decision markdown, and final report.

### Task 5: Run And Verify

**Files:**
- Output: `outputs/gate17_4_h6_equivalence_corrected/`
- Output: `outputs/gate17_5_h6_cluster_gating/`

- [ ] **Step 1: Run targeted tests**

Run targeted Gate17.5/Gate17.4 tests until green.

- [ ] **Step 2: Recompute corrected Gate17.4**

Run:

```bash
conda run -n pytorch python -m experiments.scripts.summarize_gate17_4 --input-dir outputs/gate17_4_h6_equivalence --output-dir outputs/gate17_4_h6_equivalence_corrected
```

- [ ] **Step 3: Run Gate17.5**

Run:

```bash
conda run -n pytorch python -m experiments.scripts.run_gate17_5_h6_cluster_gating --datasets ACM DBLP IMDB --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --support-ratios 0.30 0.70 --task-epochs 5 --max-paths 2 --feature-mode full --primary-eval-mode compressed_projected --output-dir outputs/gate17_5_h6_cluster_gating
```

- [ ] **Step 4: Summarize Gate17.5**

Run:

```bash
conda run -n pytorch python -m experiments.scripts.summarize_gate17_5 --input-dir outputs/gate17_5_h6_cluster_gating --output-dir outputs/gate17_5_h6_cluster_gating
```

- [ ] **Step 5: Full verification and checklist**

Run full pytest, py_compile, `git diff --check`, parse output files, create `gate17_5_requirement_checklist.md`, commit changes, and push `main`.
