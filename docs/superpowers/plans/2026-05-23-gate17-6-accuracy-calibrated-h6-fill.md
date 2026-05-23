# Gate17.6 Accuracy-Calibrated H6 Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run Gate17.6 exactly as the prompt requires: Accuracy-Calibrated H6-Assisted Validation Fill with TypedHash closure, H6-fill ablation, per-class audit, and a final requirement checklist.

**Architecture:** Keep Gate17.5 intact and add Gate17.6 as a new runner plus summarizer. Reuse Gate17.5 normalized CSV helpers, selection pipeline, budget diagnostics, H6 equivalence controls, and H6 fill helpers; add only the extra Gate17.6 scoring, fill provenance, per-class diagnostics, and decision rules.

**Tech Stack:** Python, pytest, NumPy, existing HeSF-Coarsen experiment scripts, local conda environment `pytorch`.

---

### Task 1: RED Tests For Gate17.6 Contract

**Files:**
- Create: `tests/test_gate17_6_runner_summary.py`

- [ ] **Step 1: Write tests before implementation**

Test requirements:
- `parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])` returns exactly those three pairs, not a Cartesian product.
- `DEFAULT_METHODS` includes TypedHash and every required Gate17.6 main/diagnostic method.
- Accuracy objective components include `validation_micro_f1_available=false` when micro is unavailable and compute all score component fields.
- Summary excludes `HeSF-SS-H6-fill-only` from `best_eligible_method`.
- Summary chooses among eligible methods by DBLP 0.30/0.70 macro pass first, then DBLP mean accuracy gap.
- Summary writes the required Gate17.6 main output names and blocks Gate18 when DBLP accuracy is below `-0.005`.

- [ ] **Step 2: Run RED**

Run:
```powershell
conda run -n pytorch python -m pytest tests/test_gate17_6_runner_summary.py -q
```

Expected: fail because Gate17.6 modules do not exist yet.

### Task 2: Evaluation Diagnostics

**Files:**
- Modify: `hesf_coarsen/eval/hettree_task.py`

- [ ] **Step 1: Add optional prediction diagnostics**

Add a default-off `return_predictions: bool = False` argument to `evaluate_hettree_task`. When true, include projected test/val true and predicted labels, test nodes, and predicted class histogram in `TaskEvalResult.metrics`.

- [ ] **Step 2: Verify existing tests**

Run:
```powershell
conda run -n pytorch python -m pytest tests/test_gate17_5_runner_summary.py tests/test_gate17_5_header_normalization.py tests/test_gate17_5_h6_cluster_gating.py -q
```

Expected: pass; default behavior remains unchanged.

### Task 3: Gate17.6 Runner

**Files:**
- Create: `experiments/scripts/run_gate17_6_accuracy_calibrated_h6_fill.py`

- [ ] **Step 1: Define constants and CLI**

Runner must expose the prompt CLI defaults:
```text
--dataset-seeds ACM:23456 DBLP:23456 IMDB:45678
--support-ratios 0.30 0.70
--primary-eval-mode compressed_projected
--monitor projected_val_macro_f1
--include-typedhash true
--alpha-accuracy-grid 0.0 0.25 0.50 1.00
```

- [ ] **Step 2: Implement methods and provenance**

Implement all required baselines, main candidates, and diagnostic-only methods. Record provenance counts:
`validation_core_support_count`, `validation_neutral_support_count`, `validation_negative_support_count`, `h6_fill_support_count`, `random_fill_support_count`, `h6_fill_only_support_count`, `selected_support_count`, `requested_support_count`.

- [ ] **Step 3: Write diagnostics**

Write:
`diagnostics/gate17_6_fill_ablation.csv`,
`diagnostics/gate17_6_per_class_metrics.csv`,
`diagnostics/gate17_6_confusion_matrix_by_method.csv`,
`diagnostics/gate17_6_h6_cluster_coverage_diagnostics.csv`,
`diagnostics/gate17_6_typedhash_baseline_check.csv`,
`diagnostics/gate17_6_accuracy_objective_components.csv`,
`diagnostics/gate17_6_budget_breakdown.csv`,
`diagnostics/gate17_6_header_normalization_check.csv`.

### Task 4: Gate17.6 Summarizer

**Files:**
- Create: `experiments/scripts/summarize_gate17_6.py`

- [ ] **Step 1: Reuse Gate17 normalized helpers**

Import `normalize_header`, `read_csv`, `assert_dataset_integrity`, and `validation_selected` from `experiments/scripts/summarize_gate17.py`.

- [ ] **Step 2: Implement eligibility and decision**

Eligibility must exclude diagnostics, baselines, and `HeSF-SS-H6-fill-only`. Best method selection order:
DBLP 0.30 macro pass, DBLP 0.70 macro pass, DBLP mean accuracy gap, overall exact macro gap, validation macro-F1.

- [ ] **Step 3: Write all summary outputs**

Write:
`gate17_6_validation_selected_by_method.csv`,
`gate17_6_exact_budget_paired_gaps.csv`,
`gate17_6_by_dataset_selected.csv`,
`gate17_6_result.json`,
`gate17_6_decision.md`,
`gate17_6_final_report.md`,
and `diagnostics/gate17_6_class_shift_report.md`.

### Task 5: Formal Run And Verification

**Files:**
- Output only: `outputs/gate17_6_accuracy_calibrated_h6_fill/**`
- Create: `outputs/gate17_6_accuracy_calibrated_h6_fill/gate17_6_requirement_checklist.md`

- [ ] **Step 1: Run tests**

Run:
```powershell
conda run -n pytorch python -m pytest tests/test_gate17_6_runner_summary.py tests/test_gate17_5_runner_summary.py tests/test_gate17_5_header_normalization.py tests/test_gate17_5_h6_cluster_gating.py -q
```

- [ ] **Step 2: Run Gate17.6 formal experiment**

Run:
```powershell
conda run -n pytorch python -m experiments.scripts.run_gate17_6_accuracy_calibrated_h6_fill --output-dir outputs/gate17_6_accuracy_calibrated_h6_fill --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --support-ratios 0.30 0.70 --task-epochs 5 --max-paths 2 --feature-mode full --primary-eval-mode compressed_projected --include-typedhash true --alpha-accuracy-grid 0.0 0.25 0.50 1.00 --delta-micro 0.25 --beta-underfill 0.10 --gamma-class-collapse 0.05 --neutral-fill-max-drop 1e-4 --negative-fill-max-drop 5e-4
```

- [ ] **Step 3: Verify prompt checklist**

Create `gate17_6_requirement_checklist.md` and check items 0-20 from the prompt, including TypedHash, primary eval mode, H6-fill-only exclusion, per-class/confusion files, and nonzero H6 cluster diagnostics.

- [ ] **Step 4: Commit and push code changes**

Stage code/tests/plan only, commit to `main`, and push to GitHub `main`. Do not stage generated `outputs/` unless explicitly requested.
