# Gate18R Accuracy-First Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Gate18R as an accuracy-first reset that tests validation-only logit calibration, Pareto upper-bound selection, cluster-unit gating, and semantic-tree feature condensation.

**Architecture:** Add Gate18R scripts and modules without overwriting Gate17.6. Reuse existing Gate17.6 selection/evaluation plumbing where it is still valid, but change ranking to Pareto upper-bound comparisons and add validation-only calibration, cluster unit inventories/scores, and STC diagnostics.

**Tech Stack:** Python, NumPy, PyTorch through existing local conda env `pytorch`, pytest, current HeSF-Coarsen experiment utilities.

---

### Task 1: RED Contract Tests

**Files:**
- Create: `tests/test_gate18r_accuracy_reset.py`

- [ ] **Step 1: Write failing tests**

Tests must cover:
- `parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])` returns explicit pairs, not Cartesian product.
- Gate18R default methods include all required baselines, new candidates, and old ablations.
- Calibration grid never uses test labels and improves/keeps validation objective using only val logits/labels.
- Support units are non-empty and score includes accuracy-first terms.
- Pareto summary excludes H6-fill-only/full-residual upperbound from final best and writes required result fields.

- [ ] **Step 2: Run RED**

Run:
```powershell
conda run -n pytorch python -m pytest tests/test_gate18r_accuracy_reset.py -q
```

Expected: fail because Gate18R modules do not exist yet.

### Task 2: Evaluator Logit Payload And Calibration

**Files:**
- Modify: `hesf_coarsen/eval/hettree_task.py`
- Create: `hesf_coarsen/eval/calibration.py`

- [ ] **Step 1: Add optional logits**

Add default-off arguments:
`return_logits`, `return_val_logits`, `return_test_logits`, `return_prediction_payload`.

When enabled, metrics include:
`projected_val_logits`, `projected_test_logits`, `projected_val_labels`, `projected_test_labels`, `projected_val_nodes`, `projected_test_nodes`, `projected_val_pred`, `projected_test_pred`, `train_class_prior`, `val_class_prior`, `num_classes`.

- [ ] **Step 2: Add validation-only calibration**

Implement temperature scaling, class-bias adjustment, grid search, and macro-constrained accuracy selection.

### Task 3: Cluster Unit Package

**Files:**
- Create package: `hesf_coarsen/task_first/units/`

- [ ] **Step 1: Add unit dataclass and extractors**

Implement `SupportUnit` and extractors for H6, TypedHash, flatten, validation blocks, plus union units.

- [ ] **Step 2: Add scoring and gated graph helpers**

Implement accuracy-first unit scoring and support-unit selection under an upper-bound budget. Preserve source coarse assignment semantics where available.

### Task 4: Semantic Tree Condensation Package

**Files:**
- Create package: `hesf_coarsen/task_first/feature_condensation/`

- [ ] **Step 1: Add semantic tree cache helpers**

Build/save/load full and compressed semantic-tree caches.

- [ ] **Step 2: Add path prune/prototype/distill candidates**

Implement lightweight path pruning, path prototypes, feature-cache distillation, and cached evaluation.

### Task 5: Gate18R Runner And Summarizer

**Files:**
- Create: `experiments/scripts/run_gate18r_accuracy_first_reset.py`
- Create: `experiments/scripts/summarize_gate18r.py`

- [ ] **Step 1: Runner CLI**

Support:
```powershell
python -m experiments.scripts.run_gate18r_accuracy_first_reset --output-dir outputs/gate18r --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --support-ratios 0.30 0.50 0.70 --primary-eval-mode compressed_projected --task-epochs 10 --max-paths 2 --feature-mode full --include-typedhash true --return-logits true
```

- [ ] **Step 2: Required outputs**

Main:
`gate18r_raw_rows.csv`, `gate18r_validation_selected_by_method.csv`, `gate18r_pareto_frontier.csv`, `gate18r_by_dataset_selected.csv`, `gate18r_result.json`, `gate18r_decision.md`.

Diagnostics:
`gate18r_calibration.csv`, `gate18r_per_class_metrics.csv`, `gate18r_confusion_matrix_by_method.csv`, `gate18r_unit_inventory.csv`, `gate18r_unit_scores.csv`, `gate18r_selected_units.csv`, `gate18r_unit_overlap.csv`, `gate18r_feature_condensation.csv`, `gate18r_evaluator_ceiling_audit.csv`.

- [ ] **Step 3: Decision logic**

Do not gate on ACM. Gate18 entry requires DBLP 0.30/0.70 macro frontier gaps `>=0`, accuracy gaps `>= -0.005`, no IMDB collapse vs TypedHash, no leakage, compressed projected mode, TypedHash included, and best method not H6-fill-only/full-residual upperbound.

### Task 6: Formal Run, Checklist, Commit

**Files:**
- Output: `outputs/gate18r/**`
- Create: `outputs/gate18r/gate18r_requirement_checklist.md`
- Create: `outputs/gate18r/code_change_report.md`

- [ ] **Step 1: Run verification**

Run:
```powershell
conda run -n pytorch python -m pytest tests/test_gate18r_accuracy_reset.py tests/test_gate17_6_runner_summary.py -q
conda run -n pytorch python -m py_compile experiments/scripts/run_gate18r_accuracy_first_reset.py experiments/scripts/summarize_gate18r.py
```

- [ ] **Step 2: Run formal Gate18R**

Use local env:
```powershell
conda run -n pytorch python -m experiments.scripts.run_gate18r_accuracy_first_reset --output-dir outputs/gate18r --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --support-ratios 0.30 0.50 0.70 --primary-eval-mode compressed_projected --task-epochs 10 --max-paths 2 --feature-mode full --include-typedhash true --return-logits true
```

- [ ] **Step 3: Final verification**

Run full pytest, `git diff --check`, and write the requirement checklist covering prompt sections 0-15.

- [ ] **Step 4: Commit and push**

Stage only Gate18R source/tests/plan, commit to `main`, and push `origin main`.
