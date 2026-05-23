# Gate20-CAL Multiseed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Gate20-CAL multi-seed validation for task-calibrated compressed support graphs, with STC frozen as diagnostic-only.

**Architecture:** The Gate20 runner reuses Gate19.2 support-graph evaluation and calibration primitives, but narrows the experiment to five seeds, three support ratios, formal uncalibrated support baselines, and canonical `HeSF-CAL-*` methods. The summarizer binds nested calibration statistics to the exact selected DBLP method/ratio and keeps all decisions validation-selected. The official bridge script only reports official evaluator availability and never fabricates official results from the lite evaluator.

**Tech Stack:** Python, NumPy, existing HeSF-Coarsen HGB loaders, `hettree_lite` evaluator, CSV/JSON artifacts, local conda env `pytorch`.

---

### Task 1: Gate20 Summary Tests

**Files:**
- Create: `tests/test_gate20_cal_multiseed.py`
- Create: `experiments/scripts/summarize_gate20_cal.py`

- [ ] **Step 1: Write failing tests**

```python
from experiments.scripts.summarize_gate20_cal import build_gate20_pareto, summarize_rows, validation_selected_rows

def test_gate20_binds_nested_stats_to_exact_best_method_ratio():
    rows = [
        {"dataset": "DBLP", "seed": 12345, "method": "HeSF-CAL-best-support", "method_family": "hesf_cal", "ratio": 0.30, "accuracy": 0.90, "macro_f1": 0.89, "validation_accuracy": 0.91, "validation_macro_f1": 0.90, "total_storage_ratio_vs_full_stc": 0.03, "diagnostic_only": False, "eligible_for_main_decision": True, "primary_eval_mode": "compressed_projected", "no_test_leakage": True, "calibration_uses_test_labels": False},
        {"dataset": "DBLP", "seed": 12345, "method": "HeSF-CAL-H6", "method_family": "hesf_cal", "ratio": 0.50, "accuracy": 0.93, "macro_f1": 0.92, "validation_accuracy": 0.80, "validation_macro_f1": 0.80, "total_storage_ratio_vs_full_stc": 0.04, "diagnostic_only": False, "eligible_for_main_decision": True, "primary_eval_mode": "compressed_projected", "no_test_leakage": True, "calibration_uses_test_labels": False},
    ]
    nested = [
        {"dataset": "DBLP", "method": "HeSF-CAL-best-support", "ratio": 0.30, "nested_accuracy_std": 0.014, "nested_macro_std": 0.013, "calibration_constraint_satisfied_rate": 1.0, "nested_accuracy_mean": 0.90, "nested_macro_mean": 0.89},
        {"dataset": "DBLP", "method": "HeSF-CAL-H6", "ratio": 0.50, "nested_accuracy_std": 0.0, "nested_macro_std": 0.0, "calibration_constraint_satisfied_rate": 1.0, "nested_accuracy_mean": 0.93, "nested_macro_mean": 0.92},
    ]
    result = summarize_rows(rows, nested_rows=nested, quality_rows=[], per_class_present=True, confusion_present=True)
    assert result["best_method"] == "HeSF-CAL-best-support"
    assert result["best_method_ratio"] == 0.30
    assert result["best_method_nested_accuracy_std"] == 0.014
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n pytorch python -m pytest tests/test_gate20_cal_multiseed.py -q`
Expected: FAIL because `experiments.scripts.summarize_gate20_cal` does not exist.

- [ ] **Step 3: Implement summarizer**

Implement validation-selected rows, exact-method/ratio nested lookup, Pareto, result JSON decision labels, exact-ratio comparison, and required diagnostic summaries in `experiments/scripts/summarize_gate20_cal.py`.

- [ ] **Step 4: Run tests**

Run: `conda run -n pytorch python -m pytest tests/test_gate20_cal_multiseed.py -q`
Expected: PASS.

### Task 2: Gate20 Runner

**Files:**
- Create: `experiments/scripts/run_gate20_cal_multiseed.py`
- Modify: `tests/test_gate20_cal_multiseed.py`

- [ ] **Step 1: Add failing runner tests**

Add tests that check calibration candidate logging, no test leakage fields, metadata fields, and diagnostic-only exclusion for ensemble/STC placeholder rows.

- [ ] **Step 2: Run tests to verify failure**

Run: `conda run -n pytorch python -m pytest tests/test_gate20_cal_multiseed.py -q`
Expected: FAIL because the runner functions do not exist.

- [ ] **Step 3: Implement runner**

Implement dataset/seed/ratio loops, uncalibrated baselines, canonical HeSF-CAL calibration, best-support validation selection, quality metrics, per-class/confusion diagnostics, method path and leakage audits, reproducibility metadata, and requirement checklist.

- [ ] **Step 4: Smoke run**

Run: `conda run -n pytorch python -m experiments.scripts.run_gate20_cal_multiseed --datasets DBLP --seeds 12345 --support-ratios 0.30 --nested-split-seeds 11 --primary-eval-mode compressed_projected --output-dir outputs/gate20_cal_smoke`
Expected: required Gate20 files exist and summarizer completes.

### Task 3: Official Bridge v0

**Files:**
- Create: `experiments/scripts/run_gate20_official_evaluator_bridge_v0.py`

- [ ] **Step 1: Add bridge script**

Write a minimal official bridge script that probes for official HETTREE/SeHGNN dependencies, writes unavailable status when missing, and does not use lite evaluator as official output.

- [ ] **Step 2: Run bridge**

Run: `conda run -n pytorch python -m experiments.scripts.run_gate20_official_evaluator_bridge_v0 --datasets DBLP ACM --output-dir outputs/gate20_official_bridge_v0`
Expected: status, results, and missing-items files are written.

### Task 4: Full Experiment and Verification

**Files:**
- Output only under `outputs/gate20_cal` and `outputs/gate20_official_bridge_v0`

- [ ] **Step 1: Run the formal Gate20 command**

Run: `conda run -n pytorch python -m experiments.scripts.run_gate20_cal_multiseed --datasets ACM DBLP IMDB --seeds 12345 23456 34567 45678 56789 --support-ratios 0.30 0.50 0.70 --primary-eval-mode compressed_projected --output-dir outputs/gate20_cal`

- [ ] **Step 2: Run standalone summarizer**

Run: `conda run -n pytorch python -m experiments.scripts.summarize_gate20_cal --input-dir outputs/gate20_cal --output-dir outputs/gate20_cal`

- [ ] **Step 3: Verify**

Run tests, py_compile, `git diff --check`, inspect `gate20_cal_requirement_checklist.md`, and commit/push code changes to `main`.
