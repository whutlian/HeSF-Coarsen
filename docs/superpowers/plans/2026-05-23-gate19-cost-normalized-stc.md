# Gate19 Cost-Normalized STC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Gate19 as a cost-normalized semantic-tree condensation stage with faithful Full-STC baselines, real teacher-student distillation diagnostics, and DBLP-primary decision logic.

**Architecture:** Add focused cost-accounting and STC baseline/distillation modules under `hesf_coarsen/task_first/`, then add Gate19 runner/summarizer scripts that write the required main and diagnostic outputs. Keep raw support/ClusterGate methods diagnostic and make Pareto/decision logic use `total_storage_ratio_vs_full_stc` for STC.

**Tech Stack:** Python, NumPy, PyTorch in conda env `pytorch`, existing HGB loaders/evaluators, CSV/JSON outputs.

---

### Task 1: RED Tests

**Files:**
- Create: `tests/test_gate19_cost_normalized_stc.py`

- [x] Write tests for nonzero STC cost, Full-STC ratio one, compressed ratio below one, teacher KL not self-reference, no test leakage flags, header normalization, and Pareto storage-axis use.
- [x] Run `conda run -n pytorch python -m pytest tests/test_gate19_cost_normalized_stc.py -q` and confirm failure before implementation.

### Task 2: Cost Accounting

**Files:**
- Create: `hesf_coarsen/task_first/costs/__init__.py`
- Create: `hesf_coarsen/task_first/costs/accounting.py`
- Create: `hesf_coarsen/task_first/costs/reports.py`

- [x] Implement `CompressionCost` exactly with Gate19 fields.
- [x] Implement cache byte counting, model byte counting, total storage ratios, finite assertions, and CSV row conversion.
- [x] Verify tests covering STC support ratio zero but nonzero storage pass.

### Task 3: Full-STC Baselines and Compression Utilities

**Files:**
- Create: `hesf_coarsen/task_first/feature_condensation/baselines.py`
- Create: `hesf_coarsen/task_first/feature_condensation/distillation.py`
- Modify: `hesf_coarsen/task_first/feature_condensation/__init__.py`

- [x] Implement Full-STC MLP, calibrated MLP, linear, and centroid baselines using train labels only and validation metrics for selection.
- [x] Implement compressed cache helpers for path energy, validation accuracy/loss ranking, hard gate, fp16/int8 quantization.
- [x] Implement `TeacherLogits` and true teacher-student KL training; never self-reference teacher logits.
- [x] Verify teacher KL status tests pass.

### Task 4: Gate19 Runner

**Files:**
- Create: `experiments/scripts/run_gate19_cost_normalized_stc.py`

- [x] Add required CLI flags and dataset seed parser.
- [x] Write code sync report and method-to-code-path audit.
- [x] Evaluate support baselines, Full-STC baselines, STC compressed methods, true distillation, quantized methods, and lightweight ClusterGate diagnostics.
- [x] Write required raw, cost, baseline, feature, distillation, teacher, calibration, per-class, confusion, leakage, cache-size, and evaluator audit files.

### Task 5: Gate19 Summarizer

**Files:**
- Create: `experiments/scripts/summarize_gate19.py`

- [x] Implement header normalization, CSV read, dataset integrity, validation-selected, Pareto using total storage ratio, and result JSON fields.
- [x] Enforce `compressed_projected`, no leakage, Full-STC availability, cost accounting, and teacher KL validity.
- [x] Write decision markdown with DBLP primary and ACM sanity-only treatment.

### Task 6: Formal Run and Verification

**Commands:**
- `conda run -n pytorch python -m py_compile experiments/scripts/run_gate19_cost_normalized_stc.py`
- `conda run -n pytorch python -m py_compile experiments/scripts/summarize_gate19.py`
- `conda run -n pytorch python -m pytest tests/test_gate19_cost_normalized_stc.py -q`
- `conda run -n pytorch python experiments/scripts/run_gate19_cost_normalized_stc.py --output-dir outputs/gate19 --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --cost-budgets 0.30 0.50 0.70 1.00 --support-ratios 0.30 0.50 0.70 --primary-eval-mode compressed_projected --task-epochs 10 --max-paths 2 --feature-mode full --include-typedhash true --return-logits true --include-full-stc-baselines true --include-true-distillation true`

- [x] If local OOM occurs, stop and return a server command instead of silently weakening the matrix.
- [x] Write `outputs/gate19/code_change_report.md` and `outputs/gate19/gate19_requirement_checklist.md`.
- [x] Run final verification and commit/push code changes to `main`.
