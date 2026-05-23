# Gate19.1 Calibrated Support Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Gate19.1 so calibrated H6 / flatten / TypedHash are formal baselines, nested calibration is audited, per-class/confusion diagnostics are present, and Gate20 is blocked unless STC is Pareto-competitive against calibrated support baselines.

**Architecture:** Reuse Gate19 raw STC and Full-STC outputs as the fixed single-seed STC reference, rerun support baselines locally with logits to produce formal calibrated support rows and nested calibration audits, then summarize all eligible methods with cost-normalized Pareto logic that includes calibrated support baselines. Add small reusable calibration/per-class helpers and a Gate19.1 runner/summarizer pair.

**Tech Stack:** Python, NumPy, PyTorch via conda env `pytorch`, existing HGB loaders/evaluators, CSV/JSON/Markdown outputs.

---

### Task 1: RED Tests

**Files:**
- Create: `tests/test_gate19_1_calibrated_support_audit.py`

- [x] Write failing tests for deterministic nested validation split, calibrated support Pareto eligibility, alias exclusion, calibration cost accounting, and DBLP decision blocking when calibrated support dominates.
- [x] Run `conda run -n pytorch python -m pytest tests/test_gate19_1_calibrated_support_audit.py -q` and verify failure before implementation.

### Task 2: Calibration and Per-Class Utilities

**Files:**
- Modify: `hesf_coarsen/eval/calibration.py`
- Create: `hesf_coarsen/eval/per_class.py`

- [x] Add `apply_logit_calibration`, `calibrate_logits_temperature_bias`, and `nested_calibration_split`.
- [x] Add per-class metrics and confusion row helpers with class support counts and delta fields.
- [x] Verify utility tests pass.

### Task 3: Gate19.1 Summarizer

**Files:**
- Create: `experiments/scripts/summarize_gate19_1.py`

- [x] Normalize headers via Gate19 reader behavior.
- [x] Build Pareto frontier including uncalibrated support, calibrated support, Full-STC references, and compressed STC; exclude aliases/diagnostic true-distill.
- [x] Emit `gate19_1_result.json`, selected tables, and decision markdown with Gate20 blocking logic.
- [x] Verify decision tests pass.

### Task 4: Gate19.1 Runner

**Files:**
- Create: `experiments/scripts/run_gate19_1_calibrated_baseline_audit.py`

- [x] Verify Gate19 runner/summarizer and feature/unit paths are present on main.
- [x] Load Gate19 STC/Full-STC rows, rerun support baselines with logits for ACM/DBLP/IMDB and budgets `0.30 0.50 0.70 1.00`.
- [x] Add formal calibrated support baselines and optional random calibrated sanity rows.
- [x] Add nested calibration audit, leakage audit, per-class/confusion diagnostics, calibration shift report, method aliases, cost breakdown, and zip packages.
- [x] Write `code_sync_report.md`, `method_to_code_path.csv`, `code_change_report.md`, and `gate19_1_requirement_checklist.md`.

### Task 5: Formal Run and Verification

**Commands:**
- `conda run -n pytorch python -m py_compile experiments/scripts/run_gate19_1_calibrated_baseline_audit.py`
- `conda run -n pytorch python -m py_compile experiments/scripts/summarize_gate19_1.py`
- `conda run -n pytorch python -m pytest tests/test_gate19_1_calibrated_support_audit.py -q`
- `conda run -n pytorch python experiments/scripts/run_gate19_1_calibrated_baseline_audit.py --output-dir outputs/gate19_1 --gate19-input-dir outputs/gate19 --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --budgets 0.30 0.50 0.70 1.00 --primary-eval-mode compressed_projected --task-epochs 10 --max-paths 2 --include-typedhash true --nested-calibration true --write-per-class-confusion true`

- [x] If local OOM occurs, stop and return a server command instead of weakening the matrix.
- [x] Verify every required output and both zip packages exist.
- [x] Commit and push code changes to `main`.
