# Gate21 OpenSOTA Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Gate21 bridge that exports HeSF-Coarsen heterogeneous graphs for SeHGNN/OpenHGNN, records dependency/adapter failures honestly, and applies validation-only HeSF-CAL calibration to saved logits.

**Architecture:** Add a focused `hesf_coarsen.eval.official` package for graph export, audit, metrics, calibration, and external evaluator subprocess wrappers. The Gate21 runner owns dataset/method matrix orchestration and writes raw rows, diagnostics, configs, logs, and final reports; the summarizer turns those artifacts into the required decision files.

**Tech Stack:** Python 3, NumPy, optional PyTorch/DGL, pytest, subprocess-based external GitHub repo invocation.

---

### Task 1: Plan, Preflight, and Test Skeleton

**Files:**
- Create: `docs/superpowers/plans/2026-05-23-gate21-open-sota-bridge.md`
- Create: `tests/eval_official/test_graph_export.py`
- Create: `tests/eval_official/test_calibration_adapter.py`
- Create: `tests/eval_official/test_no_test_leakage.py`
- Create: `tests/eval_official/test_gate21_smoke.py`

- [ ] **Step 1: Write export tests**

Create a tiny two-type graph with two relations and assert `export_hgb_graph(...)` writes node features, edge arrays, split arrays, labels, target mapping CSV, `metadata.json`, and `export_audit.json`. Assertions must check target mapping bijection, disjoint splits, label alignment, type names, relation names, and edge counts.

- [ ] **Step 2: Write calibration tests**

Call `calibrate_logits_nested(val_logits, val_labels, test_logits, split_seeds=(11, 22))` and assert finite ECE/NLL/Brier, deterministic nested stats, same test-logit shape, `calibration_uses_test_labels is False`, and selected parameters are based on validation-only candidates.

- [ ] **Step 3: Write no-leakage tests**

Inspect the calibration function signature to ensure it does not accept `test_labels`. Export a graph and verify `splits/train_labels.npy` and `splits/val_labels.npy` exist, while `splits/test_labels_for_training.npy` and calibration test-label artifacts do not. Verify runner dry-run rows set `calibration_uses_test_labels=false`.

- [ ] **Step 4: Write smoke test**

Use the tiny graph to run `export_hgb_graph`, `calibrate_logits_nested`, and `summarize_rows` on synthetic failed-dependency and success-like rows without requiring SeHGNN/OpenHGNN.

- [ ] **Step 5: Run tests and confirm red**

Run: `conda run -n pytorch python -m pytest tests/eval_official -q`

Expected before implementation: import failures for `hesf_coarsen.eval.official` and missing Gate21 scripts.

### Task 2: Official Eval Package

**Files:**
- Create: `hesf_coarsen/eval/official/__init__.py`
- Create: `hesf_coarsen/eval/official/metrics.py`
- Create: `hesf_coarsen/eval/official/calibration_adapter.py`
- Create: `hesf_coarsen/eval/official/graph_export.py`
- Create: `hesf_coarsen/eval/official/hgb_export.py`
- Create: `hesf_coarsen/eval/official/audit.py`
- Create: `hesf_coarsen/eval/official/runner_utils.py`

- [ ] **Step 1: Implement metrics**

Add `classification_metrics_from_logits`, `calibration_quality`, `per_class_rows`, and `confusion_rows`. Use stable softmax, truth/pred union macro-F1, accuracy, micro-F1, ECE, NLL, and Brier.

- [ ] **Step 2: Implement validation-only calibration**

Add `calibrate_logits_nested` with temperature grid, class-bias grid, nested validation split seeds, macro guard, candidate logging, deterministic tie breaks, and `calibration_uses_test_labels=false`. Do not accept or read test labels.

- [ ] **Step 3: Implement graph export**

Add `export_hgb_graph` and `write_hgb_metadata_files`. Export NumPy features by type, edge arrays by relation, split arrays, train/val label artifacts only, full `labels.npy` for evaluation, target ids, mapping CSV, metadata, and audit JSON. Raise on non-bijective mapping, overlapping splits, label mismatch, or test-label leakage.

- [ ] **Step 4: Implement audit and runner utilities**

Add `audit_export_record`, `write_dependency_report`, `repo_commit_hash`, `clone_external_repo`, and `write_json`. HETTREE must always be represented as `excluded_code_unavailable` when mentioned.

- [ ] **Step 5: Run tests and confirm green**

Run: `conda run -n pytorch python -m pytest tests/eval_official/test_graph_export.py tests/eval_official/test_calibration_adapter.py tests/eval_official/test_no_test_leakage.py -q`

Expected after implementation: all tests pass.

### Task 3: External Evaluator Bridges

**Files:**
- Create: `hesf_coarsen/eval/official/sehgnn_bridge.py`
- Create: `hesf_coarsen/eval/official/openhgnn_bridge.py`

- [ ] **Step 1: Implement SeHGNN official wrapper**

Add `run_sehgnn_official(export_dir, repo_dir, dataset_name, target_type, seed, config, output_dir)`. If repo is missing, return `status=failed_dependency` with `error_message=missing_repo`. If no runnable known entrypoint/logit adapter is available, return `status=failed_format_adapter`. Always write stdout/stderr paths and config JSON; never use `hettree_lite`.

- [ ] **Step 2: Implement OpenHGNN wrapper**

Add `run_openhgnn_model(...)` with the same schema for `OpenHGNN-SeHGNN`, `OpenHGNN-HGT`, and `OpenHGNN-SimpleHGN`. Missing repo returns `failed_dependency`; unsupported export adapter returns `failed_format_adapter`; missing logits returns `failed_missing_logits`.

- [ ] **Step 3: Verify wrapper schema**

Run smoke tests that call wrappers with missing repo paths and assert unified row fields, clear failures, stdout/stderr/config files, and no fabricated metrics.

### Task 4: Gate21 Runner and Summarizer

**Files:**
- Create: `experiments/scripts/run_gate21_open_sota_bridge.py`
- Create: `experiments/scripts/summarize_gate21_open_sota.py`

- [ ] **Step 1: Implement CLI parsing**

Support `--datasets`, `--dataset-seeds`, `--methods`, `--support-ratios`, `--models`, `--sehgnn-repo`, `--openhgnn-repo`, `--output-dir`, `--calibrate`, `--primary-eval-mode`, `--export-only`, `--dry-run`, `--skip-models`, `--max-datasets`, `--max-runs`, `--strict`, and `--clone-external`.

- [ ] **Step 2: Implement graph/method construction**

Map `full` to original graph, `target-only` to target-only sanity graph, `H6` to `H6-no-spec-support-only`, `flatten` to `flatten-sum-support-only`, and `typedhash` to `TypedHash-ChebHeat-support-only`. Use existing bounded support-baseline builders and preserve `support_ratio=0.30`.

- [ ] **Step 3: Implement output writing**

Write `gate21_raw_rows.csv`, export audit CSV, calibration CSV, per-class CSV, confusion CSV, dependency report JSON, failure report CSV, configs, logs, `code_audit.md`, and `final_report.md`.

- [ ] **Step 4: Implement summarizer**

Read raw rows and diagnostics; write `gate21_by_method.csv`, `gate21_by_dataset_model.csv`, `gate21_he_sf_cal_vs_full.csv`, `gate21_calibration_effect.csv`, `gate21_result.json`, and `gate21_decision.md`. Use only exact decision values: `GATE21_BRIDGE_PASS_HESF_CAL_TRANSFERS`, `GATE21_BRIDGE_PASS_HESF_CAL_DOES_NOT_TRANSFER`, `FIX_OFFICIAL_BRIDGE`, `FIX_COMPRESSED_GRAPH_EXPORT`, `RESET_095_TARGET_TO_RECOVERY`.

- [ ] **Step 5: Run Gate21 smoke**

Run: `conda run -n pytorch python -m pytest tests/eval_official/test_gate21_smoke.py -q`

Expected: smoke passes without external repos.

### Task 5: Required Local Runs and Final Verification

**Files:**
- Outputs under: `outputs/gate21_open_sota/`

- [ ] **Step 1: Preflight**

Record HeSF-Coarsen commit, Gate20 archive presence, GitHub repo remote availability, local external path status, Python/Torch/DGL/CUDA status, and dataset-load status in `code_audit.md` and `diagnostics/gate21_dependency_report.json`.

- [ ] **Step 2: Export-only run**

Run DBLP seed `23456` for `full` and `H6` ratio `0.30` with `--export-only --strict false`. Pass criteria: export audit rows report bijective mapping, disjoint splits, preserved split counts, label alignment, and `no_test_label_export_leakage=true`.

- [ ] **Step 3: Minimal bridge attempt**

Run the initial matrix with DBLP seeds `23456,56789`, ACM seed `23456`, IMDB seed `45678`, methods `full,target-only,H6,flatten,typedhash`, ratio `0.30`, models `sehgnn_official,openhgnn_sehgnn,openhgnn_hgt,openhgnn_simplehgn`, `--calibrate`, `--strict false`. If external repos are missing or adapters unavailable, record failed statuses and do not fabricate official scores.

- [ ] **Step 4: Summarize**

Run: `conda run -n pytorch python -m experiments.scripts.summarize_gate21_open_sota --input-dir outputs/gate21_open_sota`

Expected in missing-dependency state: `decision=FIX_OFFICIAL_BRIDGE`, `official_bridge_pass=false`, `no_test_leakage=true`, `export_audit_pass=true`.

- [ ] **Step 5: Final checks**

Run: `conda run -n pytorch python -m pytest tests/eval_official tests/test_gate20_cal_multiseed.py -q`; `conda run -n pytorch python -m py_compile` on new scripts/package; `git diff --check`; inspect `gate21_requirement_checklist.md`; commit and push only Gate21 code/test/plan changes.
