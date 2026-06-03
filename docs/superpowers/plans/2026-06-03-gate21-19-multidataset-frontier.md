# Gate21.19 Multidataset Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Gate21.19 as a real DBLP/ACM/IMDB stage-report frontier with dataset-aware planner backends, validation-selected representatives, and no full-fallback compression rows.

**Architecture:** Gate21.19 extends Gate21.18 with a common planner protocol and dataset-specific backends. The runner exports real HGB artifacts for ACM/IMDB frontiers, reuses verified DBLP official metrics where available, runs unmodified official SeHGNN through the existing queue, and writes decision/frontier/checklist outputs.

**Tech Stack:** Python, pytest, local conda `pytorch`, official SeHGNN HGB runner, existing HGB export/audit modules.

---

### Task 1: Gate21.19 Contract Tests

**Files:**
- Create: `tests/test_gate21_19_multidataset_frontier_contract.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:
- `DatasetPlannerBackend`, `Plan`, and `ExportResult` exist.
- ACM backend emits HeSF/Degree/Random/ValidationGreedy field plans for required ratios.
- IMDB backend emits MDfull mixture and ValidationGreedy channel plans.
- Gate21.19 decision requires all prompt flags and rejects fallback compression claims.
- Rep selection uses validation metrics and keeps test-oracle rows diagnostic only.

- [ ] **Step 2: Verify RED**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_gate21_19_multidataset_frontier_contract.py -q
```

Expected: fail because Gate21.19 modules do not exist yet.

### Task 2: Planner Backend Protocol

**Files:**
- Create: `hesf_coarsen/eval/official/gate21_19_planner_backends.py`
- Modify: `hesf_coarsen/eval/official/acm_closure_compression.py`
- Modify: `hesf_coarsen/eval/official/imdb_constraint_compression.py`

- [ ] **Step 1: Implement shared dataclasses/protocol**

Create:
- `Plan(dataset, method, planner_backend, planner_mode, requested_budget_type, requested_budget, params)`
- `ExportResult(plan, export_dir, manifest, audit)`
- `DatasetPlannerBackend.candidate_plans(...)`, `export_plan(...)`, `audit_constraints(...)`, `budget_metrics(...)`

- [ ] **Step 2: Implement DBLP backend**

Wrap existing Gate21.18 DBLP rows and Gate21.17 prior metrics:
- HeSF structural12/16 and metric-reuse placeholders for structural20/30 if exact prior rows exist.
- Random/Degree/Proportional support_edge20.
- Herding/HGCond/GCond/FreeHGC local rows.
- FreeHGC-score-as-selector structural16/20 as fair local selector probes.

- [ ] **Step 3: Implement ACM backend**

Use `export_acm_closure_compressed()` with planner modes:
- `coverage_greedy`
- `field_degree`
- `random`
- `validation_greedy`
- `cost_normalized_validation_delta`

Required methods:
- `ACM-HeSF-RCS-auto-field30/20/15/10`
- `ACM-Degree-field30/20/15/10`
- `ACM-Random-field30/20/15/10`
- `ACM-ValidationGreedy-field20/15/10`

- [ ] **Step 4: Implement IMDB backend**

Use `export_imdb_constraint_compressed()` with planner modes:
- `degree`
- `random`
- `validation_greedy`

Required methods:
- `IMDB-HeSF-RCS-auto structural30/20`
- `IMDB-Random-channel20`
- `IMDB-Degree-channel20`
- `IMDB-MDfull-MA50-MK20`
- `IMDB-MDfull-MA20-MK50`
- `IMDB-MDfull-MA50-MK50`
- `IMDB-MDfull-MA75-MK25`
- `IMDB-MDfull-MA25-MK75`
- `IMDB-MDfull-MA100-MK00`
- `IMDB-MDfull-MA00-MK100`
- `IMDB-ValidationGreedy-channel20/30/40/50`

- [ ] **Step 5: Verify GREEN**

Run the Gate21.19 contract test again and fix implementation gaps.

### Task 3: Decision and Summary Logic

**Files:**
- Create: `hesf_coarsen/eval/official/gate21_19_decision.py`
- Modify: `hesf_coarsen/eval/official/validation_metric_resolver.py` if Gate21.19 needs a dedicated selector helper.

- [ ] **Step 1: Implement required flags**

Emit:
- `FULL_NATIVE_READY_BY_DATASET`
- `EXPORT_FULL_FIDELITY_PASS_BY_DATASET`
- `BUDGET_METRIC_SEMANTICS_PASS`
- `NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE`
- `DBLP_FRONTIER_READY`
- `ACM_CLOSURE_FRONTIER_READY`
- `IMDB_CHANNEL_FRONTIER_READY`
- `ACM_VALIDATION_GREEDY_READY`
- `IMDB_VALIDATION_GREEDY_READY`
- `EXTERNAL_TP_LOCAL_BASELINES_READY`
- `HESF_RCS_REP_VALIDATED_READY`
- `HESF_RCS_REP_NO_TEST_LEAKAGE`
- `STAGE_REPORT_SMOKE_READY`
- `STAGE_REPORT_QUICK_ROBUSTNESS_READY`

- [ ] **Step 2: Enforce rules**

Reject:
- `constraint_safe_fallback=true` with `eligible_for_compression_claim=true`
- `training_executed=false` with `success=true`
- test-only selection with `eligible_for_decision=true`
- raw byte ratio being used as semantic structural ratio.

### Task 4: Runner and Summarizer

**Files:**
- Create: `experiments/scripts/run_gate21_19_multidataset_frontier.py`
- Create: `experiments/scripts/summarize_gate21_19_multidataset_frontier.py`

- [ ] **Step 1: Build runner**

Runner must:
- accept `--mode smoke|quick|preflight`, `--datasets`, `--device`, `--output`.
- create full/export anchors from prior outputs.
- create DBLP metric-reuse rows where exact prior official metrics exist.
- export ACM/IMDB real compressed rows.
- build and execute official SeHGNN training queue for rows without metrics.
- write every required Gate21.19 output.

- [ ] **Step 2: Build summarizer**

Summarizer must:
- refresh decision, budget audit, frontier tables, rep selection, and checklist from an existing output directory.
- not rerun training.

### Task 5: Smoke Execution

**Files:**
- Output: `outputs/gate21_19_smoke/*`

- [ ] **Step 1: Run smoke with local GPU**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m experiments.scripts.run_gate21_19_multidataset_frontier --mode smoke --datasets DBLP ACM IMDB --output outputs/gate21_19_smoke --device cuda
```

If OOM occurs, capture the exact failed command and produce server commands instead of claiming completion.

- [ ] **Step 2: Debug failures systematically**

For any failure:
- read stdout/stderr logs,
- reproduce the failing export/audit,
- fix root cause in exporter/planner/runner,
- rerun affected smoke.

### Task 6: Final Verification and GitHub Push

**Files:**
- Output: `outputs/gate21_19_smoke/gate21_19_requirement_checklist.md`

- [ ] **Step 1: Verify tests and compilation**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_gate21_19_multidataset_frontier_contract.py tests/test_gate21_18_budget_truth_real_compression_contract.py -q
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m py_compile experiments/scripts/run_gate21_19_multidataset_frontier.py experiments/scripts/summarize_gate21_19_multidataset_frontier.py hesf_coarsen/eval/official/gate21_19_planner_backends.py hesf_coarsen/eval/official/gate21_19_decision.py
git diff --check
```

- [ ] **Step 2: Verify attachment checklist**

Open `outputs/gate21_19_smoke/gate21_19_requirement_checklist.md` and confirm every prompt section P0-P7 and output/decision requirement is marked PASS or explicitly documented with a justified non-pass only if the user accepts it. The target is all PASS for smoke.

- [ ] **Step 3: Commit and push**

Stage only Gate21.19 code/tests/plan and small top-level Gate21.19 CSV/JSON/MD outputs. Do not stage `.dat` exports or full training logs.

Run:

```powershell
git add docs/superpowers/plans/2026-06-03-gate21-19-multidataset-frontier.md tests/test_gate21_19_multidataset_frontier_contract.py experiments/scripts/run_gate21_19_multidataset_frontier.py experiments/scripts/summarize_gate21_19_multidataset_frontier.py hesf_coarsen/eval/official/gate21_19_planner_backends.py hesf_coarsen/eval/official/gate21_19_decision.py
git add -f outputs/gate21_19_smoke/*.csv outputs/gate21_19_smoke/*.json outputs/gate21_19_smoke/*.md
git commit -m "Add Gate21.19 multidataset frontier"
git push origin main
```
