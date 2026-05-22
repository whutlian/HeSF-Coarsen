# Gate17 Real Validation/Occlusion Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for delegated implementation slices and superpowers:verification-before-completion before claiming completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run Gate17 real validation/occlusion-guided prototype support compression without reverting to preservation-first spectral coarsening.

**Primary metric:** compressed/projected target-node classification macro-F1 and accuracy under `primary_eval_mode="compressed_projected"` and `monitor="projected_val_macro_f1"`.

**Key constraint:** Fix summary/decision semantics before full sweeps. Do not claim single-row maxima as means, do not label proxy selectors as true validation or occlusion, and do not use test labels in selector or teacher training.

---

### Task 1: Branch, Audit, And Baseline State

**Files:**
- Create: `experiments/scripts/audit_gate17_code.py`
- Output: `outputs/gate17_code_audit/code_sync_report.md`
- Output: `outputs/gate17_code_audit/method_to_code_path.csv`
- Output: `outputs/gate17_code_audit/gate17_smoke_report.md`

- [ ] **Step 1: Create/switch branch**

Use branch `gate17-real-validation-occlusion-prototype`.

- [ ] **Step 2: Implement Gate17 audit script**

Record git state, required path existence, Gate17 method implementation status, `primary_eval_mode` default, and smoke-field readiness.

- [ ] **Step 3: Run baseline audit**

Run under local conda env `pytorch`:

```powershell
conda run -n pytorch python experiments/scripts/audit_gate17_code.py --output-dir outputs/gate17_code_audit
```

### Task 2: Correct Gate17 Summary And Decision Logic First

**Files:**
- Create: `experiments/scripts/summarize_gate17.py`
- Create: `tests/test_gate17_summary_aggregation.py`
- Create: `tests/test_gate17_exact_budget_gaps.py`

- [ ] **Step 1: Write summary tests first**

Verify method-level validation-selected means, separate best single run, exact-only gap filtering, and requested-ratio non-exact labeling.

- [ ] **Step 2: Implement summarizer**

Read Gate17 raw rows and write all required `outputs/gate17_tables/*` files, including `result.json`, `gate17_decision.md`, and `final_report.md`.

- [ ] **Step 3: Verify tests**

Run the Gate17 summary tests under conda `pytorch`.

### Task 3: True Validation And Real Occlusion Selection

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/selector.py`
- Create: `hesf_coarsen/task_first/selection/validation_selector.py`
- Create: `tests/test_gate17_true_validation_selector.py`
- Create: `tests/test_gate17_occlusion_importance.py`
- Create: `tests/test_gate17_no_test_leakage.py`

- [ ] **Step 1: Add explicit selector names**

Add `real_validation_block_greedy`, `real_occlusion_block_selector`, and `occlusion_plus_dblp_prototype`. Keep legacy `true_validation_block_greedy` out of primary Gate17 methods.

- [ ] **Step 2: Implement reusable block helpers**

Add `build_support_block_keys()` and `group_support_by_block()` with default and DBLP-aware block key modes.

- [ ] **Step 3: Implement lite honest validation greedy**

Run actual validation trial scoring through a callback/evaluator seam, record validation trial diagnostics, and never set `selector_uses_true_validation_feedback=True` without nonzero trials.

- [ ] **Step 4: Implement real occlusion block importance**

Mask/remove blocks through an evaluator seam, write per-block occlusion rows, and record trial/cache/objective diagnostics.

### Task 4: DBLP-Aware Prototype Condensation

**Files:**
- Modify: `hesf_coarsen/task_first/selection/config.py`
- Modify: `hesf_coarsen/task_first/selection/condensation.py`
- Create: `tests/test_gate17_dblp_prototype_condensation.py`

- [ ] **Step 1: Add `dblp_aware_prototype` strategy**

Use support type, relation channel, anchor bucket, class bucket, degree bucket, and bridge flag where available.

- [ ] **Step 2: Split large prototypes deterministically**

Honor `max_members_per_prototype`, split by degree/anchor/relation/hash as needed, and diagnose all split reasons.

- [ ] **Step 3: Preserve rare class/relation and high-degree bridge context**

Record forced raw bridge, rare class prototype, relation prototype, and budget conflict counts.

### Task 5: Gate17 Runner And Diagnostics

**Files:**
- Create: `experiments/scripts/run_gate17_support_selection.py`
- Create: `experiments/scripts/run_gate17_teacher_stability.py`
- Modify as needed: `hesf_coarsen/task_first/selection/pipeline.py`

- [ ] **Step 1: Runner smoke interface**

Support prompt command lines for smoke, DBLP pilot, and full Gate17 runs.

- [ ] **Step 2: Required row fields**

Every row must include projected/transfer metrics, recovery fields, leakage flags, budget fields, validation/occlusion/prototype diagnostics, and success/failure metadata.

- [ ] **Step 3: Teacher stability diagnostic**

Run full-graph teacher stability as auxiliary evidence only; do not promote teacher-only selectors to primary methods unless thresholds are met.

### Task 6: Local Experiments

- [ ] **Step 1: Smoke run**

Run ACM seed `12345`, ratio `0.30`, with full graph, H6, and one Gate17 method.

- [ ] **Step 2: DBLP pilot**

Run DBLP all 5 seeds at ratios `0.30`, `0.50`, `0.70` for baselines and Gate17 methods.

- [ ] **Step 3: Full Gate17 run**

Run ACM/DBLP/IMDB, 5 seeds, ratios `0.30`, `0.50`, `0.70` for all primary methods. If local OOM/GPU OOM occurs, stop and return a server command.

- [ ] **Step 4: Summarize and audit outputs**

Generate all required tables, diagnostics, optional figures if cheap, and final decision report.

### Task 7: Verification, Commit, Push

- [ ] **Step 1: Run focused tests and compile checks**

Run Gate17 tests plus relevant Gate16 regression tests under conda `pytorch`.

- [ ] **Step 2: Verify required outputs**

Check all required Gate17 table/diagnostic files exist and have nonzero size.

- [ ] **Step 3: Stage only Gate17-owned files**

Do not stage unrelated pre-existing untracked paths such as `build/`, `exports/`, `session.md`, or prior docs artifacts.

- [ ] **Step 4: Commit and push**

Commit Gate17 code changes and push branch `gate17-real-validation-occlusion-prototype` to GitHub.

---

**Spec coverage check:** This plan covers branch setup, Gate17 audit, corrected summary/decision logic, exact-only gaps, true validation trials, real occlusion trials, DBLP-aware prototype condensation, teacher reliability, runner/diagnostics, smoke/DBLP/full local experiments, verification, and GitHub push.
