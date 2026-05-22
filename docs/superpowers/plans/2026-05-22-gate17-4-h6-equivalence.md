# Gate17.4 H6 Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Gate17.4 as a narrow H6 construction-equivalence and best-eligible real-feedback diagnostic gate.

**Architecture:** Reuse Gate17.3 runner patterns for seeds, exact budgets, existing real-feedback methods, and outputs. Add a Gate17.4 runner that exports H6 assignment/coarse artifacts, evaluates H6 construction equivalence through the same HETTREE path, emits H6 selected-set and cluster controls separately, and delegates decision logic to a Gate17.4 summarizer.

**Tech Stack:** Python, NumPy, pytest, local conda env `pytorch`, existing `HeteroGraph`, `coarsen_graph`, `evaluate_hettree_task`, Gate17.1 semantic-tree helpers, Gate17.3 budget helpers.

---

### Task 1: Gate17.4 Code Audit

**Files:**
- Create: `outputs/gate17_4_code_audit/code_sync_report.md`
- Create: `outputs/gate17_4_code_audit/method_to_code_path.csv`
- Create: `outputs/gate17_4_code_audit/gate17_4_config_overrides.md`

- [ ] Verify `main` contains Gate17.2/Gate17.3 runner/summarizer files using `git ls-files`.
- [ ] Write the method-to-code-path table with `method_name,selector_path,construction_path,budget_policy,feedback_source,eligible_for_main_decision,notes`.
- [ ] Document explicit unsafe-config overrides used by Gate17.3 and planned for Gate17.4.
- [ ] Confirm H6 construction is reusable via `experiments/scripts/gate13_task_first_common.py::run_support_baseline`.

### Task 2: Red Tests

**Files:**
- Create: `tests/test_gate17_4_h6_equivalence.py`
- Create: `tests/test_gate17_4_summary.py`

- [ ] Add a failing test that `parse_dataset_seeds(["ACM:23456", "DBLP:23456", "IMDB:45678"])` returns exact pairs and not a Cartesian product.
- [ ] Add a failing test that H6 construction equivalence rows pass when assignment/tree/edge/feature deltas are zero.
- [ ] Add a failing test that selected-set control is reported separately from construction equivalence.
- [ ] Add a failing summary fixture where sensitivity has the worst DBLP gap but real-validation-neutral-fill is the best eligible method; assert Gate17.4 reports the latter.
- [ ] Add a failing summary fixture asserting full-residual upperbound and baselines are excluded from `best_eligible_method`.

### Task 3: Gate17.4 H6 Helpers

**Files:**
- Create: `experiments/scripts/gate17_4_h6.py`

- [ ] Implement exact dataset-seed parsing for `DATASET:SEED` tokens.
- [ ] Implement H6 selected support representative export from assignment clusters.
- [ ] Implement H6 artifact export: assignment, cluster members, coarse node map, coarse graph edges, relation edge mass, feature mean by type, semantic hash/checksum, and limitations markdown.
- [ ] Implement edge-mass and feature-mean delta helpers.
- [ ] Implement coarse-graph hash and assignment equivalence helpers.
- [ ] Implement H6 selected-set graph construction by reusing existing selected-raw graph builder.
- [ ] Implement H6 construction-control graph evaluation by using H6 coarse graph and assignment directly.
- [ ] Implement H6 cluster-control rows using H6 coarse cluster units without converting clusters back into raw selected nodes.

### Task 4: Gate17.4 Summarizer

**Files:**
- Create: `experiments/scripts/summarize_gate17_4.py`
- Test: `tests/test_gate17_4_summary.py`

- [ ] Read `gate17_4_raw_rows.csv`.
- [ ] Compute eligible rows only when success, Gate17.4 main method, exact node budget, represented-context bounded, no full residual, no test leakage, and `primary_eval_mode=compressed_projected`.
- [ ] Compute best eligible method before DBLP gap reporting.
- [ ] Write `gate17_4_validation_selected_by_method.csv`, `gate17_4_by_dataset_selected.csv`, `gate17_4_exact_budget_paired_gaps.csv`, `result.json`, `gate17_4_decision.md`, and `final_report.md`.
- [ ] Include dataset-role flags for ACM/DBLP/IMDB and never use ACM saturation as success evidence.
- [ ] Use specific Gate17.4 decision reason names from the prompt.

### Task 5: Gate17.4 Runner

**Files:**
- Create: `experiments/scripts/run_gate17_4_h6_equivalence.py`
- Test: `tests/test_gate17_4_h6_equivalence.py`

- [ ] Implement CLI with `--dataset-seeds`, `--support-ratios`, `--task-epochs`, `--max-paths`, `--feature-mode`, `--primary-eval-mode`, and `--out-dir`.
- [ ] Validate `primary_eval_mode == compressed_projected`.
- [ ] Include required baselines and methods.
- [ ] Map real-validation-neutral-fill and real-occlusion-neutral-fill to existing Gate17.3 configs with explicit unsafe override fields.
- [ ] Emit H6 construction-equivalence and selected-set-control rows plus `diagnostics/gate17_4_h6_equivalence.csv`.
- [ ] Emit all required Gate17.4 diagnostics and artifact directories.
- [ ] Call `summarize_gate17_4.summarize()` at the end.

### Task 6: Verification and Experiment

**Files:**
- Output: `outputs/gate17_4_h6_equivalence/`

- [ ] Run red tests before implementation and confirm missing-module failures.
- [ ] Run focused Gate17.4 tests after implementation.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m py_compile` on new and modified scripts.
- [ ] Run `git diff --check`.
- [ ] Run the formal local experiment with conda env `pytorch` and prompt parameters.
- [ ] Parse output files, confirm all required files exist and `gate17_4_raw_rows.csv` has expected rows.
- [ ] Re-read the prompt checklist and create a final requirement-completion table.
- [ ] Commit and push code changes to GitHub `main`.

---

## Self-Review

- Spec coverage: tasks cover code audit, best-eligible summarizer, H6 construction-equivalence, H6 artifact export, cluster controls, runner matrix, output files, diagnostics, decision rules, and final checklist.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: plan uses `Gate17.4`, `H6`, `eligible_for_main_decision`, `primary_eval_mode`, and exact output names consistently.
