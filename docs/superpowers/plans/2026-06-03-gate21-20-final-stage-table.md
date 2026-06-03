# Gate21.20 Final Stage Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Gate21.20 final-style official SeHGNN stage-report outputs with corrected HeSF representative selection, ACM selector-overlap diagnostics, upgraded IMDB HeSF channel rows, FreeHGC-score-as-selector proxy rows, and critical-row quick robustness evidence.

**Architecture:** Reuse Gate21.19 official outputs as the canonical smoke input, then add focused Gate21.20 modules for selection, diagnostics, planner upgrades, and final table assembly. Training stays in the existing official SeHGNN queue/runner path so main-table rows remain unmodified-official-compatible.

**Tech Stack:** Python, pytest, local conda `pytorch`, existing HGB exporters, official SeHGNN `external/SeHGNN/hgb/main.py`, CSV/JSON report artifacts.

---

## Files

- Create `tests/test_gate21_20_final_stage_table_contract.py`: contract tests for representative pools, decision flags, frontier semantics, and required output fields.
- Create `hesf_coarsen/eval/official/rep_selection.py`: Gate21.20 representative selection and validation metric resolver.
- Create `hesf_coarsen/eval/official/acm_selector_overlap.py`: ACM selected keyword/PK-edge overlap diagnostics.
- Create `hesf_coarsen/eval/official/imdb_planner_upgrade.py`: upgraded IMDB HeSF channel planner rows and export helpers.
- Create `hesf_coarsen/eval/official/freehgc_score_selector.py`: DBLP FreeHGC-score-as-selector local proxy export and row builder.
- Create `hesf_coarsen/eval/official/final_stage_report_tables.py`: best-method comparison and Pareto frontier tables.
- Create `hesf_coarsen/eval/official/critical_robustness_runner.py`: robustness row aggregation and deterministic-proof bookkeeping.
- Create `hesf_coarsen/eval/official/gate21_20_decision.py`: Gate21.20 decision flags and rule checks.
- Create `experiments/scripts/run_gate21_20_final_stage_table.py`: runner for smoke and quick-robust modes.
- Create `experiments/scripts/summarize_gate21_20_final_stage_table.py`: summarizer from a Gate21.20 output directory.
- Modify existing exporter code only if a root-cause failure requires it.

## Tasks

### Task 1: Contract Tests

- [ ] **Step 1: Write failing tests**

Create `tests/test_gate21_20_final_stage_table_contract.py` with tests that assert:

```python
def test_hesf_rep_validated_uses_only_hesf_family():
    from hesf_coarsen.eval.official.rep_selection import select_gate21_20_representatives
    rows = [
        ready("DBLP", "GCond-score-TP-local", "external_tp_baseline", 0.90, 0.90),
        ready("DBLP", "HeSF-RCS-auto structural16", "schema_preserving_rcs", 0.80, 0.79),
    ]
    reps = select_gate21_20_representatives(rows, datasets=["DBLP"])
    hesf = next(row for row in reps if row["rep_type"] == "HeSF-RCS-Rep-Validated")
    assert hesf["selected_method"] == "HeSF-RCS-auto structural16"
    assert hesf["selected_method_family"].startswith("schema_preserving")
    assert hesf["uses_test_for_selection"] is False
    assert hesf["eligible_for_main_decision"] is True
```

Add companion tests for:

- missing HeSF validation metrics emits `missing_real_validation_metric`;
- `Best-Compressed-Validated` may select a non-HeSF row;
- `TestOracle-Best` uses test metrics and is decision-ineligible;
- ACM overlap output columns exist and Jaccard is in `[0, 1]`;
- IMDB upgraded planner emits `IMDB-HeSF-RCS-channel40/50` with MD full and reciprocal-safe fields;
- FreeHGC selector emits structural16/20 rows;
- decision requires `FREEHGC_SCORE_AS_SELECTOR_READY`, `ACM_SELECTOR_OVERLAP_READY`, `IMDB_HEFS_UPGRADED_PLANNER_READY`, and `STAGE_REPORT_QUICK_ROBUSTNESS_READY`.

- [ ] **Step 2: Run RED**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_gate21_20_final_stage_table_contract.py -q
```

Expected: import/function failures because Gate21.20 modules do not exist yet.

### Task 2: Gate21.20 Representative Selection

- [ ] **Step 1: Implement `rep_selection.py`**

Add:

```python
def select_gate21_20_representatives(rows, *, datasets=("DBLP", "ACM", "IMDB")) -> list[dict]:
    ...
```

Rules:

- HeSF pool: method contains `HeSF-RCS` or method_family in `{"schema_preserving_rcs", "hesf_rcs"}` and excludes `Rep`, `TestOracle`, external baselines, random/degree/proportional, and full anchors.
- Selection metric: validation micro, then validation macro, then lower semantic cost.
- Missing real validation metrics produce a diagnostic row with `selection_reason=missing_real_validation_metric`.
- Best compressed pool: all eligible compressed rows with validation metrics.
- Test oracle pool: all eligible compressed rows with test metrics and `eligible_for_main_decision=False`.

- [ ] **Step 2: Run GREEN tests for representative selection**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_gate21_20_final_stage_table_contract.py -q
```

Expected: representative tests pass; remaining module tests still fail until later tasks.

### Task 3: ACM Selector Overlap

- [ ] **Step 1: Implement `acm_selector_overlap.py`**

Read exported ACM HGB directories from Gate21.19, derive selected keyword ids from `PK` relation and selected PK edge pairs, and compute pairwise Jaccard for HeSF vs Degree, HeSF vs ValidationGreedy, Degree vs ValidationGreedy at field10/15/20/30.

Output rows include:

- `selected_keyword_jaccard_hesf_vs_degree`
- `selected_keyword_jaccard_hesf_vs_validation_greedy`
- `selected_keyword_jaccard_degree_vs_validation_greedy`
- `selected_PK_edge_jaccard_hesf_vs_degree`
- `selected_PK_edge_jaccard_hesf_vs_validation_greedy`
- `field_degree_distribution_mean`
- `field_degree_distribution_std`
- `validation_gain_by_field_bucket`

- [ ] **Step 2: Verify overlap tests**

Run the contract test. Expected: ACM overlap tests pass.

### Task 4: IMDB Upgraded HeSF Planner

- [ ] **Step 1: Implement `imdb_planner_upgrade.py`**

Use Gate21.19 IMDB validation rows to select channel40/channel50 allocations:

- Prefer candidates with full MD/DM and reciprocal MA/AM, MK/KM.
- Score = `validation_micro_f1 - lambda * semantic_structural_storage_ratio`, with lambda default `0.02`.
- For channel40/50, map to existing strongest validation-greedy or MDfull allocation and export under method names `IMDB-HeSF-RCS-channel40` and `IMDB-HeSF-RCS-channel50`.

Rows must include the required `gate21_20_imdb_planner_upgrade.csv` columns.

- [ ] **Step 2: Train upgraded rows**

Run official SeHGNN on local GPU for both upgraded rows in smoke mode if prior exact metrics are unavailable.

### Task 5: FreeHGC-Score-As-Selector Proxy

- [ ] **Step 1: Implement `freehgc_score_selector.py`**

For DBLP structural16/20:

- Keep all target nodes.
- Score support nodes by degree/relation-channel reachability and feature-presence proxy.
- Select support nodes/edges into an HGB export that preserves target incidence and official loader compatibility.
- Use existing DBLP target-preserving export helpers where possible.
- Emit `FreeHGC-score-as-selector structural16` and `FreeHGC-score-as-selector structural20`.

- [ ] **Step 2: Train selector rows**

Run official SeHGNN on local GPU for both rows. Output `gate21_20_freehgc_score_selector.csv`.

### Task 6: Robustness Aggregation

- [ ] **Step 1: Implement `critical_robustness_runner.py`**

For critical rows, aggregate per-run metrics into `gate21_20_robustness_by_method.csv`.

Deterministic proof is allowed only when:

- selected edge hash is identical for graph seeds or graph seed is documented as ignored;
- training seed count is at least 3.

Rows requiring stochastic graph seeds must have 3 graph seeds or a failure row that prevents `STAGE_REPORT_QUICK_ROBUSTNESS_READY`.

- [ ] **Step 2: Run quick-robust critical training**

Use:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' experiments/scripts/run_gate21_20_final_stage_table.py --datasets DBLP ACM IMDB --mode quick-robust --reuse-gate21-19 --graph-seeds 1 2 3 --training-seeds 1 2 3 --run-critical-rows --out-dir results/gate21_20
```

If OOM occurs, stop local GPU work and return the equivalent server command.

### Task 7: Final Tables and Decision

- [ ] **Step 1: Implement `final_stage_report_tables.py`**

Create:

- `gate21_20_best_method_comparison.csv`
- `gate21_20_dblp_frontier.csv`
- `gate21_20_acm_frontier.csv`
- `gate21_20_imdb_frontier.csv`
- `gate21_20_dataset_frontier_by_method.csv`

Pareto dominance rule: another method dominates if it has lower/equal semantic ratio, higher/equal micro, higher/equal macro, with one strict improvement.

- [ ] **Step 2: Implement `gate21_20_decision.py`**

Emit required flags and hard checks:

- no fallback in main compression table;
- HeSF rep pool is HeSF-only;
- no test leakage;
- FreeHGC selector ready;
- ACM overlap ready;
- IMDB upgraded planner ready;
- smoke-ready, quick-robust-ready, final-table-ready separated.

### Task 8: Runner and Summarizer

- [ ] **Step 1: Implement `run_gate21_20_final_stage_table.py`**

Support both prompt commands:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' experiments/scripts/run_gate21_20_final_stage_table.py --datasets DBLP ACM IMDB --mode smoke --reuse-gate21-19 --out-dir results/gate21_20
```

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' experiments/scripts/run_gate21_20_final_stage_table.py --datasets DBLP ACM IMDB --mode quick-robust --graph-seeds 1 2 3 --training-seeds 1 2 3 --run-critical-rows --out-dir results/gate21_20
```

- [ ] **Step 2: Implement summarizer**

Support:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' experiments/scripts/summarize_gate21_20_final_stage_table.py --in-dir results/gate21_20 --out-dir results/gate21_20_summary
```

### Task 9: Final Verification and Git

- [ ] **Step 1: Run tests and compile checks**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_gate21_20_final_stage_table_contract.py -q
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m py_compile experiments/scripts/run_gate21_20_final_stage_table.py experiments/scripts/summarize_gate21_20_final_stage_table.py hesf_coarsen/eval/official/gate21_20_decision.py hesf_coarsen/eval/official/rep_selection.py hesf_coarsen/eval/official/critical_robustness_runner.py hesf_coarsen/eval/official/acm_selector_overlap.py hesf_coarsen/eval/official/imdb_planner_upgrade.py hesf_coarsen/eval/official/freehgc_score_selector.py hesf_coarsen/eval/official/final_stage_report_tables.py
```

- [ ] **Step 2: Verify prompt checklist**

Read `results/gate21_20/gate21_20_requirement_checklist.md`; ensure every prompt item is PASS or explicitly listed as a blocker with reason.

- [ ] **Step 3: Commit and push**

Stage only Gate21.20 code/test/plan files. Commit and push to `origin/main`.

---

## Spec Coverage Self-Review

- P0/P1 rep definition and validation resolver: Task 2.
- P2 IMDB upgraded planner: Task 4.
- P3 ACM overlap: Task 3.
- P4 3x3/quick robustness: Task 6.
- P5 best-method table: Task 7.
- P6 FreeHGC-score-as-selector: Task 5.
- P7 frontier tables: Task 7.
- P8 no fallback assertions: Task 7.
- P9 decision flags: Task 7.
- P10 run modes: Task 8.
- P15 acceptance criteria: Task 9 checklist.
