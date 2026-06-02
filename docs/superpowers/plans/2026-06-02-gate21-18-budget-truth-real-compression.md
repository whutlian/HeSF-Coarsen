# Gate21.18 Budget Truth Real Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible Gate21.18 smoke run with truthful budget semantics, no full-fallback rows in the main compression table, and at least one real compressed ACM and IMDB official SeHGNN row.

**Architecture:** Add Gate21.18-specific row schema, budget audit, compression exporters, decision logic, runner, and summarizer while reusing the existing official SeHGNN training queue. ACM and IMDB exporters write official HGB files that satisfy the exact upstream loader constraints instead of falling back to full graph copies.

**Tech Stack:** Python, pytest, local conda environment `pytorch`, official SeHGNN HGB runner under `external/SeHGNN`, HGB text exports under `data/*/raw`.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_gate21_18_budget_truth_real_compression_contract.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
from hesf_coarsen.eval.official.budget_truth_audit import annotate_budget_truth
from hesf_coarsen.eval.official.gate21_18_decision import gate21_18_decision
from hesf_coarsen.eval.official.acm_closure_compression import audit_acm_closure_export
from hesf_coarsen.eval.official.imdb_constraint_compression import audit_imdb_constraint_export

def test_structural_budget_uses_semantic_ratio_not_raw_bytes():
    row = annotate_budget_truth({
        "requested_budget_type": "structural_storage_ratio",
        "requested_budget": 0.2,
        "semantic_structural_storage_ratio": 0.97,
        "raw_hgb_text_byte_ratio": 0.2,
    })
    assert row["budget_match_for_requested_metric"] is False
    assert row["budget_metric_used_for_match"] == "semantic_structural_storage_ratio"
    assert row["budget_match_failure_type"] == "budget_mismatch"

def test_full_fallback_rows_are_excluded_from_gate21_18_main_table():
    decision = gate21_18_decision(main_rows=[{
        "dataset": "ACM",
        "method": "Random-edge-relwise",
        "constraint_safe_fallback": True,
        "eligible_for_main_table": False,
        "eligible_for_compression_claim": False,
        "selected_edge_hash": "full",
        "success": True,
        "training_executed": True,
    }], fallback_rows=[{
        "dataset": "ACM",
        "constraint_safe_fallback": True,
        "selected_edge_hash": "full",
    }])
    assert decision["NO_FULL_FALLBACK_IN_MAIN_COMPRESSION_TABLE"] is True
    assert decision["FULL_HASH_ROWS_ONLY_IN_SANITY_TABLE"] is True

def test_acm_and_imdb_auditors_detect_nonfallback_compressed_exports(tmp_path):
    acm = audit_acm_closure_export(tmp_path / "ACM")
    imdb = audit_imdb_constraint_export(tmp_path / "IMDB")
    assert set(acm) >= {"constraint_safe_fallback", "P_matches_PK", "PK_KP_reciprocal"}
    assert set(imdb) >= {"constraint_safe_fallback", "MD_DM_reciprocal", "movie_single_director_constraint_pass"}
```

- [ ] **Step 2: Run RED**

Run:

```powershell
conda run -n pytorch python -m pytest tests/test_gate21_18_budget_truth_real_compression_contract.py -q
```

Expected: fail because Gate21.18 modules do not exist yet.

### Task 2: Budget Truth Schema

**Files:**
- Create: `hesf_coarsen/eval/official/budget_truth_audit.py`
- Create: `hesf_coarsen/eval/official/gate21_18_decision.py`

- [ ] **Step 1: Implement explicit budget fields**

Add `annotate_budget_truth(row, tolerance=0.03)` that fills:

```text
actual_edge_ratio
actual_support_edge_ratio
actual_support_node_ratio
semantic_structural_storage_ratio
raw_hgb_text_byte_ratio
static_inference_package_ratio
reconstructable_package_ratio
budget_match_for_requested_metric
budget_match_failure_type
budget_match_failure_reason
```

For `requested_budget_type=structural_storage_ratio`, compare only `semantic_structural_storage_ratio`.

- [ ] **Step 2: Implement fallback exclusion decision flags**

Add `gate21_18_decision()` with the required Gate21.18 flags, including no full fallback in main table and no mixed structural ratio.

- [ ] **Step 3: Run GREEN for budget/decision tests**

Run the same pytest command and fix only failures in these modules.

### Task 3: ACM Closure-Preserving Export

**Files:**
- Create: `hesf_coarsen/eval/official/acm_closure_compression.py`

- [ ] **Step 1: Implement ACM export writer**

Implement `export_acm_closure_compressed(source_dir, export_dir, method, keyword_ratio, graph_seed)`:

```text
keep all P/A/C nodes
select K nodes by coverage_greedy, degree, or random
write PK/KP for selected K only
rewrite paper P feature vector to selected K columns
rewrite author A and conference C features as AP*PK and CP*PK
preserve PP/PP_r, AP/PA, CP/PC reciprocals
write label.dat, label.dat.test, info.dat
```

- [ ] **Step 2: Implement ACM audit**

`audit_acm_closure_export(export_dir)` checks `P_matches_PK`, `A_matches_AP_PK`, `C_matches_CP_PK`, reciprocal pairs, non-full ratios, and `constraint_safe_fallback=False`.

- [ ] **Step 3: Verify official loader preflight**

Run export in dry-run smoke and import `external/SeHGNN/data/data_loader.py` on the produced ACM directory.

### Task 4: IMDB Constraint-Preserving Export

**Files:**
- Create: `hesf_coarsen/eval/official/imdb_constraint_compression.py`

- [ ] **Step 1: Implement IMDB export writer**

Implement `export_imdb_constraint_compressed(source_dir, export_dir, method, actor_ratio, keyword_ratio, graph_seed)`:

```text
keep all M/D/A/K nodes
keep MD/DM fully so each movie has exactly one director
select MA/AM and MK/KM channel edges by degree or random
preserve reciprocal pairs exactly
write labels and info
```

- [ ] **Step 2: Implement IMDB audit**

`audit_imdb_constraint_export(export_dir)` checks MD/DM, MA/AM, MK/KM reciprocity, one-director-per-movie, non-full edge/semantic ratios, and `constraint_safe_fallback=False`.

### Task 5: Gate21.18 Runner And Summarizer

**Files:**
- Create: `experiments/scripts/run_gate21_18_budget_truth_real_compression.py`
- Create: `experiments/scripts/summarize_gate21_18_budget_truth_real_compression.py`
- Modify: `hesf_coarsen/eval/official/validation_metric_resolver.py`
- Modify: `hesf_coarsen/eval/official/freehgc_score_tp_local.py`

- [ ] **Step 1: Generate exact smoke rows**

Emit the attachment's exact smoke rows for DBLP, ACM, and IMDB, with explicit requested budget types and budget truth fields.

- [ ] **Step 2: Reuse official training queue**

Queue rows only when they are exported, schema compatible, target preserving, unmodified official SeHGNN, and pending official training.

- [ ] **Step 3: Emit all required files**

Write every P10 file, including `gate21_18_main_official_table.csv`, `gate21_18_budget_truth_audit.csv`, `gate21_18_fallback_loader_sanity.csv`, ACM/IMDB audits, decision files, and summary.

- [ ] **Step 4: Resolve HeSF-RCS representative selection**

Use actual validation metrics from training runs. If missing, mark validation missing instead of selecting a proxy row for the main table.

### Task 6: Execute Smoke And Verify

**Files:**
- Output: `outputs/gate21_18_smoke/*`

- [ ] **Step 1: Run contract tests**

```powershell
conda run -n pytorch python -m pytest tests/test_gate21_18_budget_truth_real_compression_contract.py tests/test_gate21_17_executed_stage_report_contract.py -q
```

- [ ] **Step 2: Run Gate21.18 smoke**

```powershell
conda run -n pytorch python -m experiments.scripts.run_gate21_18_budget_truth_real_compression --mode smoke --datasets DBLP ACM IMDB --output outputs/gate21_18_smoke --device cuda
```

- [ ] **Step 3: Summarize**

```powershell
conda run -n pytorch python -m experiments.scripts.summarize_gate21_18_budget_truth_real_compression --input-dir outputs/gate21_18_smoke --output-dir outputs/gate21_18_smoke --mode smoke --datasets DBLP ACM IMDB
```

- [ ] **Step 4: Compile and final verification**

```powershell
conda run -n pytorch python -m py_compile experiments/scripts/run_gate21_18_budget_truth_real_compression.py experiments/scripts/summarize_gate21_18_budget_truth_real_compression.py hesf_coarsen/eval/official/budget_truth_audit.py hesf_coarsen/eval/official/acm_closure_compression.py hesf_coarsen/eval/official/imdb_constraint_compression.py hesf_coarsen/eval/official/gate21_18_decision.py
git diff --check
```

### Task 7: Requirement Checklist And Git

**Files:**
- Output: `outputs/gate21_18_smoke/gate21_18_requirement_checklist.md`

- [ ] **Step 1: Check P0-P13**

Write PASS/FAIL for every attachment section, with concrete failure reason for any remaining gap.

- [ ] **Step 2: Commit and push code/output metadata**

Stage only Gate21.18 code/tests/plan and lightweight output CSV/JSON/MD files. Do not stage large `.dat` exports.

```powershell
git status --short
git add <Gate21.18 files>
git commit -m "Add Gate21.18 budget truth real compression"
git push origin main
```
