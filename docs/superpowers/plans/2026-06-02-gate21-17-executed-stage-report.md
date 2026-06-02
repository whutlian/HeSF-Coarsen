# Gate21.17 Executed Stage Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Gate21.16 pending official-training rows into Gate21.17 rows with either official SeHGNN task metrics or concrete export/runtime failure traces.

**Architecture:** Add a Gate21.17 protocol layer, an official training queue, a stage-report executor/table builder, validation-metric selection, and summary/decision scripts. The runner reuses Gate21.16 evidence and existing official SeHGNN runner functions, but it is not allowed to emit `implemented_pending_official_training` in final eligible rows.

**Tech Stack:** Python, pytest, CSV/JSON artifacts, existing `hesf_coarsen.eval.official` helpers, local conda env `pytorch`, official SeHGNN HGB runner.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_gate21_17_executed_stage_report_contract.py`

- [ ] **Step 1: Write failing tests**

```python
from experiments.scripts.run_gate21_17_executed_stage_report import build_arg_parser, run
from hesf_coarsen.eval.official.gate21_17_decision import GATE21_17_DECISION_FLAGS, gate21_17_decision
from hesf_coarsen.eval.official.official_training_queue import build_training_queue, verify_hgb_export_dir
from hesf_coarsen.eval.official.stage_report_table import GATE21_17_MAIN_FIELDS


def test_gate21_17_protocol_has_required_fields_and_flags():
    assert {"stdout_path", "stderr_path", "raw_hgb_text_byte_ratio"}.issubset(GATE21_17_MAIN_FIELDS)
    assert {"NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE", "STAGE_REPORT_SMOKE_READY"}.issubset(GATE21_17_DECISION_FLAGS)


def test_training_queue_selects_only_eligible_pending_rows(tmp_path):
    export_dir = tmp_path / "DBLP"
    for name in ("node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat"):
        (export_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (export_dir / name).write_text("", encoding="utf-8")
    rows = [{"dataset": "DBLP", "method": "Random-edge-relwise", "schema_compatible": True, "target_preserving": True, "official_hgb_exported": True, "official_sehgnn_unmodified": True, "training_executed": False, "failure_type": "implemented_pending_official_training", "export_dir": str(export_dir), "requested_budget_type": "structural_storage_ratio", "requested_budget": "0.2"}]
    queue = build_training_queue(rows)
    assert len(queue) == 1
    assert verify_hgb_export_dir(export_dir)["export_dir_ready"] is True


def test_preflight_runner_replaces_pending_with_concrete_failures(tmp_path):
    decision = run(build_arg_parser().parse_args(["--mode", "preflight", "--datasets", "DBLP", "ACM", "IMDB", "--output", str(tmp_path)]))
    main = (tmp_path / "gate21_17_main_official_table.csv").read_text(encoding="utf-8")
    assert "implemented_pending_official_training" not in main
    assert (tmp_path / "gate21_17_training_queue.csv").exists()
    assert decision["NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE"] is True
```

- [ ] **Step 2: Run RED**

Run: `conda run -n pytorch python -m pytest tests/test_gate21_17_executed_stage_report_contract.py -q`

Expected: fails because Gate21.17 modules/scripts do not exist.

### Task 2: Gate21.17 Protocol and Decision

**Files:**
- Create: `hesf_coarsen/eval/official/stage_report_table.py`
- Create: `hesf_coarsen/eval/official/gate21_17_decision.py`

- [ ] **Step 1: Implement `GATE21_17_MAIN_FIELDS`**

Use the exact prompt schema, including `stdout_path` and `stderr_path`. Provide row normalization helpers that coerce boolean columns and never default a failed eligible row to a vague pending failure.

- [ ] **Step 2: Implement decision flags**

`gate21_17_decision()` must compute all required flags and mark smoke ready only when DBLP structural metrics, DBLP external TP metrics, ACM metric, IMDB metric, full/export readiness, and no pending rows are all present.

- [ ] **Step 3: Run focused tests**

Run: `conda run -n pytorch python -m pytest tests/test_gate21_17_executed_stage_report_contract.py -q`

Expected: remaining failures move to missing queue/runner modules.

### Task 3: Official Training Queue

**Files:**
- Create: `hesf_coarsen/eval/official/official_training_queue.py`

- [ ] **Step 1: Implement export verification**

`verify_hgb_export_dir(path)` returns a row with `export_dir_ready`, one boolean per required file, and a concrete `failure_type="export_schema_failure"` if any required file is missing.

- [ ] **Step 2: Implement queue builder**

`build_training_queue(rows)` selects rows with schema-compatible, target-preserving, official exported, unmodified official SeHGNN, `training_executed=false`, and `failure_type=implemented_pending_official_training`. Queue rows include dataset, method, budget fields, `export_dir`, hashes, seeds, and `source_row_id`.

- [ ] **Step 3: Implement queue executor**

`execute_training_queue()` calls `build_official_hgb_command()` and `run_native_command()` for ready export dirs. For missing exports, it writes `training_executed=false`, `success=false`, `failure_type=export_schema_failure`, and `failure_reason` listing missing files. Runtime errors become `official_training_runtime_error` with stdout/stderr paths.

### Task 4: Compatibility Repair Wrappers and Validation Resolver

**Files:**
- Create: `hesf_coarsen/eval/official/acm_consistency_export_repair.py`
- Create: `hesf_coarsen/eval/official/imdb_consistency_export_repair.py`
- Create: `hesf_coarsen/eval/official/validation_metric_resolver.py`
- Create or extend: `hesf_coarsen/eval/official/condensation_score_tp_local.py`

- [ ] **Step 1: Wrap existing ACM/IMDB repair audits**

Expose Gate21.17 field names while reusing Gate21.16 ACM/IMDB consistency implementations.

- [ ] **Step 2: Implement representative selection**

Prefer actual validation metrics, then validation proxy, never test metrics for main Rep. Emit diagnostic-only `HeSF-RCS-TestOracleRep` with `eligible_for_main_table=false`.

- [ ] **Step 3: Provide HGCond/GCond local TP proxy rows**

Reuse the Gate21.16 condensation score implementation and label rows as local TP proxies under the official target-preserving protocol.

### Task 5: Runner and Summarizer

**Files:**
- Create: `hesf_coarsen/eval/official/stage_report_executor.py`
- Create: `experiments/scripts/run_gate21_17_executed_stage_report.py`
- Create: `experiments/scripts/summarize_gate21_17_executed_stage_report.py`

- [ ] **Step 1: Implement preflight mode**

Read Gate21.16 quick rows, build queue, verify exports, replace every final eligible `implemented_pending_official_training` with `export_schema_failure` or metric rows, and write all required Gate21.17 CSV/MD/JSON outputs.

- [ ] **Step 2: Implement smoke/quick/full modes**

Map mode to graph/training seeds and required smoke rows. Execute available export dirs through official SeHGNN on GPU by default. If CUDA OOM occurs, capture the trace and provide server command guidance in the final response.

- [ ] **Step 3: Implement summary and checklist**

Write `gate21_17_summary.md`, `gate21_17_failure_to_execution_report.md`, and `gate21_17_requirement_checklist.md` with P0-P12 pass/fail entries.

### Task 6: Verification and Git

**Files:**
- All Gate21.17 files above
- Outputs under `outputs/gate21_17_preflight`, `outputs/gate21_17_smoke`, and, if feasible, `outputs/gate21_17_quick`

- [ ] **Step 1: Run tests**

Run: `conda run -n pytorch python -m pytest tests/test_gate21_17_executed_stage_report_contract.py tests/test_gate21_16_implementation_first_contract.py -q`

- [ ] **Step 2: Run smoke**

Run: `conda run -n pytorch python -m experiments.scripts.run_gate21_17_executed_stage_report --mode smoke --datasets DBLP ACM IMDB --output outputs/gate21_17_smoke`

- [ ] **Step 3: Check final table**

Run: `Import-Csv outputs/gate21_17_smoke/gate21_17_main_official_table.csv | Where-Object { $_.failure_type -eq 'implemented_pending_official_training' }`

Expected: no rows.

- [ ] **Step 4: Push**

Run non-interactive `git add`, `git commit`, and `git push origin main` for Gate21.17 code/tests/outputs only.
