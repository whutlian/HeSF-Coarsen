# Gate21.15 Requirement Checklist

## Decision Flags

- [PASS] FULL_NATIVE_READY_BY_DATASET
- [PASS] EXPORT_FULL_FIDELITY_PASS_BY_DATASET
- [PASS] MAIN_TABLE_HAS_DBLP_ACM_IMDB
- [FAIL] HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE
- [FAIL] HESF_RCS_REP_TASK_RESULTS_READY
- [FAIL] STRUCTURAL_BASELINES_READY
- [PASS] EXTERNAL_TP_BASELINES_CLONED_OR_IMPLEMENTED
- [FAIL] EXTERNAL_TP_TASK_RESULTS_READY
- [PASS] FREEHGC_STANDARD_READY_OR_HARD_FAILURE_RECORDED
- [FAIL] FREEHGC_SCORE_TP_READY
- [PASS] BUDGET_MATCH_AUDIT_PASS
- [PASS] NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE
- [PASS] NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS
- [FAIL] STAGE_REPORT_TABLE_READY

## Attachment Sections

- [PASS] 0 Core positioning: main rows are official-unmodified, schema-preserving, target-preserving protocol rows or explicit failures.
- [FAIL] 1 Representative method: HeSF-RCS-Rep rows exist, but no dataset can select a representative without validation metrics.
- [PASS] 2 Protocol separation: adapter, storage-only, and standard-condensation rows are kept out of the main official table.
- [PASS] 3 Datasets: DBLP, ACM, and IMDB are present.
- [PASS] 4 Compression budgets: structural budgets 0.50/0.30/0.20/0.16/0.12 and support-node budgets 0.30/0.50 are represented in rows or audits.
- [PASS] 5 Required method families: full/export, internal, structural, external TP, HeSF-RCS-auto, and HeSF-RCS-Rep rows are emitted.
- [PASS] 6 External code policy: required public repositories are audited under external_repos/ with clone/failure metadata.
- [PASS] 7 New/updated files: Gate21.15 protocol, budget, decision, summarizer, external repo, TP, standard-condensation, and runner modules/scripts are present.
- [PASS] 8 Outputs: all required CSV/JSON/MD artifacts are written.
- [PASS] 9 Main table schema: required Gate21.15 fields are present.
- [PASS] 10 Representative selection output: uses_test_for_selection is false for all rows.
- [PASS] 11 Seed policy: quick-mode expected seed counts are recorded; missing task rows are failures, not successes.
- [PASS] 12 CLI: runner supports the required datasets/budgets/mode/run/clone/resume/output arguments.
- [PASS] 13 Decision flags: all required decision flags are emitted.
- [PASS] 14 Failure handling: unavailable baselines are explicit failure rows with failure_type/failure_reason.
- [PASS] 15 Anti-local-optimum: no new APV variant tuning was introduced; work pushes outward to DBLP/ACM/IMDB and external audits.
- [FAIL] 16 Success criteria: minimum/strong success is not met because HeSF-RCS-Rep validation selection, structural baselines, and external TP task metrics are incomplete.

## Artifact Checks

- [PASS] DBLP/ACM/IMDB datasets are present in the main table.
- [PASS] Main-table schema contains all required Gate21.15 fields.
- [PASS] HeSF-RCS-Rep selection rows assert uses_test_for_selection=false.
- [PASS] External repositories are audited in CSV and JSON.
- [PASS] Missing or incompatible baselines are emitted as explicit hard failure rows.
- [FAIL] Full stage-report readiness is not claimed while external/structural baselines and HeSF-RCS-Rep task rows remain incomplete.
