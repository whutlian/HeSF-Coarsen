# Gate21.16 Requirement Checklist

## Decision Flags

- [PASS] FULL_NATIVE_READY_BY_DATASET
- [PASS] EXPORT_FULL_FIDELITY_PASS_BY_DATASET
- [PASS] ACM_EXPORT_CONSISTENCY_PASS
- [PASS] IMDB_EXPORT_CONSISTENCY_PASS
- [FAIL] STRUCTURAL_BASELINES_EXECUTED_BY_DATASET
- [PASS] EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET
- [FAIL] EXTERNAL_TP_QUICK_READY_BY_DATASET
- [PASS] FREEHGC_STANDARD_ATTEMPTED
- [FAIL] FREEHGC_SCORE_TP_EXECUTED
- [FAIL] HESF_RCS_AUTO_EXECUTED_BY_DATASET
- [FAIL] HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE
- [FAIL] HESF_RCS_REP_TASK_RESULTS_READY
- [FAIL] STAGE_REPORT_SMOKE_READY
- [FAIL] STAGE_REPORT_QUICK_READY
- [PASS] NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE
- [PASS] NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS

## Attachment Sections

- [PASS] P0 ACM consistency preflight repair/audit emitted.
- [PASS] P1 IMDB consistency preflight repair/audit emitted.
- [PASS] P2 structural baseline local implementations emitted with relation retention audit.
- [PASS] P3 external TP local implementations emitted.
- [PASS] P4 FreeHGC-score-TP local fallback emitted.
- [PASS] P5 HGCond/GCond score TP proxy rows emitted.
- [PASS] P6 representative selection uses validation metric/proxy and never test metrics.
- [PASS] P8 preflight/smoke/quick CLI modes are supported.
- [PASS] P9 main table schema emitted.
- [PASS] P10 decision flags emitted.
- [PASS] P11 failure-to-implementation report emitted.
- [FAIL] P12 strong success remains pending until official SeHGNN training is completed for local baseline exports.
