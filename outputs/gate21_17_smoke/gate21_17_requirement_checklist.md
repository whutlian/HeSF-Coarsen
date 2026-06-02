# Gate21.17 Requirement Checklist

## Decision Flags

- [PASS] FULL_NATIVE_READY_BY_DATASET
- [PASS] EXPORT_FULL_FIDELITY_PASS_BY_DATASET
- [PASS] ACM_EXPORT_CONSISTENCY_PASS
- [PASS] IMDB_EXPORT_CONSISTENCY_PASS
- [PASS] STRUCTURAL_BASELINES_SMOKE_EXECUTED_BY_DATASET
- [FAIL] STRUCTURAL_BASELINES_QUICK_READY_BY_DATASET
- [PASS] EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET
- [PASS] EXTERNAL_TP_QUICK_READY_BY_DATASET
- [PASS] FREEHGC_SCORE_TP_SMOKE_EXECUTED_BY_DATASET
- [PASS] CONDENSATION_SCORE_TP_SMOKE_EXECUTED_BY_DATASET
- [PASS] HESF_RCS_AUTO_EXECUTED_BY_DATASET
- [PASS] HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE
- [FAIL] HESF_RCS_REP_ACTUAL_VALIDATION_READY
- [PASS] STAGE_REPORT_SMOKE_READY
- [FAIL] STAGE_REPORT_QUICK_READY
- [PASS] NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE
- [PASS] NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE
- [PASS] NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS

## Attachment Sections

- [PASS] P0 official training queue emitted and formerly pending rows are resolved to metrics or concrete failures.
- [PASS] P1 structural baselines smoke rows produced task metrics where required.
- [PASS] P2 external TP smoke rows produced task metrics where required.
- [PASS] P3 ACM consistency audit emitted and ACM rows moved to metrics/trace.
- [PASS] P4 IMDB consistency audit emitted and IMDB rows moved to metrics/trace.
- [PASS] P5 HeSF-RCS representative selector avoids test leakage and emits test-oracle diagnostic row.
- [PASS] P6 external repos audited and local score-TP proxies executed under TP protocol.
- [PASS] P7 smoke CLI mode summarized.
- [PASS] P8 main table schema emitted.
- [PASS] P9 decision flags emitted.
- [PASS] P10 summary and failure report emitted.
- [PASS] P11 minimal smoke acceptance.
- [PASS] P12 local TP proxy priority followed; no hard pending placeholders remain.
