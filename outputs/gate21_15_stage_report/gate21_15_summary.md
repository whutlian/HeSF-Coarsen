# Gate21.15 Stage-Report Benchmark Table

- STAGE_REPORT_TABLE_READY: False
- main_official_rows: 168
- failure_rows_recorded: 316

## Decision Flags

- FULL_NATIVE_READY_BY_DATASET: True
- EXPORT_FULL_FIDELITY_PASS_BY_DATASET: True
- MAIN_TABLE_HAS_DBLP_ACM_IMDB: True
- HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE: False
- HESF_RCS_REP_TASK_RESULTS_READY: False
- STRUCTURAL_BASELINES_READY: False
- EXTERNAL_TP_BASELINES_CLONED_OR_IMPLEMENTED: True
- EXTERNAL_TP_TASK_RESULTS_READY: False
- FREEHGC_STANDARD_READY_OR_HARD_FAILURE_RECORDED: True
- FREEHGC_SCORE_TP_READY: False
- BUDGET_MATCH_AUDIT_PASS: True
- NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE: True
- NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS: True
- STAGE_REPORT_TABLE_READY: False

## Key Blockers

- ACM H6-node30: failed_runtime - Traceback (most recent call last):
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\main.py", line 588, in <module>
    main(args)
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\main.py", line 22, in main
    g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = load_dataset(args)
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\utils.py", line 383, in load_dataset
    assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
RuntimeError: The size of tensor a (255619) must match the size of tensor b (213104) at non-singleton dimension 0
- ACM flatten-node30: missing_task_metric - flatten-node30 has no official SeHGNN task metric for ACM.
- ACM TypedHash-node30: missing_task_metric - TypedHash-node30 has no official SeHGNN task metric for ACM.
- IMDB flatten-node30: missing_task_metric - flatten-node30 has no official SeHGNN task metric for IMDB.
- IMDB TypedHash-node30: missing_task_metric - TypedHash-node30 has no official SeHGNN task metric for IMDB.
- DBLP Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.50 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.30 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.20 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.16 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.12 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.50 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.30 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.20 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.16 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.12 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Proportional-relation-budget: not_executed - Proportional-relation-budget at structural budget 0.50 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Proportional-relation-budget: not_executed - Proportional-relation-budget at structural budget 0.30 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Proportional-relation-budget: not_executed - Proportional-relation-budget at structural budget 0.20 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Proportional-relation-budget: not_executed - Proportional-relation-budget at structural budget 0.16 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- DBLP Proportional-relation-budget: not_executed - Proportional-relation-budget at structural budget 0.12 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.50 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.30 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.20 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.16 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Random-edge-relwise: not_executed - Random-edge-relwise at structural budget 0.12 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.50 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.30 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.20 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.16 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ACM Degree-edge-relwise: not_executed - Degree-edge-relwise at structural budget 0.12 is planned for Gate21.15 quick, but no official SeHGNN task metric exists locally.
- ... 286 additional failure rows in gate21_15_failures.json
