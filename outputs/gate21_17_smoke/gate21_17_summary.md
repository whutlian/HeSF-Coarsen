# Gate21.17 Executed Stage Report Summary

- rows: 30

- FULL_NATIVE_READY_BY_DATASET: True
- EXPORT_FULL_FIDELITY_PASS_BY_DATASET: True
- ACM_EXPORT_CONSISTENCY_PASS: True
- IMDB_EXPORT_CONSISTENCY_PASS: True
- STRUCTURAL_BASELINES_SMOKE_EXECUTED_BY_DATASET: True
- STRUCTURAL_BASELINES_QUICK_READY_BY_DATASET: False
- EXTERNAL_TP_SMOKE_EXECUTED_BY_DATASET: True
- EXTERNAL_TP_QUICK_READY_BY_DATASET: True
- FREEHGC_SCORE_TP_SMOKE_EXECUTED_BY_DATASET: True
- CONDENSATION_SCORE_TP_SMOKE_EXECUTED_BY_DATASET: True
- HESF_RCS_AUTO_EXECUTED_BY_DATASET: True
- HESF_RCS_REP_SELECTED_WITHOUT_TEST_LEAKAGE: True
- HESF_RCS_REP_ACTUAL_VALIDATION_READY: False
- STAGE_REPORT_SMOKE_READY: True
- STAGE_REPORT_QUICK_READY: False
- NO_IMPLEMENTED_PENDING_ROWS_IN_FINAL_TABLE: True
- NO_DIAGNOSTIC_OR_ADAPTER_ROWS_IN_MAIN_TABLE: True
- NO_PLACEHOLDER_NUMERIC_VALUES_IN_SUCCESS_ROWS: True

## Task Metrics

- DBLP Full-native-SeHGNN = actual_structural= support_node= micro=0.9533802 macro=0.9498198
- DBLP Export-full-SeHGNN = actual_structural= support_node= micro=0.9533802 macro=0.9498198
- ACM Full-native-SeHGNN = actual_structural= support_node= micro=0.9384324 macro=0.93918
- ACM Export-full-SeHGNN = actual_structural= support_node= micro=0.9384324 macro=0.93918
- IMDB Full-native-SeHGNN = actual_structural= support_node= micro=0.697974 macro=0.6712752
- IMDB Export-full-SeHGNN = actual_structural= support_node= micro=0.697974 macro=0.6712752
- IMDB H6-node30 support_node_ratio=0.3 actual_structural=0.8926449630767522 support_node=0.3 micro=0.6570166000000001 macro=0.634685
- DBLP HeSF-RCS-auto structural16 structural_storage_ratio=0.16 actual_structural=0.15916430179078186 support_node=0.30003171582619725 micro=0.9497888 macro=0.946167
- DBLP HeSF-RCS-auto structural12 structural_storage_ratio=0.12 actual_structural=0.11951718894668302 support_node=0.30003171582619725 micro=0.9447888000000001 macro=0.9405382
- ACM HeSF-RCS-auto structural20 structural_storage_ratio=0.2 actual_structural=1.0 support_node=1.0 micro=0.9395659999999999 macro=0.9402370000000001
- IMDB HeSF-RCS-auto structural20 structural_storage_ratio=0.2 actual_structural=1.0 support_node=1.0 micro=0.6993510000000001 macro=0.67012
- DBLP Random-edge-relwise structural_storage_ratio=0.2 actual_structural=0.973352050915338 support_node=all_target_preserved micro=0.785211 macro=0.7811929999999999
- DBLP Degree-edge-relwise structural_storage_ratio=0.2 actual_structural=0.9733514448741529 support_node=all_target_preserved micro=0.7859149999999999 macro=0.78125
- DBLP Proportional-relation-budget structural_storage_ratio=0.2 actual_structural=0.9731947453502058 support_node=all_target_preserved micro=0.796831 macro=0.7913209999999999
- ACM Random-edge-relwise structural_storage_ratio=0.2 actual_structural=1.0 support_node=1.0 micro=0.9395659999999999 macro=0.9402370000000001
- IMDB Random-edge-relwise structural_storage_ratio=0.2 actual_structural=1.0 support_node=1.0 micro=0.6993510000000001 macro=0.67012
- DBLP Herding-HG-TP support_node_ratio=0.5 actual_structural=0.511910701649485 support_node=0.487925331883467 micro=0.870775 macro=0.867442
- DBLP FreeHGC-score-TP structural_storage_ratio=0.2 actual_structural=0.48523595039340783 support_node=0.43849395133886093 micro=0.780282 macro=0.775069
- ACM Herding-HG-TP support_node_ratio=0.5 actual_structural=1.0 support_node=1.0 micro=0.9395659999999999 macro=0.9402370000000001
- IMDB Herding-HG-TP support_node_ratio=0.5 actual_structural=1.0 support_node=1.0 micro=0.6993510000000001 macro=0.67012
- DBLP HGCond-score-TP-local support_node_ratio=0.5 actual_structural=0.5118805814025809 support_node=0.4876081736214943 micro=0.874296 macro=0.8713219999999999
- DBLP GCond-score-TP-local support_node_ratio=0.5 actual_structural=0.5119158908771333 support_node=0.4880612568528839 micro=0.867254 macro=0.8634350000000001
- ACM HGCond-score-TP-local support_node_ratio=0.5 actual_structural=1.0 support_node=1.0 micro=0.9395659999999999 macro=0.9402370000000001
- ACM GCond-score-TP-local support_node_ratio=0.5 actual_structural=1.0 support_node=1.0 micro=0.9395659999999999 macro=0.9402370000000001
- IMDB HGCond-score-TP-local support_node_ratio=0.5 actual_structural=1.0 support_node=1.0 micro=0.6993510000000001 macro=0.67012
- IMDB GCond-score-TP-local support_node_ratio=0.5 actual_structural=1.0 support_node=1.0 micro=0.6993510000000001 macro=0.67012
- DBLP HeSF-RCS-Rep structural_storage_ratio=0.12 actual_structural=0.11951718894668302 support_node=0.30003171582619725 micro=0.9447888000000001 macro=0.9405382
- ACM HeSF-RCS-Rep structural_storage_ratio=0.2 actual_structural=1.0 support_node=1.0 micro=0.9395659999999999 macro=0.9402370000000001
- IMDB HeSF-RCS-Rep structural_storage_ratio=0.2 actual_structural=1.0 support_node=1.0 micro=0.6993510000000001 macro=0.67012

## Concrete Failures

- ACM H6-node30: official_training_runtime_error | Traceback (most recent call last):
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\main.py", line 588, in <module>
    main(args)
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\main.py", line 22, in main
    g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = load_dataset(args)
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\utils.py", line 383, in load_dataset
    assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
RuntimeError: The size of tensor a (255619) must m
