# Gate21.17 Failure-to-Execution Report

## export/schema failure

- ACM HeSF-RCS-auto structural20: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB HeSF-RCS-auto structural20: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Random-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Degree-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Proportional-relation-budget: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM Random-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB Random-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Herding-HG-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP FreeHGC-score-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM Herding-HG-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB Herding-HG-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP HGCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP GCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM HGCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM GCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB HGCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB GCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM HeSF-RCS-auto structural20: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB HeSF-RCS-auto structural20: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Random-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Degree-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Proportional-relation-budget: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM Random-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB Random-edge-relwise: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP Herding-HG-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP FreeHGC-score-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM Herding-HG-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB Herding-HG-TP: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP HGCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- DBLP GCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM HGCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- ACM GCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB HGCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat
- IMDB GCond-score-TP-local: export_schema_failure | Missing required HGB files in export_dir: node.dat;link.dat;label.dat;label.dat.test;info.dat

## official training runtime failure

- ACM H6-node30: official_training_runtime_error | Traceback (most recent call last):
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\main.py", line 588, in <module>
    main(args)
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\main.py", line 22, in main
    g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = load_dataset(args)
  File "D:\HeSF-Coarsen\external\SeHGNN\hgb\utils.py", line 383, in load_dataset
    assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
RuntimeError: The size of tensor a (255619) must m

## budget infeasible

- none

## external repo failure with local fallback used

- FreeHGC : missing_required_file | Missing required files: HGB/model_hgb.py

## validation metric missing

- none

## intentionally diagnostic-only

- none
