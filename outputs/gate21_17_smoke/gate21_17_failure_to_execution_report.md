# Gate21.17 Failure-to-Execution Report

## export/schema failure

- none

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
