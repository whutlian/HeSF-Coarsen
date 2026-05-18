# Next17 Accuracy Branch

Next17 adds an accuracy-first branch without changing the existing preservation-first HeSF-LVC-P/S configs.

## Scope

- P/S mainline remains `lambda_spec={0.25,0.5}`, `lambda_conv=0`, `lambda_rel=0`.
- Target nodes are preserved or selected explicitly instead of being merged by default.
- Support nodes continue to use HeSF-style same-type coarsening.
- Every downstream result is tagged as Mode A `coarse_transfer` or Mode B `full_target_inference`.
- Lite adapters are marked as `official_repo=no` and `adapter_mode=approximate`.

## Blocks

- P1 target-preserve support coarsening.
- P2 Hybrid-A keep-all-target and Hybrid-B target-selection plus support coarsening.
- P3 full-target inference protocol split.
- P4 meta-reconstruction and distillation diagnostics.
- P5 type-wise budget reporting.
- P6 model fidelity tagging.

The final result directory is `outputs/exp_next17_accuracy_branch_20260518`.
