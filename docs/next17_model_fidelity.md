# Next17 Model Fidelity

The current Next17 run uses local SeHGNN/HETTREE-style adapters. It does not claim official SeHGNN or official HETTREE reproduction numbers.

## Fidelity Tags

- `official_repo=no`
- `official_preprocess=no`
- `adapter_mode=approximate`
- `path_set=lite`
- `full_target_inference=true` for Mode B rows

The separate official-named runner files are fidelity-labeled entrypoints that keep this limitation explicit. They should not be used to compare directly against paper SOTA numbers until official preprocessing and model code are connected.
