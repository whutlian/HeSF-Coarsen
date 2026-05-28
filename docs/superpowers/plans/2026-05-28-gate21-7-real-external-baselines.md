# Gate21.7 Execution Plan

1. Tighten Gate21.7 pass/fail logic so estimates, missing dependencies, empty hashes, and NaN task metrics cannot become READY rows.
2. Add repaired diagnostics for semantic coverage, feature-shape-preserving ablations, cache/hash audit, adapter package accounting v2, FreeHGC environment probing, and storage/system costs.
3. Add Gate21.7 runners under `experiments/scripts/` that write the requested `outputs/gate21_7_icde_ready/` layout without overwriting Gate21.6 outputs.
4. Execute local `conda run -n pytorch` tests and Gate21.7 quick run; record FreeHGC as real upstream GitHub clone execution or hard dependency/runtime failure with setup commands.
5. Generate decision JSON/MD and both requirement checklists, verify every prompt item, then commit and push code changes to `main`.
