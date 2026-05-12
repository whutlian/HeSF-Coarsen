# Experiments

This package contains lightweight runners for reproducible HeSF-Coarsen experiments.

Run scripts from the repository root, for example:

```bash
python experiments/scripts/run_sanity.py --output outputs/experiments/sanity
python experiments/scripts/summarize_experiments.py outputs/experiments --output outputs/experiments/summary
```

Each runner writes per-run metadata and diagnostics that can be collected by
`collect_diagnostics.py` and summarized by `summarize_experiments.py`.
