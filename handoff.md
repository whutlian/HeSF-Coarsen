# Handoff

## Goal: what we are trying to build

Build and validate the Gate17 "real validation + occlusion + DBLP-aware prototype" stage for HeSF support selection.

The intended direction is to move beyond proxy-only support selection by adding:

- real validation-feedback block greedy selection;
- real validation-occlusion block scoring;
- DBLP-aware prototype condensation for large/ambiguous support buckets;
- strict no-test-label leakage checks;
- exact-budget reporting and paper-facing diagnostic tables.

The broader project goal is to determine whether the task-first HeSF/HeSF-TC line can produce compressed heterogeneous graphs that preserve task performance and operator/spectral structure under controlled support budgets.

## Current state: where the work stands right now

Branch:

- `gate17-real-validation-occlusion-prototype`

Latest pushed commit:

- `7a0f8e7 Add Gate17 real validation occlusion prototype`

Remote:

- `origin https://github.com/whutlian/HeSF-Coarsen.git`

Gate17 code, scripts, tests, and plan have been committed and pushed to GitHub.

The local Gate17 diagnostic run completed successfully:

- `420/420` runs succeeded.
- No OOM/GPU OOM occurred.
- `no_test_leakage = true`.
- `decision = PARTIAL_DBLP_BLOCKER`.
- Best validation-selected method: `HeSF-SS-dblp-aware-prototype`.
- Best validation-selected macro-F1 mean: `0.21916162489966343`.
- Best validation-selected accuracy mean: `0.2528520241615379`.
- Mean exact-budget macro gap vs strongest baseline: `0.0`.
- Teacher reliability: `false`.

Important caveat:

This Gate17 run is a local fast diagnostic run, not paper-level task-quality evidence. It used fast settings such as `task_epochs=0`, `max_paths=1`, and `feature-mode fast`. The results are valid for checking wiring, leakage, budget handling, selector behavior, prototype diagnostics, and output completeness, but should not be interpreted as final HGNN performance.

Verification already passed:

- `conda run -n pytorch python -m pytest ... -q`: `17 passed`.
- `conda run -n pytorch python -m py_compile ...`: passed.
- `git diff --check`: passed.

Packaged files for ChatGPT App analysis:

- `outputs/gate17_chatgpt_packages/gate17_main_tables_for_chatgpt.zip`
- `outputs/gate17_chatgpt_packages/gate17_diagnostics_for_chatgpt.zip`
- `outputs/gate17_chatgpt_packages/gate17_figures_for_chatgpt.zip`

## Files in flight: active files being modified

No tracked source files are currently modified after the Gate17 commit and push.

This handoff file is newly created:

- `handoff.md`

There are unrelated untracked local files that were not staged or committed:

- `build/`
- `docs/hesf_algorithm_flow.pdf`
- `docs/hesf_algorithm_flow.tex`
- `docs/hesf_algorithm_report.pptx`
- `docs/hesf_deck_montage.png`
- `docs/superpowers/plans/2026-05-17-next9-experiment-code.md`
- `exports/`
- `session.md`

Do not treat those as Gate17 changes unless the user explicitly asks to process them.

## Changed: what's been touched this session

New Gate17 plan:

- `docs/superpowers/plans/2026-05-22-gate17-real-validation-occlusion-prototype.md`

New experiment/audit/summary/plot scripts:

- `experiments/scripts/audit_gate17_code.py`
- `experiments/scripts/run_gate17_support_selection.py`
- `experiments/scripts/run_gate17_teacher_stability.py`
- `experiments/scripts/summarize_gate17.py`
- `experiments/scripts/plot_gate17_figures.py`

New selection helper module:

- `hesf_coarsen/task_first/selection/validation_selector.py`

Modified selection pipeline/configuration:

- `hesf_coarsen/task_first/selection/config.py`
- `hesf_coarsen/task_first/selection/contribution.py`
- `hesf_coarsen/task_first/selection/selector.py`
- `hesf_coarsen/task_first/selection/condensation.py`
- `hesf_coarsen/task_first/selection/pipeline.py`

New Gate17 tests:

- `tests/test_gate17_summary_aggregation.py`
- `tests/test_gate17_exact_budget_gaps.py`
- `tests/test_gate17_true_validation_selector.py`
- `tests/test_gate17_occlusion_importance.py`
- `tests/test_gate17_no_test_leakage.py`
- `tests/test_gate17_dblp_prototype_condensation.py`

Generated result locations:

- `outputs/gate17_tables/`
- `outputs/gate17_diagnostics/`
- `outputs/gate17_figures/`
- `outputs/gate17_chatgpt_packages/`

Key generated summaries:

- `outputs/gate17_tables/result.json`
- `outputs/gate17_tables/final_report.md`
- `outputs/gate17_tables/gate17_decision.md`
- `outputs/gate17_diagnostics/teacher_stability_report.md`

## Failed attempts: what didn't work and why

Full local evaluation with heavier feature/path settings was too slow for interactive local iteration.

- Earlier ACM smoke attempts using fuller feature construction and higher path counts did not finish quickly enough.
- The final local run intentionally used fast diagnostic settings to complete all datasets/seeds/ratios/methods.

DBLP prototype condensation initially exposed performance issues.

- Prototype construction had expensive per-node/per-key scans.
- Degree bucket computation also repeatedly scanned typed nodes.
- These were fixed by caching prototype type counts and making degree bucketing constant-time.

Parallel `conda run` calls on Windows caused a temporary-file conflict.

- Running pytest and py_compile through separate `conda run` commands in parallel triggered a conda temp file access error.
- The command was rerun serially and passed.

No OOM or GPU memory failure occurred.

- No server fallback command was needed.

## Next step: the single next thing to try

Upload the three Gate17 zip packages to ChatGPT App and ask it to analyze whether `PARTIAL_DBLP_BLOCKER` is mainly caused by the fast diagnostic protocol tying methods, or by a real weakness in the Gate17 selection/prototype design.

The analysis prompt should explicitly mention that the run used `task_epochs=0`, `max_paths=1`, and `feature-mode fast`, so the app should focus first on diagnostics, leakage, budget exactness, selector/prototype behavior, and whether a paper-level rerun is justified.
