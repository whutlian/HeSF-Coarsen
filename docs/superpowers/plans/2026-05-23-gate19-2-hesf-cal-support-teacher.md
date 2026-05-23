# Gate19.2 HeSF-CAL Support Teacher Plan

## Objective

Implement and run Gate19.2 as a single-seed diagnostic gate for task-calibrated support graphs and support-teacher STC distillation, using `primary_eval_mode=compressed_projected` and validation-only selection.

## Required Guardrails

1. Keep all model, calibration, ensemble, teacher, and student selection off test labels.
2. Separate validation-selected winners from test-oracle and Pareto diagnostics.
3. Treat calibrated H6, flatten, TypedHash, and best-support as formal baselines.
4. Include HeSF-CAL single methods, HeSF-CAL ensembles, STC references, compressed STC, and support-teacher STC students in the cost/Pareto outputs.
5. Mark unavailable support-teacher distillation explicitly instead of fabricating teacher logits.
6. Write the required preflight, summary, diagnostic, and checklist files under `outputs/gate19_2`.

## Implementation Steps

1. Add tests for Gate19.2 selection semantics, ensemble constraints, distillation failure handling, Pareto cost accounting, and decision labels.
2. Add `hesf_coarsen/eval/logit_ensemble.py` for calibration metrics and validation-selected logit ensembles.
3. Add `hesf_coarsen/task_first/feature_condensation/support_teacher_distill.py` for support-teacher student training with CE + KL + margin loss and explicit unavailable-teacher handling.
4. Add `experiments/scripts/summarize_gate19_2.py` with validation-selected, test-oracle, and Pareto summaries.
5. Add `experiments/scripts/run_gate19_2_hesf_cal_support_teacher.py`, reusing Gate19/Gate19.1 utilities and writing every required Gate19.2 artifact.
6. Run the exact requested local conda command and the summarizer.
7. Verify outputs, write requirement checklist, and commit/push code changes to `main`.

## Verification

Use:

```powershell
conda run -n pytorch python -m pytest tests/test_gate19_2_hesf_cal_support_teacher.py -q
conda run -n pytorch python -m experiments.scripts.run_gate19_2_hesf_cal_support_teacher --datasets ACM DBLP IMDB --dataset-seeds ACM:23456 DBLP:23456 IMDB:45678 --support-ratios 0.30 0.50 0.70 1.00 --feature-cache-ratios 0.30 0.50 0.70 1.00 --nested-calibration-repeats 5 --primary-eval-mode compressed_projected --output-dir outputs/gate19_2
conda run -n pytorch python -m experiments.scripts.summarize_gate19_2 --input-dir outputs/gate19_2 --output-dir outputs/gate19_2
```
