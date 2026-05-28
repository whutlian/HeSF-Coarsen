from __future__ import annotations

import importlib
import math


def _evaluate_external_tp_readiness():
    spec = importlib.util.find_spec("hesf_coarsen.eval.official.gate21_7_decision")
    assert spec is not None, "gate21_7_decision module must exist"
    module = importlib.import_module("hesf_coarsen.eval.official.gate21_7_decision")
    fn = getattr(module, "evaluate_external_tp_readiness", None)
    assert callable(fn), "evaluate_external_tp_readiness must be reusable by runners/summarizers"
    return fn


def test_external_tp_task_results_not_ready_without_training_and_metrics() -> None:
    evaluate_external_tp_readiness = _evaluate_external_tp_readiness()

    decision = evaluate_external_tp_readiness(
        [
            {
                "method": "Random-HG-TP",
                "official_hgb_exported": True,
                "training_executed": False,
                "success_count": 1,
                "test_micro_f1": math.nan,
                "test_macro_f1": math.nan,
            }
        ],
        required_methods=("Random-HG-TP",),
    )

    assert decision["EXTERNAL_TP_TASK_RESULTS_READY"] is False
    assert decision["method_status"]["Random-HG-TP"]["ready"] is False
    assert "training_executed" in decision["method_status"]["Random-HG-TP"]["missing_requirements"]
    assert "test_micro_f1" in decision["method_status"]["Random-HG-TP"]["missing_requirements"]
    assert "test_macro_f1" in decision["method_status"]["Random-HG-TP"]["missing_requirements"]
