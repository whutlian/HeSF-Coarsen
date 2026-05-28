from __future__ import annotations

import importlib


def _decision_module():
    spec = importlib.util.find_spec("hesf_coarsen.eval.official.gate21_7_decision")
    assert spec is not None, "gate21_7_decision module must exist"
    return importlib.import_module("hesf_coarsen.eval.official.gate21_7_decision")


def _eligible_external_tp_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "method": "Random-HG-TP",
        "protocol": "schema_preserving_tp",
        "external_baseline": True,
        "official_hgb_exported": True,
        "official_sehgnn_unmodified": True,
        "training_executed": True,
        "success_count": 3,
        "test_micro_f1": 0.91,
        "test_macro_f1": 0.89,
        "missing_dependency": False,
        "eligible_for_tp_main_comparison": True,
    }
    row.update(overrides)
    return row


def test_external_tp_main_eligibility_requires_export_unmodified_training_and_metrics() -> None:
    module = _decision_module()
    external_tp_main_eligible = getattr(module, "external_tp_main_eligible", None)
    assert callable(external_tp_main_eligible), "external_tp_main_eligible must be reusable"

    assert external_tp_main_eligible(_eligible_external_tp_row()) is True
    assert external_tp_main_eligible(_eligible_external_tp_row(official_hgb_exported=False)) is False
    assert external_tp_main_eligible(_eligible_external_tp_row(official_sehgnn_unmodified=False)) is False
    assert external_tp_main_eligible(_eligible_external_tp_row(training_executed=False)) is False
    assert external_tp_main_eligible(_eligible_external_tp_row(test_micro_f1="NaN")) is False
    assert external_tp_main_eligible(_eligible_external_tp_row(test_macro_f1="NaN")) is False


def test_freehgc_missing_dependency_cannot_be_ready() -> None:
    module = _decision_module()
    evaluate_external_tp_readiness = getattr(module, "evaluate_external_tp_readiness", None)
    assert callable(evaluate_external_tp_readiness), "evaluate_external_tp_readiness must be reusable"

    decision = evaluate_external_tp_readiness(
        [
            _eligible_external_tp_row(
                method="FreeHGC-TP",
                missing_dependency=True,
                missing_dependency_name="torch_sparse",
                failure_type="missing_dependency",
            )
        ],
        required_methods=("FreeHGC-TP",),
    )

    assert decision["EXTERNAL_TP_FREEHGC_READY"] is False
    assert decision["EXTERNAL_TP_TASK_RESULTS_READY"] is False
    assert "missing_dependency" in decision["method_status"]["FreeHGC-TP"]["missing_requirements"]
