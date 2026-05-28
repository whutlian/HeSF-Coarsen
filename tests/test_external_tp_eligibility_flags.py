from __future__ import annotations

import importlib
from pathlib import Path


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


def test_freehgc_preflight_reports_all_required_hgb_model_files(tmp_path) -> None:
    from hesf_coarsen.eval.official.freehgc_env_bridge import freehgc_preflight

    hgb_root = tmp_path / "FreeHGC" / "HGB"
    hgb_root.mkdir(parents=True)
    (tmp_path / "FreeHGC" / "README.md").write_text("FreeHGC\n", encoding="utf-8")
    (hgb_root / "train_hgb.py").write_text("from model_hgb import *\nfrom model_SeHGNN import *\n", encoding="utf-8")
    (hgb_root / "data_hgb.py").write_text("", encoding="utf-8")

    preflight = freehgc_preflight(freehgc_root=tmp_path / "FreeHGC")

    assert str(Path("HGB") / "model_hgb.py") in preflight["missing_dependency_name"]
    assert str(Path("HGB") / "model_SeHGNN.py") in preflight["missing_dependency_name"]


def test_freehgc_metric_parser_reads_training_logger_info_lines() -> None:
    from hesf_coarsen.eval.official.freehgc_protocol_runner import parse_freehgc_metrics

    metrics = parse_freehgc_metrics(
        "[INFO] Epoch 1, Times(s): 0.2676, mac,mic: "
        "Tra(0.5512 0.5455), Val(0.2494 0.2593), Tes(0.3177 0.3250) Val_loss(1.6658)\n"
        "[INFO] macro: Best Val 0.2494, Best Test 0.3177\n"
        "[INFO] micro: Best Val 0.2593, Best Test 0.3250\n"
    )

    assert metrics["validation_macro_f1"] == 0.2494
    assert metrics["validation_micro_f1"] == 0.2593
    assert metrics["test_macro_f1"] == 0.3177
    assert metrics["test_micro_f1"] == 0.3250
