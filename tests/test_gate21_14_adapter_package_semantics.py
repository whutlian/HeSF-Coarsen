from __future__ import annotations

from hesf_coarsen.eval.official.gate21_14_decision import gate21_14_decision


def test_apv12_rp64_adapter_uses_static_inference_ratio_and_is_not_main_table() -> None:
    rows = [
        {
            "base_method": "HeSF-RCS-APV12",
            "adapter_method": "random_projection_dim64",
            "training_executed": True,
            "success": True,
            "test_micro_f1": 0.948,
            "test_macro_f1": 0.944,
            "static_inference_package_ratio": 0.09,
            "transform_recipe_package_ratio": 0.001,
            "eligible_for_adapter_table": True,
            "eligible_for_official_main_table": False,
        }
    ]

    flags = gate21_14_decision(adapter_rows=rows)

    assert flags["APV12_RP64_ADAPTER_RESTORED"] is True
    assert flags["APV16_RP64_ADAPTER_READY"] is False
