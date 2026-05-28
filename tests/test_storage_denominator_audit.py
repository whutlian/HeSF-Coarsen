from __future__ import annotations


def test_storage_denominator_audit_uses_three_explicit_denominators() -> None:
    from hesf_coarsen.eval.official.storage_denominator_audit import storage_denominator_audit

    row = storage_denominator_audit(
        {
            "dataset": "DBLP",
            "method": "APV12",
            "method_artifact_bytes": 25,
            "original_native_full_hgb_text_bytes": 100,
            "export_full_hgb_text_bytes": 50,
            "current_control_artifact_bytes": 200,
        }
    )

    assert row["ratio_vs_original_native_full_hgb_text"] == 0.25
    assert row["ratio_vs_export_full_hgb_text"] == 0.5
    assert row["ratio_vs_current_control_artifact"] == 0.125
    assert row["ratio_field_name_consistent"] is True
    assert "ratio_vs_native_full_text" not in row
