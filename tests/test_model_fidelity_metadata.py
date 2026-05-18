from hesf_coarsen.accuracy.model_fidelity_registry import fidelity_record, validate_fidelity_row


def test_fidelity_record_marks_local_adapters_as_non_official() -> None:
    record = fidelity_record("sehgnn_lite")

    assert record["model_fidelity"] == "lite_adapter"
    assert record["official_repo"] == "no"
    assert record["official_preprocess"] == "no"
    assert record["adapter_mode"] == "lite"


def test_fidelity_record_tracks_official_repo_availability_without_claiming_integration() -> None:
    record = fidelity_record("official_sehgnn")

    assert record["model_name"] == "official_sehgnn"
    assert record["official_repo"] in {"available_not_integrated", "unavailable"}
    assert record["model_fidelity"] in {"official_not_integrated", "unavailable"}
    assert "ICT-GIMLab/SeHGNN" in record["repository"]


def test_validate_fidelity_row_requires_reporting_fields() -> None:
    row = {
        "model_name": "hettree_lite",
        "model_fidelity": "lite_adapter",
        "official_repo": "no",
        "official_preprocess": "no",
        "adapter_mode": "lite",
        "split_policy": "synthetic_stratified",
        "path_set": "lite",
        "max_hops": 2,
    }

    assert validate_fidelity_row(row)["ok"] is True


def test_validate_fidelity_row_reports_missing_fields() -> None:
    result = validate_fidelity_row({"model_name": "sehgnn_lite"})

    assert result["ok"] is False
    assert "model_fidelity" in result["missing_fields"]
