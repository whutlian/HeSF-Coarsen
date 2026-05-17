from pathlib import Path

from experiments.scripts._common import write_csv, write_json
from experiments.scripts.audit_next11_dee_consistency import audit_next11_dee_consistency


def test_audit_detects_same_scale_metrics(tmp_path: Path):
    paper = tmp_path / "paper"
    resource = tmp_path / "resource"
    out = tmp_path / "out"
    write_csv(
        paper / "final_main_table_by_seed.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345, "DEE": 0.1, "coarse_nodes": 5}],
    )
    write_csv(
        resource / "hgb_resource_logged_runs.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345, "DEE": 0.1, "coarse_nodes": 5}],
    )

    result = audit_next11_dee_consistency(paper_final=paper, resource_logged=resource, output=out)

    assert result["conclusion"] == "same_metric"
    assert (out / "dee_consistency_by_run.csv").exists()
    text = (out / "dee_consistency_by_run.csv").read_text(encoding="utf-8")
    assert "same_metric" in text


def test_audit_detects_10x_field_mismatch(tmp_path: Path):
    paper = tmp_path / "paper"
    resource = tmp_path / "resource"
    run = resource / "runs" / "next10_resource_ACM_HeSF_LVC_P_seed12345"
    out = tmp_path / "out"
    write_csv(
        paper / "final_main_table_by_seed.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345, "DEE": 0.01, "coarse_nodes": 5}],
    )
    write_csv(
        resource / "hgb_resource_logged_runs.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345, "DEE": 0.1, "coarse_nodes": 5}],
    )
    write_json(run / "metadata.json", {"status": "success", "method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345})
    write_json(
        run / "level_1" / "diagnostics.json",
        {
            "spectral": {"dirichlet_energy_relative_error": 0.01},
            "cumulative_spectral": {"dirichlet_energy_relative_error": 0.1},
        },
    )

    result = audit_next11_dee_consistency(paper_final=paper, resource_logged=resource, output=out)

    assert result["conclusion"] in {"field_mismatch_fixed", "different_metric_renamed"}
    row_text = (out / "dee_consistency_by_run.csv").read_text(encoding="utf-8")
    assert "metric_scale_ratio_resource_to_paper" in row_text
    assert "resource_logged_cumulative_dee" in row_text


def test_audit_warns_on_ambiguous_dee_without_source_field(tmp_path: Path):
    paper = tmp_path / "paper"
    resource = tmp_path / "resource"
    out = tmp_path / "out"
    write_csv(
        paper / "final_main_table_by_seed.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345, "DEE": 0.01}],
    )
    write_csv(
        resource / "hgb_resource_logged_runs.csv",
        [{"method": "HeSF-LVC-P", "dataset": "ACM", "seed": 12345, "DEE": 0.1}],
    )

    audit_next11_dee_consistency(paper_final=paper, resource_logged=resource, output=out)

    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "ambiguous" in text.lower()


def test_audit_allows_full_graph_missing_spectral_fields(tmp_path: Path):
    paper = tmp_path / "paper"
    resource = tmp_path / "resource"
    out = tmp_path / "out"
    write_csv(
        paper / "final_main_table_by_seed.csv",
        [{"method": "full RGCN tuned", "dataset": "ACM", "seed": 12345, "DEE": ""}],
    )
    write_csv(
        resource / "hgb_resource_logged_runs.csv",
        [{"method": "full RGCN tuned", "dataset": "ACM", "seed": 12345, "DEE": ""}],
    )

    result = audit_next11_dee_consistency(paper_final=paper, resource_logged=resource, output=out)

    assert result["rows"][0]["status"] == "spectral_not_applicable"

