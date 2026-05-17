from pathlib import Path

from experiments.scripts._common import write_csv


def test_next12_paper_tables_include_metapath_columns_and_note(tmp_path: Path):
    from experiments.scripts.summarize_next12_paper_tables import summarize_next12_paper_tables

    rebuttal = tmp_path / "rebuttal"
    external = tmp_path / "external"
    metapath = tmp_path / "metapath"
    output = tmp_path / "out"
    write_csv(
        rebuttal / "paper_rebuttal_table_aggregate.csv",
        [
            {
                "method": "HeSF-LVC-P",
                "cumulative_dee_or_audited_dee_mean": 0.02,
                "relation_energy_error_mean_mean": 0.03,
                "relation_js_drift_mean_mean": 0.01,
                "best_macro_f1_mean": 0.74,
            },
            {
                "method": "flatten-sum",
                "cumulative_dee_or_audited_dee_mean": 0.18,
                "relation_energy_error_mean_mean": 0.19,
                "relation_js_drift_mean_mean": 0.04,
                "best_macro_f1_mean": 0.75,
            },
        ],
    )
    write_csv(external / "external_baseline_by_method.csv", [{"method": "AH-UGC-style", "best_mean": 0.70}])
    write_csv(
        metapath / "metapath_retention_by_method.csv",
        [
            {
                "method": "HeSF-LVC-P",
                "typed_exact_step_survival_rate_mean": 0.8,
                "schema_path_survival_gap_mean": 0.1,
                "endpoint_pair_collapse_rate_mean": 0.2,
                "log_path_count_error_mean": 0.3,
            },
            {
                "method": "flatten-sum",
                "typed_exact_step_survival_rate_mean": 0.4,
                "schema_path_survival_gap_mean": 0.5,
                "endpoint_pair_collapse_rate_mean": 0.6,
                "log_path_count_error_mean": 0.7,
            },
        ],
    )
    (metapath / "summary.md").write_text("diagnostic enough for main text\n", encoding="utf-8")

    summarize_next12_paper_tables(rebuttal=rebuttal, external=external, metapath=metapath, output=output)

    table2 = (output / "table2_flatten_h6_rebuttal_with_metapath.csv").read_text(encoding="utf-8")
    assert "typed_exact_step_survival_rate" in table2
    assert "survival_gap_untyped_minus_typed" in table2
    assert "paper_final_dee" in table2
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "diagnostic enough" in summary
