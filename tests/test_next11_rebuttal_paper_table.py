from pathlib import Path

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_next11_rebuttal_paper_table import summarize_next11_rebuttal_paper_table


def _write_rebuttal_inputs(root: Path) -> None:
    methods = ["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best"]
    write_csv(
        root / "paper_rebuttal_table.csv",
        [
            {
                "method": method,
                "dataset": "ACM",
                "DEE_mean": 0.02 if method.startswith("HeSF") else 0.2,
                "relation_mass_l1_drift_mean": 0.1,
                "relation_mass_js_drift_mean": 0.01 if method.startswith("HeSF") else 0.05,
                "relation_energy_error_mean": 0.02 if method.startswith("HeSF") else 0.2,
                "duplicate_collapse_ratio_mean": 0.03 if method.startswith("HeSF") else 0.3,
                "projected_macro_f1_mean": 0.7,
                "refined_macro_f1@5_mean": 0.75,
                "best_macro_f1_mean": 0.76,
            }
            for method in methods
        ],
    )
    write_csv(
        root / "self_loop_share_by_relation.csv",
        [{"method": method, "dataset": "ACM", "relation_id": 0, "self_loop_share_mean": 0.01} for method in methods],
    )
    write_csv(
        root / "duplicate_collapse_ratio_by_relation.csv",
        [{"method": method, "dataset": "ACM", "relation_id": 0, "duplicate_collapse_ratio_mean": 0.02} for method in methods],
    )
    write_csv(
        root / "checkpoint_refine_masking_curve.csv",
        [
            {"method": method, "dataset": "ACM", "checkpoint": checkpoint, "macro_f1_mean": 0.7}
            for method in methods
            for checkpoint in ("projected", "refined@0", "refined@1", "refined@3", "refined@5", "best")
        ],
    )


def test_rebuttal_paper_table_includes_all_methods_and_deltas(tmp_path: Path):
    rebuttal = tmp_path / "rebuttal"
    out = tmp_path / "out"
    _write_rebuttal_inputs(rebuttal)

    summarize_next11_rebuttal_paper_table(rebuttal=rebuttal, paper_final=tmp_path / "paper", dee_consistency=tmp_path / "dee", output=out)

    aggregate = (out / "paper_rebuttal_table_aggregate.csv").read_text(encoding="utf-8")
    for method in ["HeSF-LVC-P", "HeSF-LVC-S", "flatten-sum", "H6-no-spec", "H0-mutual-best"]:
        assert method in aggregate
    assert (out / "paper_rebuttal_delta_vs_flatten_sum.csv").exists()
    assert "full RGCN" not in (out / "paper_rebuttal_delta_vs_flatten_sum.csv").read_text(encoding="utf-8")
    for figure in [
        "figures/relation_js_drift_by_dataset.png",
        "figures/relation_energy_error_by_dataset.png",
        "figures/edge_collapse_by_dataset.png",
        "figures/checkpoint_refine_masking_curve.png",
    ]:
        assert (out / figure).exists(), figure
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Paper-safe claim" in summary
    assert "Not-supported claim" in summary

