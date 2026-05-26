from __future__ import annotations

import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_gate21_6_top_level_dry_run_writes_required_plans(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_6_icde_ready import main

    out = tmp_path / "gate21_6"
    assert main(["--datasets", "DBLP", "--quick", "--dry-run", "--output-dir", str(out)]) == 0

    assert (out / "planned_runs.csv").exists()
    assert (out / "planned_methods.json").exists()
    assert (out / "planned_artifacts.json").exists()
    rows = _rows(out / "planned_runs.csv")
    assert {"export-full-SeHGNN", "H6-node30", "HeSF-RCS-APV12", "Random-HG-TP"}.issubset({row["method"] for row in rows})


def test_gate21_6_summarizer_writes_required_tables_from_minimal_inputs(tmp_path: Path) -> None:
    from experiments.scripts.summarize_gate21_6_icde_ready import main
    from hesf_coarsen.eval.official.runner_utils import write_csv

    out = tmp_path / "gate21_6"
    out.mkdir()
    write_csv(
        out / "gate21_6_directed_skeleton_by_method.csv",
        [
            {
                "dataset": "DBLP",
                "method": "HeSF-RCS-APV12",
                "method_family": "schema_preserving_rcs",
                "schema_compatible": True,
                "official_sehgnn_unmodified": True,
                "uses_feature_adapter": False,
                "uses_weighted_superedges": False,
                "keeps_all_target_nodes": True,
                "eligible_for_official_main_table": True,
                "structural_storage_ratio": 0.1195,
                "raw_hgb_text_byte_ratio": 0.5300,
                "test_micro_mean": 0.9448,
                "test_macro_mean": 0.9405,
            }
        ],
    )

    assert main(["--results-dir", str(out)]) == 0
    for name in [
        "gate21_6_decision.json",
        "gate21_6_decision.md",
        "gate21_6_main_table_official.csv",
        "gate21_6_adapter_table.csv",
        "gate21_6_external_tp_table.csv",
        "gate21_6_storage_system_table.csv",
        "gate21_6_ablation_table.csv",
    ]:
        assert (out / name).exists()
