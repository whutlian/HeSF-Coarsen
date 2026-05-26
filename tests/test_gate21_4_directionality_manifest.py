from __future__ import annotations

import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_apv_dry_run_manifest_contains_canonical_method_and_cache_dir(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_4_apv_skeleton_validation import main

    out = tmp_path / "gate21_4"
    assert (
        main(
            [
                "--dataset",
                "DBLP",
                "--output-dir",
                str(out),
                "--graph-seeds",
                "1",
                "2",
                "--training-seeds",
                "1",
                "--methods",
                "APV-skeleton",
                "PTTP10",
                "--dry-run",
            ]
        )
        == 0
    )

    rows = _rows(out / "gate21_4_run_manifest.csv")
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"H6-APV-skeleton", "H6-relgrid-APPA100-PVVP100-PTTP10"}
    assert all(row["canonical_method"] for row in rows)
    assert all("graph_seed_" in row["cache_dir"] for row in rows)
    assert all(row["status"] == "planned" for row in rows)


def test_directionality_dry_run_writes_required_output_contract(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_4_apv_skeleton_validation import main

    out = tmp_path / "gate21_4_directionality"
    main(
        [
            "--dataset",
            "DBLP",
            "--output-dir",
            str(out),
            "--graph-seeds",
            "1",
            "--training-seeds",
            "1",
            "--methods",
            "directionality",
            "--dry-run",
        ]
    )

    assert (out / "gate21_4_plan.json").exists()
    assert (out / "gate21_4_directionality_ablation.csv").exists()
    assert (out / "gate21_4_directionality_plan.json").exists()
    assert (out / "gate21_4_directionality_run_manifest.csv").exists()
    assert (out / "gate21_4_directionality_summary.md").exists()
    assert (out / "gate21_4_cache_audit.csv").exists()
