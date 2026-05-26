from __future__ import annotations

import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_gate21_5_directed_dry_run_writes_manifest_and_checklist(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_5_directed_apv_skeleton import main

    out = tmp_path / "gate21_5"
    assert (
        main(
            [
                "--dataset",
                "DBLP",
                "--out-dir",
                str(out),
                "--methods",
                "custom",
                "--custom-methods",
                "AP100-PA00-PV100-VP00-PTTP00",
                "--graph-seeds",
                "1",
                "2",
                "--training-seeds",
                "1",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )

    rows = _rows(out / "manifests" / "gate21_5_directed_run_manifest.csv")
    assert len(rows) == 2
    assert rows[0]["method"] == "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"
    assert rows[0]["deterministic_graph_method"] == "True"
    assert rows[0]["graph_seed_independence_status"] == "not_applicable_deterministic"
    assert (out / "gate21_5_requirement_checklist.md").exists()


def test_gate21_5_feature_adapter_dry_run_marks_adapter_boundaries(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_5_feature_adapter_validation import main

    out = tmp_path / "gate21_5"
    assert (
        main(
            [
                "--dataset",
                "DBLP",
                "--out-dir",
                str(out),
                "--base-graphs",
                "APV",
                "--adapters",
                "pca128,rp64",
                "--training-seeds",
                "1",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )

    rows = _rows(out / "gate21_5_feature_adapter_raw_rows.csv")
    assert len(rows) == 4
    assert all(row["official_sehgnn_unmodified"] == "False" for row in rows)
    assert all(row["eligible_for_main_decision"] == "False" for row in rows)
    assert all(row["eligible_for_adapter_table"] == "True" for row in rows)


def test_gate21_5_feature_channel_ablation_dry_run_writes_required_rows(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_5_feature_channel_ablation import main

    out = tmp_path / "gate21_5"
    assert (
        main(
            [
                "--dataset",
                "DBLP",
                "--out-dir",
                str(out),
                "--base-graphs",
                "APV",
                "--feature-transforms",
                "raw,zero-paper,pca-paper-128",
                "--term-channel-specs",
                "PTTP00",
                "--training-seeds",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    rows = _rows(out / "gate21_5_feature_channel_ablation.csv")
    assert len(rows) == 3
    assert {row["feature_transform_name"] for row in rows} == {"raw", "zero-paper", "pca-paper-128"}
    assert all(row["eligible_for_main_decision"] == "False" for row in rows)


def test_gate21_5_pathaware_prune_dry_run_writes_diagnostics(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_5_pathaware_prune_ap_pv import main

    out = tmp_path / "gate21_5"
    assert (
        main(
            [
                "--dataset",
                "DBLP",
                "--out-dir",
                str(out),
                "--graph-seeds",
                "1",
                "--training-seeds",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    raw = _rows(out / "gate21_5_ap_pv_pruning_raw_rows.csv")
    assert len(raw) == 40
    assert (out / "gate21_5_edge_score_diagnostics.csv").exists()
    assert (out / "gate21_5_coverage_diagnostics.csv").exists()
