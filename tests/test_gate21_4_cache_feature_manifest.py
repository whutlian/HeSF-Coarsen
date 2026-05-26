from __future__ import annotations

import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cache_feature_dry_run_writes_feature_and_adapter_contract(tmp_path: Path) -> None:
    from experiments.scripts.run_gate21_4_cache_feature_validation import main

    out = tmp_path / "gate21_4_cache_feature"
    assert (
        main(
            [
                "--dataset",
                "DBLP",
                "--output-dir",
                str(out),
                "--base-graph",
                "APV-skeleton",
                "--graph-seeds",
                "1",
                "--training-seeds",
                "1",
                "--feature-transforms",
                "raw",
                "zero-paper",
                "--term-channel-specs",
                "PTTP00",
                "PTTP30",
                "--feature-compression-methods",
                "raw_features_adapter_control",
                "fp16_node_features",
                "--dry-run",
            ]
        )
        == 0
    )

    feature_rows = _read_rows(out / "gate21_4_feature_channel_ablation.csv")
    adapter_rows = _read_rows(out / "gate21_4_feature_cache_compression_results.csv")
    assert len(feature_rows) == 4
    assert len(adapter_rows) == 2
    assert all(row["official_sehgnn_unmodified"] == "False" for row in adapter_rows)
    assert all(row["eligible_for_main_decision"] == "False" for row in adapter_rows)
    manifest_rows = _read_rows(out / "gate21_4_run_manifest.csv")
    storage_rows = _read_rows(out / "gate21_4_storage_audit.csv")
    assert len(manifest_rows) == 6
    assert all(row["status"] == "planned" for row in manifest_rows)
    assert storage_rows == []
    with (out / "gate21_4_storage_audit.csv").open(newline="", encoding="utf-8") as handle:
        assert "effective_total_byte_ratio" in (csv.DictReader(handle).fieldnames or [])
    assert (out / "gate21_4_decision.json").exists()
