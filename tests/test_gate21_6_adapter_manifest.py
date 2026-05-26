from __future__ import annotations

import json
from pathlib import Path


def test_adapter_manifest_package_bytes_equal_sum_of_included_artifacts(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.adapter_package_manifest import build_adapter_manifest

    artifacts = {
        "link_dat": tmp_path / "link.dat",
        "node_id_mapping": tmp_path / "node_id_mapping.json",
        "loader_config": tmp_path / "loader_config.json",
    }
    artifacts["link_dat"].write_bytes(b"src dst\n0 1\n")
    artifacts["node_id_mapping"].write_bytes(b'{"author": [0, 1]}')
    artifacts["loader_config"].write_bytes(b'{"dataset": "DBLP"}')

    manifest = build_adapter_manifest(
        dataset="DBLP",
        method="HeSF-RCS-APV12+random_projection_dim64",
        graph_export_hash="abc123",
        included_artifacts=artifacts,
        excluded_bytes_with_reason=[
            {"artifact": "preprocessed_cache", "reason": "not required for deployable adapter package", "bytes": 0}
        ],
    )

    expected_total = sum(path.stat().st_size for path in artifacts.values())
    assert manifest["adapter_package_total_bytes"] == expected_total
    assert manifest["link_dat_bytes"] == artifacts["link_dat"].stat().st_size
    assert manifest["node_id_mapping_bytes"] == artifacts["node_id_mapping"].stat().st_size
    assert manifest["loader_config_bytes"] == artifacts["loader_config"].stat().st_size


def test_adapter_manifest_writes_required_gate21_6_fields(tmp_path: Path) -> None:
    from hesf_coarsen.eval.official.adapter_package_manifest import (
        GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS,
        build_adapter_manifest,
        write_adapter_manifest,
    )

    sidecar = tmp_path / "paper_features.npy"
    sidecar.write_bytes(b"feature-sidecar")
    manifest_path = tmp_path / "adapter_manifest.json"

    manifest = build_adapter_manifest(
        dataset="DBLP",
        method="HeSF-RCS-APV16+fp16_node_features",
        graph_export_hash="def456",
        included_artifacts={"sidecar_feature:paper": sidecar},
        excluded_bytes_with_reason=[{"artifact": "pca_basis", "reason": "not used by fp16 adapter", "bytes": 0}],
    )

    written = write_adapter_manifest(manifest, manifest_path)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert written == manifest_path
    assert set(GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS).issubset(loaded)
    assert loaded["sidecar_feature_bytes_total"] == sidecar.stat().st_size
    assert loaded["sidecar_feature_bytes_by_node_type"] == {"paper": sidecar.stat().st_size}
    assert loaded["excluded_bytes_with_reason"] == [
        {"artifact": "pca_basis", "reason": "not used by fp16 adapter", "bytes": 0}
    ]
    assert len(loaded["adapter_package_sha256"]) == 64
