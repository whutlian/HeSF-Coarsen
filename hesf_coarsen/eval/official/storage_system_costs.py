from __future__ import annotations

import gzip
import time
from pathlib import Path
from typing import Any, Iterable

from hesf_coarsen.eval.official.storage_only_baselines import build_storage_only_row
from hesf_coarsen.eval.official.system_resource_logger import conservative_resource_row


def dir_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return int(sum(item.stat().st_size for item in root.rglob("*") if item.is_file()))


def gzip_bytes(path: str | Path) -> int:
    root = Path(path)
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += len(gzip.compress(item.read_bytes()))
    return int(total)


def hgb_counts(path: str | Path) -> dict[str, int]:
    root = Path(path)
    node_rows = 0
    feature_values = 0
    relation_edges = 0
    node_path = root / "node.dat"
    link_path = root / "link.dat"
    if node_path.exists():
        with node_path.open(encoding="utf-8") as handle:
            for line in handle:
                node_rows += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3]:
                    feature_values += len([item for item in parts[3].split(",") if item])
    if link_path.exists():
        with link_path.open(encoding="utf-8") as handle:
            relation_edges = sum(1 for _ in handle)
    return {"node_rows": node_rows, "feature_values": feature_values, "relation_edges": relation_edges}


def build_gate21_7_storage_rows(*, dataset: str, full_export_dir: str | Path, compressed_exports: Iterable[tuple[str, str | Path]] = ()) -> list[dict[str, Any]]:
    root = Path(full_export_dir)
    native = max(1, dir_bytes(root))
    counts = hgb_counts(root)
    relation_bytes = counts["relation_edges"] * 12
    fp16_feature_bytes = counts["feature_values"] * 2
    int8_feature_bytes = counts["feature_values"]
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    gz = gzip_bytes(root) if root.exists() else 0
    gzip_wall = time.perf_counter() - start
    rows.append(_row(dataset, "raw_hgb_text", native, native, root, raw_hgb_text_bytes=native, loader_supported=True))
    rows.append(
        _row(
            dataset,
            "gzip_hgb_text",
            native,
            gz,
            root,
            gzip_bytes_value=gz,
            loader_supported=False,
            notes="storage-only compression; official loader adapter not used in main table",
            write_time_seconds=gzip_wall,
        )
    )
    rows.append(_row(dataset, "binary_csr_relation_tables", native, relation_bytes, root, binary_relation_bytes=relation_bytes, loader_supported=False))
    rows.append(
        _row(
            dataset,
            "binary_csr_plus_fp16_features",
            native,
            relation_bytes + fp16_feature_bytes,
            root,
            binary_relation_bytes=relation_bytes,
            binary_feature_bytes=fp16_feature_bytes,
            loader_supported=False,
        )
    )
    rows.append(
        _row(
            dataset,
            "binary_csr_plus_int8_features",
            native,
            relation_bytes + int8_feature_bytes,
            root,
            binary_relation_bytes=relation_bytes,
            binary_feature_bytes=int8_feature_bytes,
            loader_supported=False,
        )
    )
    for name, export_dir in compressed_exports:
        disk = dir_bytes(export_dir)
        rows.append(
            {
                **_row(
                    dataset,
                    str(name),
                    native,
                    disk,
                    Path(export_dir),
                    raw_hgb_text_bytes=disk,
                    loader_supported=True,
                    notes="official text export row",
                ),
                "method": str(name),
            }
        )
    return rows


def build_gate21_7_system_resource_rows(*, output_root: str | Path, source_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    root = Path(output_root)
    inputs = list(source_paths)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    row = conservative_resource_row(
        stage_name="gate21_7_artifact_summarization",
        input_paths=inputs,
        output_paths=[root],
        num_edge_passes=1,
        num_feature_passes=1,
    )
    row["wall_time_seconds"] = float(time.perf_counter() - start_wall)
    row["cpu_time_seconds"] = float(time.process_time() - start_cpu)
    row["peak_cpu_rss_mb"] = row.get("peak_rss_mb")
    row["measurement_source"] = "local_resource_logger"
    return [
        row
    ]


def _row(
    dataset: str,
    name: str,
    native: int,
    total: int,
    path: Path,
    *,
    raw_hgb_text_bytes: int | None = None,
    gzip_bytes_value: int | None = None,
    binary_relation_bytes: int | None = None,
    binary_feature_bytes: int | None = None,
    loader_supported: bool | None,
    notes: str = "",
    write_time_seconds: float | None = None,
) -> dict[str, Any]:
    row = build_storage_only_row(
        dataset=dataset,
        artifact_name=name,
        native_full_text_bytes=native,
        total_artifact_bytes=total,
        changes_training_semantics=False,
        requires_loader_adapter=not bool(loader_supported),
        raw_hgb_text_bytes=raw_hgb_text_bytes,
        gzip_bytes=gzip_bytes_value,
        binary_relation_bytes=binary_relation_bytes,
        binary_feature_bytes=binary_feature_bytes,
        read_time_seconds=_timed_read(path),
        write_time_seconds=write_time_seconds,
        loader_supported=loader_supported,
        notes=notes,
    )
    row["disk_bytes"] = int(total)
    row["load_wall_time_seconds"] = row.get("read_time_seconds", "")
    row["system_cost_measured"] = bool((row.get("read_time_seconds") or 0) or (row.get("write_time_seconds") or 0))
    return row


def _timed_read(path: Path) -> float:
    if not path.exists():
        return 0.0
    start = time.perf_counter()
    for item in path.rglob("*"):
        if item.is_file():
            with item.open("rb") as handle:
                handle.read(1024)
    return float(time.perf_counter() - start)
