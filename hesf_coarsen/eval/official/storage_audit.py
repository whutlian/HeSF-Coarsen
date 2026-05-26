from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HGB_COMPONENT_FILES = {
    "node.dat": "node_dat_bytes",
    "link.dat": "link_dat_bytes",
    "label.dat": "label_dat_bytes",
    "label.dat.test": "label_test_dat_bytes",
    "info.dat": "info_dat_bytes",
}

STORAGE_AUDIT_FIELDS = [
    "dataset",
    "seed",
    "method",
    "method_family",
    "semantic_structural_storage_ratio",
    "hgb_raw_file_byte_ratio",
    "preprocessed_cache_byte_ratio",
    "support_node_ratio",
    "support_edge_ratio",
    "total_node_ratio",
    "total_edge_ratio",
    "node_dat_bytes",
    "link_dat_bytes",
    "label_dat_bytes",
    "label_test_dat_bytes",
    "info_dat_bytes",
    "metadata_sidecar_bytes",
    "feature_bytes_estimated",
    "edge_bytes_estimated",
    "label_bytes_estimated",
    "export_total_bytes",
    "native_full_total_bytes",
    "export_preprocessed_cache_bytes",
    "native_full_preprocessed_cache_bytes",
    "structural_storage_budget_pass",
    "raw_hgb_byte_budget_pass",
    "cache_byte_budget_pass",
]


@dataclass(frozen=True)
class StorageBreakdown:
    dataset: str
    method: str
    seed: int | None
    semantic_structural_storage_ratio: float | None
    support_node_ratio: float | None
    support_edge_ratio: float | None
    total_node_ratio: float | None
    total_edge_ratio: float | None
    hgb_raw_file_byte_ratio: float | None
    export_total_bytes: int | None
    native_full_total_bytes: int | None
    node_dat_bytes: int | None
    link_dat_bytes: int | None
    label_dat_bytes: int | None
    label_test_dat_bytes: int | None
    info_dat_bytes: int | None
    metadata_sidecar_bytes: int | None
    preprocessed_cache_byte_ratio: float | None
    export_preprocessed_cache_bytes: int | None
    native_full_preprocessed_cache_bytes: int | None
    feature_bytes_estimated: int | None
    edge_bytes_estimated: int | None
    label_bytes_estimated: int | None
    structural_storage_budget_pass: bool | None
    raw_hgb_byte_budget_pass: bool | None
    cache_byte_budget_pass: bool | None

    def to_row(self, *, method_family: str = "") -> dict[str, Any]:
        row = asdict(self)
        row["method_family"] = method_family
        return {field: row.get(field, "") for field in STORAGE_AUDIT_FIELDS}


def _iter_files(path: Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.is_file()]


def _dir_size(path: Path) -> int:
    return int(sum(p.stat().st_size for p in _iter_files(path)))


def _component_bytes(path: Path) -> dict[str, int]:
    values = {field: 0 for field in HGB_COMPONENT_FILES.values()}
    for file_path in _iter_files(path):
        field = HGB_COMPONENT_FILES.get(file_path.name)
        if field is not None:
            values[field] += int(file_path.stat().st_size)
    return values


def _count_link_lines(path: Path) -> int:
    total = 0
    for link_path in [p for p in _iter_files(path) if p.name == "link.dat"]:
        with link_path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return int(total)


def _estimate_feature_bytes(path: Path) -> int | None:
    node_paths = [p for p in _iter_files(path) if p.name == "node.dat"]
    if not node_paths:
        return None
    values = 0
    saw_features = False
    for node_path in node_paths:
        with node_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3]:
                    saw_features = True
                    values += len(parts[3].split(",")) * 4
    return int(values) if saw_features else None


def _estimate_label_bytes(path: Path) -> int | None:
    label_paths = [p for p in _iter_files(path) if p.name in {"label.dat", "label.dat.test"}]
    if not label_paths:
        return None
    count = 0
    for label_path in label_paths:
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 8
    return int(count)


def _cache_bytes(path: Path | None) -> int | None:
    if path is None or not Path(path).exists():
        return None
    return _dir_size(Path(path))


def _pass(value: float | None, budget: float | None) -> bool | None:
    if value is None or budget is None:
        return None
    return bool(float(value) <= float(budget))


def audit_hgb_directory(
    *,
    dataset: str,
    method: str,
    export_dir: Path,
    native_full_dir: Path,
    seed: int | None = None,
    semantic_structural_storage_ratio: float | None = None,
    support_node_ratio: float | None = None,
    support_edge_ratio: float | None = None,
    total_node_ratio: float | None = None,
    total_edge_ratio: float | None = None,
    structural_budget: float | None = None,
    raw_byte_budget: float | None = None,
    cache_dir: Path | None = None,
    native_full_cache_dir: Path | None = None,
) -> StorageBreakdown:
    export_dir = Path(export_dir)
    native_full_dir = Path(native_full_dir)
    export_total = _dir_size(export_dir)
    native_total = _dir_size(native_full_dir)
    if native_total <= 0:
        raise ValueError(f"native full HGB byte size must be positive: {native_full_dir}")
    components = _component_bytes(export_dir)
    component_total = int(sum(components.values()))
    export_cache = _cache_bytes(cache_dir)
    native_cache = _cache_bytes(native_full_cache_dir)
    cache_ratio = None
    if export_cache is not None and native_cache is not None and native_cache > 0:
        cache_ratio = float(export_cache / native_cache)
    return StorageBreakdown(
        dataset=str(dataset),
        method=str(method),
        seed=None if seed is None else int(seed),
        semantic_structural_storage_ratio=semantic_structural_storage_ratio,
        support_node_ratio=support_node_ratio,
        support_edge_ratio=support_edge_ratio,
        total_node_ratio=total_node_ratio,
        total_edge_ratio=total_edge_ratio,
        hgb_raw_file_byte_ratio=float(export_total / native_total),
        export_total_bytes=int(export_total),
        native_full_total_bytes=int(native_total),
        node_dat_bytes=int(components["node_dat_bytes"]),
        link_dat_bytes=int(components["link_dat_bytes"]),
        label_dat_bytes=int(components["label_dat_bytes"]),
        label_test_dat_bytes=int(components["label_test_dat_bytes"]),
        info_dat_bytes=int(components["info_dat_bytes"]),
        metadata_sidecar_bytes=max(0, int(export_total - component_total)),
        preprocessed_cache_byte_ratio=cache_ratio,
        export_preprocessed_cache_bytes=export_cache,
        native_full_preprocessed_cache_bytes=native_cache,
        feature_bytes_estimated=_estimate_feature_bytes(export_dir),
        edge_bytes_estimated=int(_count_link_lines(export_dir) * 24),
        label_bytes_estimated=_estimate_label_bytes(export_dir),
        structural_storage_budget_pass=_pass(semantic_structural_storage_ratio, structural_budget),
        raw_hgb_byte_budget_pass=_pass(float(export_total / native_total), raw_byte_budget),
        cache_byte_budget_pass=_pass(cache_ratio, raw_byte_budget),
    )
