from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheNamespace:
    dataset: str
    method: str
    graph_seed: int | None
    training_seed: int
    export_hash: str
    root_dir: Path

    @property
    def cache_dir(self) -> Path:
        graph = "graph_seed_none" if self.graph_seed is None else f"graph_seed_{int(self.graph_seed)}"
        train = f"training_seed_{int(self.training_seed)}"
        token = _safe_token(self.method)
        export = f"export_hash_{str(self.export_hash)[:12]}"
        return Path(self.root_dir) / str(self.dataset).upper() / token / graph / train / export / "cache"


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_directory_sha256(path: Path, include_filenames: bool = True) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    if not root.exists():
        return ""
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        if include_filenames:
            digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
        digest.update(compute_file_sha256(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_export_file_list_hash(export_dir: Path) -> str:
    return compute_directory_sha256(Path(export_dir), include_filenames=True)


def prepare_unique_cache_dir(namespace: CacheNamespace, force_reprocess: bool) -> dict[str, Any]:
    cache_dir = namespace.cache_dir
    exists_before = cache_dir.exists()
    files_before = _file_count(cache_dir)
    hash_before = compute_directory_sha256(cache_dir) if exists_before else ""
    deleted = False
    if bool(force_reprocess) and exists_before:
        _assert_cache_path_under_root(cache_dir, Path(namespace.root_dir))
        shutil.rmtree(cache_dir)
        deleted = True
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "preprocess_cache_dir": str(cache_dir),
        "cache_dir_exists_before_run": exists_before,
        "cache_dir_deleted_before_run": deleted,
        "force_reprocess_flag": bool(force_reprocess),
        "unique_cache_namespace_flag": True,
        "cache_files_count_before": files_before,
        "cache_hash_before": hash_before,
    }


def collect_cache_audit_before_after(cache_dir: Path, before: dict[str, Any] | None = None) -> dict[str, Any]:
    before = dict(before or {})
    cache_dir = Path(cache_dir)
    files_after = _file_count(cache_dir)
    hash_after = compute_directory_sha256(cache_dir) if cache_dir.exists() else ""
    reused = bool(before.get("cache_dir_exists_before_run")) and not bool(before.get("cache_dir_deleted_before_run"))
    pass_flag = bool(before.get("unique_cache_namespace_flag", False)) and not reused
    return {
        **before,
        "cache_files_count_after": files_after,
        "cache_hash_after": hash_after,
        "cache_reused_flag": reused,
        "cache_hygiene_pass": pass_flag,
        "sehgnn_generated_feature_cache_keys": "",
        "sehgnn_generated_label_cache_keys": "",
        "sehgnn_generated_metapath_keys": "",
        "notes": "unique namespace prepared; DBLP official path has no reusable Freebase adjacency cache",
    }


def file_hashes_for_export(export_dir: Path) -> dict[str, str]:
    root = Path(export_dir)
    return {
        "export_file_list_hash": compute_export_file_list_hash(root),
        "node_dat_hash": _maybe_hash(root / "node.dat"),
        "link_dat_hash": _maybe_hash(root / "link.dat"),
        "label_dat_hash": _maybe_hash(root / "label.dat"),
        "label_test_dat_hash": _maybe_hash(root / "label.dat.test"),
        "info_dat_hash": _maybe_hash(root / "info.dat"),
    }


def _maybe_hash(path: Path) -> str:
    return compute_file_sha256(path) if Path(path).exists() else ""


def _file_count(path: Path) -> int:
    return sum(1 for item in Path(path).rglob("*") if item.is_file()) if Path(path).exists() else 0


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _assert_cache_path_under_root(cache_dir: Path, root_dir: Path) -> None:
    cache_full = Path(cache_dir).resolve()
    root_full = Path(root_dir).resolve()
    if not str(cache_full).lower().startswith(str(root_full).lower()):
        raise ValueError(f"refusing to remove cache outside root: {cache_full}")
