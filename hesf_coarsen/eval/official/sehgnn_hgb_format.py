from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SEHGNN_HGB_SCHEMAS: dict[str, dict[str, Any]] = {
    "DBLP": {
        "target_type": "A",
        "node_type_order": {"A": 0, "P": 1, "T": 2, "V": 3},
        "relation_id_order": {"AP": 0, "PA": 1, "PT": 2, "PV": 3, "TP": 4, "VP": 5},
        "is_multilabel": False,
    },
    "ACM": {
        "target_type": "P",
        "node_type_order": {"P": 0, "A": 1, "C": 2, "K": 3},
        "relation_id_order": {"PP": 0, "PP_r": 1, "PA": 2, "AP": 3, "PC": 4, "CP": 5, "PK": 6, "KP": 7},
        "is_multilabel": False,
    },
    "IMDB": {
        "target_type": "M",
        "node_type_order": {"M": 0, "D": 1, "A": 2, "K": 3},
        "relation_id_order": {"MD": 0, "DM": 1, "MA": 2, "AM": 3, "MK": 4, "KM": 5},
        "is_multilabel": True,
    },
}


def supported_sehgnn_hgb_dataset(dataset: str) -> str:
    name = str(dataset).upper()
    if name not in SEHGNN_HGB_SCHEMAS:
        raise ValueError(f"unsupported official SeHGNN HGB dataset: {dataset}")
    return name


def _dataset_dir(data_root: Path, dataset: str) -> Path:
    root = Path(data_root)
    if root.name.upper() == dataset:
        return root
    return root / dataset


def _file_list_hash(dataset_dir: Path) -> str:
    if not dataset_dir.exists():
        return ""
    digest = hashlib.sha256()
    for path in sorted(p for p in dataset_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(dataset_dir).as_posix()
        stat = path.stat()
        digest.update(f"{rel}\t{stat.st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _parse_node_dat(path: Path) -> tuple[dict[str, int], dict[str, list[int]]]:
    counts: Counter[str] = Counter()
    feature_dims: dict[str, list[int]] = {}
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) not in {3, 4}:
                continue
            node_type = str(int(parts[2]))
            counts[node_type] += 1
            if len(parts) == 4 and parts[3]:
                feature_dims.setdefault(node_type, []).append(len(parts[3].split(",")))
    return dict(counts), feature_dims


def _parse_link_dat(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                counts[str(int(parts[2]))] += 1
    return dict(counts)


def _parse_label_dat(path: Path) -> tuple[int, int, str]:
    count = 0
    max_class = -1
    label_shape = ""
    if not path.exists():
        return 0, 0, ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            labels = [int(value) for value in parts[3].split(",") if value != ""]
            count += 1
            if labels:
                max_class = max(max_class, max(labels))
                label_shape = "multi" if len(labels) > 1 else label_shape or "single"
    return count, max_class + 1 if max_class >= 0 else 0, label_shape


def _official_loader_can_load(dataset_dir: Path, sehgnn_repo_dir: Path | None) -> tuple[bool, str]:
    required = [dataset_dir / name for name in ("node.dat", "link.dat", "label.dat", "label.dat.test")]
    if any(not path.exists() for path in required):
        return False, "missing required HGB dat files"
    if sehgnn_repo_dir is None:
        return False, "official SeHGNN repo not supplied"
    data_dir = Path(sehgnn_repo_dir) / "data"
    if not (data_dir / "data_loader.py").exists():
        return False, "missing official data_loader.py"
    sys.path.insert(0, str(data_dir))
    try:
        from data_loader import data_loader  # type: ignore

        data_loader(str(dataset_dir))
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on external official loader.
        return False, str(exc)
    finally:
        try:
            sys.path.remove(str(data_dir))
        except ValueError:
            pass


def audit_native_hgb_data_dir(dataset: str, data_root: Path, sehgnn_repo_dir: Path | None = None) -> dict[str, Any]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    dataset_dir = _dataset_dir(Path(data_root), dataset_name)
    node_dat = dataset_dir / "node.dat"
    link_dat = dataset_dir / "link.dat"
    label_dat = dataset_dir / "label.dat"
    label_dat_test = dataset_dir / "label.dat.test"
    node_counts, feature_dims = _parse_node_dat(node_dat)
    edge_counts = _parse_link_dat(link_dat)
    trainval_count, train_classes, train_label_shape = _parse_label_dat(label_dat)
    test_count, test_classes, test_label_shape = _parse_label_dat(label_dat_test)
    can_load, load_error = _official_loader_can_load(dataset_dir, sehgnn_repo_dir)
    schema = SEHGNN_HGB_SCHEMAS[dataset_name]
    return {
        "dataset": dataset_name,
        "data_root": str(Path(data_root)),
        "dataset_dir": str(dataset_dir),
        "file_list_hash": _file_list_hash(dataset_dir),
        "node_dat_exists": node_dat.exists(),
        "link_dat_exists": link_dat.exists(),
        "label_dat_exists": label_dat.exists(),
        "label_dat_test_exists": label_dat_test.exists(),
        "node_dat_line_count": _count_lines(node_dat),
        "link_dat_line_count": _count_lines(link_dat),
        "label_dat_line_count": _count_lines(label_dat),
        "label_dat_test_line_count": _count_lines(label_dat_test),
        "node_count_by_type": json.dumps(node_counts, sort_keys=True),
        "edge_count_by_relation": json.dumps(edge_counts, sort_keys=True),
        "target_type": schema["target_type"],
        "num_classes": max(int(train_classes), int(test_classes)),
        "trainval_count": int(trainval_count),
        "test_count": int(test_count),
        "label_shape": test_label_shape or train_label_shape,
        "feature_shapes_by_type": json.dumps(
            {key: sorted(set(value)) for key, value in feature_dims.items()},
            sort_keys=True,
        ),
        "can_load_with_official_data_loader": bool(can_load),
        "official_data_loader_error": load_error,
    }
