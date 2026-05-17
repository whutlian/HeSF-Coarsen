from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from experiments.scripts._common import write_csv
from experiments.scripts.summarize_next9_hgb_paper_final import _plot_scatter


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value, None)
    if number is None:
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def mean_value(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in (as_float(value, None) for value in values) if value is not None]
    return None if not clean else float(mean(clean))


def std_value(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in (as_float(value, None) for value in values) if value is not None]
    return None if len(clean) <= 1 else float(pstdev(clean))


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items()):
        item = {key: key_values[index] for index, key in enumerate(keys)}
        item["run_count"] = len(group)
        for metric in metrics:
            values = [row.get(metric, "") for row in group]
            item[f"{metric}_mean"] = mean_value(values) if any(value not in {None, ""} for value in values) else ""
            item[f"{metric}_std"] = std_value(values) if any(value not in {None, ""} for value in values) else ""
            item[f"{metric}_max"] = max(
                [float(value) for value in (as_float(value, None) for value in values) if value is not None],
                default="",
            )
        out.append(item)
    return out


def write_png(path: str | Path, rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> None:
    _plot_scatter(rows, x_key, y_key, Path(path))


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array_payload(path: str | Path, key: str = "assignment") -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        import numpy as np

        payload = np.load(path)
        arr = payload[key]
    except Exception:
        return ""
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, list(rows))

