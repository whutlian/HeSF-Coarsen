from __future__ import annotations

import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable

try:
    import resource
except ModuleNotFoundError:
    resource = None


RESOURCE_FIELDS = [
    "stage_name",
    "wall_time_seconds",
    "cpu_time_seconds",
    "peak_rss_mb",
    "peak_vms_mb",
    "peak_gpu_memory_mb",
    "input_bytes_read",
    "output_bytes_written",
    "intermediate_bytes_written",
    "num_files_read",
    "num_files_written",
    "num_edge_passes",
    "num_feature_passes",
    "io_bytes_estimated",
]


def _files(paths: Iterable[str | Path] | None) -> list[Path]:
    found: list[Path] = []
    for path in paths or []:
        p = Path(path)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(item for item in p.rglob("*") if item.is_file())
    return found


def _total_bytes(paths: Iterable[str | Path] | None) -> tuple[int, int]:
    files = _files(paths)
    return int(sum(file.stat().st_size for file in files)), len(files)


def _rss_mb() -> float:
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(getattr(usage, "ru_maxrss", 0.0))
    if os.name == "nt":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def conservative_resource_row(
    *,
    stage_name: str,
    input_paths: Iterable[str | Path] | None = None,
    output_paths: Iterable[str | Path] | None = None,
    intermediate_paths: Iterable[str | Path] | None = None,
    wall_time_seconds: float | None = None,
    cpu_time_seconds: float | None = None,
    peak_gpu_memory_mb: float | None = None,
    num_edge_passes: int | None = None,
    num_feature_passes: int | None = None,
) -> dict[str, Any]:
    input_bytes, files_read = _total_bytes(input_paths)
    output_bytes, files_written = _total_bytes(output_paths)
    intermediate_bytes, intermediate_files = _total_bytes(intermediate_paths)
    return {
        "stage_name": str(stage_name),
        "wall_time_seconds": None if wall_time_seconds is None else float(wall_time_seconds),
        "cpu_time_seconds": None if cpu_time_seconds is None else float(cpu_time_seconds),
        "peak_rss_mb": _rss_mb(),
        "peak_vms_mb": None,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "input_bytes_read": int(input_bytes),
        "output_bytes_written": int(output_bytes),
        "intermediate_bytes_written": int(intermediate_bytes),
        "num_files_read": int(files_read),
        "num_files_written": int(files_written + intermediate_files),
        "num_edge_passes": num_edge_passes,
        "num_feature_passes": num_feature_passes,
        "io_bytes_estimated": True,
    }


class SystemResourceLogger(AbstractContextManager["SystemResourceLogger"]):
    def __init__(
        self,
        stage_name: str,
        *,
        input_paths: Iterable[str | Path] | None = None,
        output_paths: Iterable[str | Path] | None = None,
        intermediate_paths: Iterable[str | Path] | None = None,
        num_edge_passes: int | None = None,
        num_feature_passes: int | None = None,
    ) -> None:
        self.stage_name = str(stage_name)
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.intermediate_paths = intermediate_paths
        self.num_edge_passes = num_edge_passes
        self.num_feature_passes = num_feature_passes
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self.row: dict[str, Any] | None = None

    def __enter__(self) -> "SystemResourceLogger":
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.row = conservative_resource_row(
            stage_name=self.stage_name,
            input_paths=self.input_paths,
            output_paths=self.output_paths,
            intermediate_paths=self.intermediate_paths,
            wall_time_seconds=time.perf_counter() - self._start_wall,
            cpu_time_seconds=time.process_time() - self._start_cpu,
            num_edge_passes=self.num_edge_passes,
            num_feature_passes=self.num_feature_passes,
        )
        return False
