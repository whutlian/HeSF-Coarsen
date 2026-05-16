from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


def _rss_gb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**3))
    except Exception:
        return None


def _cuda_memory_gb() -> dict[str, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"peak_vram_allocated_gb": None, "peak_vram_reserved_gb": None}
        return {
            "peak_vram_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
            "peak_vram_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024**3)),
        }
    except Exception:
        return {"peak_vram_allocated_gb": None, "peak_vram_reserved_gb": None}


def snapshot_resources() -> dict[str, float | None]:
    out: dict[str, float | None] = {"peak_rss_gb": _rss_gb()}
    out.update(_cuda_memory_gb())
    return out


@dataclass
class ResourceMonitor:
    start_time: float = field(default_factory=perf_counter)
    peak_rss_gb: float | None = None

    def sample(self) -> dict[str, float | None]:
        snapshot = snapshot_resources()
        rss = snapshot.get("peak_rss_gb")
        if rss is not None:
            self.peak_rss_gb = rss if self.peak_rss_gb is None else max(self.peak_rss_gb, rss)
        snapshot["elapsed_sec"] = float(perf_counter() - self.start_time)
        snapshot["peak_rss_gb"] = self.peak_rss_gb
        return snapshot


def attach_peak_resource_fields(row: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    for key in ("peak_rss_gb", "peak_vram_allocated_gb", "peak_vram_reserved_gb"):
        if row.get(key) in {None, ""} and resources.get(key) is not None:
            row[key] = resources[key]
    return row
