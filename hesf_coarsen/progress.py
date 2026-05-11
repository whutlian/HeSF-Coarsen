from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from time import monotonic
from typing import TextIO, TypeVar

T = TypeVar("T")


def _progress_config(config: dict | None) -> dict:
    return {} if config is None else dict(config.get("progress", {}))


def progress_enabled(config: dict | None) -> bool:
    return bool(_progress_config(config).get("enabled", False))


def progress_message(config: dict | None, message: str, stream: TextIO | None = None) -> None:
    if not progress_enabled(config):
        return
    target = sys.stderr if stream is None else stream
    print(f"[hesf] {message}", file=target, flush=True)


def _plain_progress_iter(
    iterable: Iterable[T],
    total: int | None,
    desc: str,
    unit: str,
    min_interval: float,
    stream: TextIO,
) -> Iterator[T]:
    count = 0
    started = monotonic()
    last = started - min_interval
    label = desc or "progress"
    suffix_unit = f" {unit}" if unit else ""
    if total is not None:
        print(f"[hesf] {label}: 0/{total}{suffix_unit}", file=stream, flush=True)
    else:
        print(f"[hesf] {label}: 0{suffix_unit}", file=stream, flush=True)
    for item in iterable:
        yield item
        count += 1
        now = monotonic()
        if now - last >= min_interval:
            last = now
            elapsed = max(now - started, 1e-9)
            rate = count / elapsed
            if total is None:
                print(f"[hesf] {label}: {count}{suffix_unit} ({rate:.1f}/s)", file=stream, flush=True)
            else:
                width = 24
                filled = int(width * min(count, total) / max(total, 1))
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"[hesf] {label}: [{bar}] {count}/{total}{suffix_unit} ({rate:.1f}/s)",
                    file=stream,
                    flush=True,
                )
    if total is None:
        print(f"[hesf] {label}: done {count}{suffix_unit}", file=stream, flush=True)
    else:
        print(f"[hesf] {label}: done {count}/{total}{suffix_unit}", file=stream, flush=True)


def progress_iter(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    config: dict | None = None,
    unit: str = "",
    stream: TextIO | None = None,
) -> Iterable[T]:
    """Wrap an iterable with optional stderr progress reporting.

    Progress is disabled unless ``config["progress"]["enabled"]`` is true.
    ``backend: auto`` uses tqdm when installed and otherwise falls back to
    log-friendly plain progress lines.
    """

    if not progress_enabled(config):
        return iterable

    progress_cfg = _progress_config(config)
    backend = str(progress_cfg.get("backend", "auto")).lower()
    min_interval = float(progress_cfg.get("min_interval_seconds", 1.0))
    target = sys.stderr if stream is None else stream

    if backend in {"auto", "tqdm"}:
        try:
            from tqdm.auto import tqdm

            return tqdm(
                iterable,
                total=total,
                desc=desc or None,
                unit=unit or "it",
                mininterval=min_interval,
                file=target,
                dynamic_ncols=True,
            )
        except Exception:
            if backend == "tqdm":
                progress_message(config, "tqdm unavailable; using plain progress", stream=target)

    return _plain_progress_iter(iterable, total, desc, unit, min_interval, target)
