from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Sequence

from hesf_coarsen.eval.official.freehgc_env_bridge import (
    FREEHGC_REPO_URL,
    build_freehgc_command,
    freehgc_preflight,
    missing_dependency_failure_row,
    resolve_freehgc_root,
    run_subprocess,
)


FREEHGC_STANDARD_PROTOCOL = "standard_condensation"
FREEHGC_TP_PROTOCOL = "schema_preserving_tp"


def run_freehgc_protocol_rows(
    *,
    dataset: str,
    ratios: Sequence[float],
    freehgc_root: str | Path | None,
    data_root: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    quick: bool = False,
    strict: bool = False,
    run_upstream: bool = True,
    timeout_seconds: int | None = None,
    python_executable: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = resolve_freehgc_root(freehgc_root)
    out = Path(output_dir)
    log_dir = out / "freehgc_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    preflight = freehgc_preflight(freehgc_root=root, python_executable=python_executable)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    methods = [
        ("FreeHGC", FREEHGC_STANDARD_PROTOCOL),
        ("FreeHGC-TP", FREEHGC_TP_PROTOCOL),
    ]
    selected_ratios = [float(ratios[0])] if quick and ratios else [float(ratio) for ratio in ratios]
    for method, protocol in methods:
        for ratio in selected_ratios:
            if preflight["missing_dependency"]:
                row = missing_dependency_failure_row(
                    dataset=dataset,
                    method=method,
                    protocol=protocol,
                    preflight=preflight,
                    reduction_rate=ratio,
                )
                rows.append(row)
                failures.append(row)
                continue
            if method == "FreeHGC-TP":
                row = _freehgc_tp_not_eligible_row(dataset=dataset, ratio=ratio, root=root)
                rows.append(row)
                failures.append(row)
                continue
            if not run_upstream:
                row = _not_executed_row(dataset=dataset, method=method, protocol=protocol, ratio=ratio, root=root)
                rows.append(row)
                failures.append(row)
                continue
            command = build_freehgc_command(
                freehgc_root=root,
                dataset=dataset,
                data_root=data_root,
                reduction_rate=ratio,
                seed=1,
                device=device,
                quick=quick,
                python_executable=python_executable,
            )
            log_stem = f"{str(dataset).upper()}_{method}_ratio_{str(ratio).replace('.', 'p')}"
            stdout_path = log_dir / f"{log_stem}.stdout"
            stderr_path = log_dir / f"{log_stem}.stderr"
            start = time.perf_counter()
            try:
                completed = run_subprocess(command, cwd=root / "HGB", timeout_seconds=timeout_seconds)
                wall = time.perf_counter() - start
                stdout_path.write_text(completed.stdout, encoding="utf-8", errors="ignore")
                stderr_path.write_text(completed.stderr, encoding="utf-8", errors="ignore")
                row = _row_from_completed(
                    dataset=dataset,
                    method=method,
                    protocol=protocol,
                    ratio=ratio,
                    root=root,
                    command=command,
                    returncode=int(completed.returncode),
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    wall_time_seconds=wall,
                )
            except subprocess_timeout() as exc:  # pragma: no cover - environment dependent.
                wall = time.perf_counter() - start
                stdout_path.write_text(getattr(exc, "stdout", "") or "", encoding="utf-8", errors="ignore")
                stderr_path.write_text(getattr(exc, "stderr", "") or str(exc), encoding="utf-8", errors="ignore")
                row = _runtime_failure_row(
                    dataset=dataset,
                    method=method,
                    protocol=protocol,
                    ratio=ratio,
                    root=root,
                    command=command,
                    failure_type="timeout",
                    failure_message=str(exc),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    wall_time_seconds=wall,
                )
            rows.append(row)
            if not bool(row.get("success")):
                failures.append(row)
            if strict and str(row.get("failure_type")) == "failed_oom":
                break
    return rows, failures


def subprocess_timeout() -> type[BaseException]:
    import subprocess

    return subprocess.TimeoutExpired


def parse_freehgc_metrics(stdout: str) -> dict[str, Any]:
    matches = list(
        re.finditer(
            r"Val\((?P<val_macro>[-+0-9.eE]+)\s+(?P<val_micro>[-+0-9.eE]+)\),\s*Tes\((?P<test_macro>[-+0-9.eE]+)\s+(?P<test_micro>[-+0-9.eE]+)\)",
            stdout,
        )
    )
    if not matches:
        return {}
    last = matches[-1]
    return {
        "validation_macro_f1": float(last.group("val_macro")),
        "validation_micro_f1": float(last.group("val_micro")),
        "test_macro_f1": float(last.group("test_macro")),
        "test_micro_f1": float(last.group("test_micro")),
    }


def _row_from_completed(
    *,
    dataset: str,
    method: str,
    protocol: str,
    ratio: float,
    root: Path,
    command: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    stdout_path: Path,
    stderr_path: Path,
    wall_time_seconds: float,
) -> dict[str, Any]:
    metrics = parse_freehgc_metrics(f"{stdout}\n{stderr}")
    if returncode == 0 and metrics:
        return {
            **_base_row(dataset=dataset, method=method, protocol=protocol, ratio=ratio, root=root, command=command),
            **metrics,
            "success": True,
            "success_count": 1,
            "training_executed": True,
            "failure_type": "",
            "failure_message": "",
            "train_time_seconds": wall_time_seconds,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    text = f"{stdout}\n{stderr}".lower()
    failure_type = "failed_oom" if "out of memory" in text or ("cuda" in text and "memory" in text) else "failed_runtime"
    if returncode == 0:
        failure_type = "failed_metric_parse"
    return _runtime_failure_row(
        dataset=dataset,
        method=method,
        protocol=protocol,
        ratio=ratio,
        root=root,
        command=command,
        failure_type=failure_type,
        failure_message=(stderr.strip() or stdout.strip())[:4000],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        wall_time_seconds=wall_time_seconds,
    )


def _base_row(
    *,
    dataset: str,
    method: str,
    protocol: str,
    ratio: float,
    root: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": str(dataset).upper(),
        "method": str(method),
        "baseline_name": str(method),
        "method_family": "external_freehgc",
        "protocol": str(protocol),
        "external_baseline": True,
        "freehgc_repo_url": FREEHGC_REPO_URL,
        "freehgc_root": str(root),
        "command": " ".join(str(item) for item in command),
        "support_node_ratio": float(ratio),
        "reduction_rate": float(ratio),
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": False,
        "eligible_for_tp_main_comparison": False,
        "uses_synthetic_nodes": False,
        "uses_weighted_edges": False,
        "requires_loader_adapter": True,
        "missing_dependency": False,
        "missing_dependency_name": "",
        "attempted_import": "",
        "expected_env": "conda:pytorch",
        "suggested_install_command": "",
    }


def _runtime_failure_row(
    *,
    dataset: str,
    method: str,
    protocol: str,
    ratio: float,
    root: Path,
    command: Sequence[str],
    failure_type: str,
    failure_message: str,
    stdout_path: Path,
    stderr_path: Path,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        **_base_row(dataset=dataset, method=method, protocol=protocol, ratio=ratio, root=root, command=command),
        "success": False,
        "success_count": 0,
        "training_executed": True,
        "failure_type": str(failure_type),
        "failure_message": str(failure_message),
        "train_time_seconds": wall_time_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "test_micro_f1": "",
        "test_macro_f1": "",
    }


def _not_executed_row(*, dataset: str, method: str, protocol: str, ratio: float, root: Path) -> dict[str, Any]:
    return {
        **_base_row(dataset=dataset, method=method, protocol=protocol, ratio=ratio, root=root, command=[]),
        "success": False,
        "success_count": 0,
        "training_executed": False,
        "failure_type": "not_executed",
        "failure_message": "FreeHGC upstream execution was disabled for this runner invocation.",
        "test_micro_f1": "",
        "test_macro_f1": "",
    }


def _freehgc_tp_not_eligible_row(*, dataset: str, ratio: float, root: Path) -> dict[str, Any]:
    return {
        **_base_row(dataset=dataset, method="FreeHGC-TP", protocol=FREEHGC_TP_PROTOCOL, ratio=ratio, root=root, command=[]),
        "success": False,
        "success_count": 0,
        "training_executed": False,
        "failure_type": "adapter_not_implemented",
        "failure_message": "Gate21.7 does not self-implement FreeHGC-TP; upstream FreeHGC is audited separately and TP export remains not READY.",
        "test_micro_f1": "",
        "test_macro_f1": "",
    }
