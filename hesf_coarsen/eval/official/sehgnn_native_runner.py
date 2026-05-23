from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from hesf_coarsen.eval.official.runner_utils import repo_commit_hash, write_csv, write_json
from hesf_coarsen.eval.official.sehgnn_hgb_format import audit_native_hgb_data_dir, supported_sehgnn_hgb_dataset


REPO_URL = "https://github.com/ICT-GIMLab/SeHGNN"
NATIVE_METRIC_FIELDS = (
    "dataset",
    "seed",
    "status",
    "command",
    "best_epoch",
    "validation_micro_f1",
    "validation_macro_f1",
    "test_micro_f1",
    "test_macro_f1",
    "test_accuracy_if_single_label",
    "is_multilabel",
    "loss_type",
    "train_time_sec",
    "peak_memory_mb",
    "stdout_path",
    "stderr_path",
    "error_message",
)


DATASET_COMMAND_ARGS: dict[str, list[str]] = {
    "DBLP": [
        "--epoch",
        "200",
        "--dataset",
        "DBLP",
        "--n-fp-layers",
        "2",
        "--n-task-layers",
        "3",
        "--num-hops",
        "2",
        "--num-label-hops",
        "4",
        "--label-feats",
        "--residual",
        "--hidden",
        "512",
        "--embed-size",
        "512",
        "--dropout",
        "0.5",
        "--input-drop",
        "0.5",
        "--amp",
    ],
    "ACM": [
        "--epoch",
        "200",
        "--dataset",
        "ACM",
        "--n-fp-layers",
        "2",
        "--n-task-layers",
        "1",
        "--num-hops",
        "4",
        "--num-label-hops",
        "4",
        "--label-feats",
        "--hidden",
        "512",
        "--embed-size",
        "512",
        "--dropout",
        "0.5",
        "--input-drop",
        "0.5",
        "--amp",
    ],
    "IMDB": [
        "--epoch",
        "200",
        "--dataset",
        "IMDB",
        "--n-fp-layers",
        "2",
        "--n-task-layers",
        "4",
        "--num-hops",
        "4",
        "--num-label-hops",
        "4",
        "--label-feats",
        "--hidden",
        "512",
        "--embed-size",
        "512",
        "--dropout",
        "0.5",
        "--input-drop",
        "0.",
        "--amp",
    ],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def native_subprocess_env(*, repo_dir: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    shim_dir = _repo_root() / "external_patches" / "sehgnn_sparse_tools_shim"
    existing = env.get("PYTHONPATH", "")
    parts = [str(shim_dir)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["SEHGNN_REPO_DIR"] = str(Path(repo_dir))
    return env


@dataclass(frozen=True)
class NativeCommand:
    command: list[str]
    cwd: Path
    dataset: str
    seed: int


def build_official_hgb_command(
    *,
    dataset: str,
    seed: int,
    repo_dir: Path,
    data_root: Path,
    device: str,
    python_executable: str = sys.executable,
) -> NativeCommand:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    command = [str(python_executable), "main.py", *DATASET_COMMAND_ARGS[dataset_name], "--root", str(Path(data_root).resolve()), "--seeds", str(int(seed))]
    if str(device).lower() == "cpu":
        command.append("--cpu")
    return NativeCommand(command=command, cwd=Path(repo_dir) / "hgb", dataset=dataset_name, seed=int(seed))


def parse_official_hgb_stdout(dataset: str, stdout: str) -> dict[str, Any]:
    dataset_name = supported_sehgnn_hgb_dataset(dataset)
    is_multilabel = dataset_name == "IMDB"
    best_match = re.search(
        r"Best Epoch\s+(?P<epoch>-?\d+).*?Final Val loss\s+[-+0-9.eE]+"
        r"\s+\((?P<val_micro>[-+0-9.eE]+),\s*(?P<val_macro>[-+0-9.eE]+)\),"
        r"\s+Test loss\s+[-+0-9.eE]+\s+\((?P<test_micro>[-+0-9.eE]+),\s*(?P<test_macro>[-+0-9.eE]+)\)",
        stdout,
        re.DOTALL,
    )
    if best_match is None:
        return {
            "status": "failed_metric_parse",
            "best_epoch": "",
            "validation_micro_f1": "",
            "validation_macro_f1": "",
            "test_micro_f1": "",
            "test_macro_f1": "",
            "test_accuracy_if_single_label": "",
            "is_multilabel": bool(is_multilabel),
            "loss_type": "bce" if is_multilabel else "ce",
            "error_message": "could not parse official SeHGNN best epoch metrics",
        }
    test_micro = float(best_match.group("test_micro")) / 100.0
    return {
        "status": "success",
        "best_epoch": int(best_match.group("epoch")),
        "validation_micro_f1": float(best_match.group("val_micro")) / 100.0,
        "validation_macro_f1": float(best_match.group("val_macro")) / 100.0,
        "test_micro_f1": test_micro,
        "test_macro_f1": float(best_match.group("test_macro")) / 100.0,
        "test_accuracy_if_single_label": "" if is_multilabel else test_micro,
        "is_multilabel": bool(is_multilabel),
        "loss_type": "bce" if is_multilabel else "ce",
        "error_message": "",
    }


def collect_native_environment(*, repo_dir: Path, data_root: Path, device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "repo_url": REPO_URL,
        "repo_dir": str(repo_dir),
        "repo_commit": repo_commit_hash(repo_dir),
        "data_root": str(data_root),
        "device": str(device),
        "sparse_tools_import_mode": "external_patches/sehgnn_sparse_tools_shim",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception as exc:  # pragma: no cover - environment dependent.
        info["torch_error"] = str(exc)
    try:
        import dgl  # type: ignore

        info["dgl_version"] = dgl.__version__
    except Exception as exc:  # pragma: no cover - environment dependent.
        info["dgl_error"] = str(exc)
    try:
        import torch_sparse  # type: ignore

        info["torch_sparse_version"] = getattr(torch_sparse, "__version__", "")
    except Exception as exc:  # pragma: no cover - environment dependent.
        info["torch_sparse_error"] = str(exc)
    return info


def write_sehgnn_repo_manifest(*, repo_dir: Path, data_root: Path, out_dir: Path, device: str) -> dict[str, Any]:
    manifest = collect_native_environment(repo_dir=Path(repo_dir), data_root=Path(data_root), device=device)
    manifest.update(
        {
            "has_hgb_main_py": (Path(repo_dir) / "hgb" / "main.py").exists(),
            "has_hgb_utils_py": (Path(repo_dir) / "hgb" / "utils.py").exists(),
            "has_data_loader_py": (Path(repo_dir) / "data" / "data_loader.py").exists(),
        }
    )
    write_json(Path(out_dir) / "preflight" / "sehgnn_repo_manifest.json", manifest)
    return manifest


def _status_from_failure(stdout: str, stderr: str, returncode: int) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "not downloaded" in text or "node.dat" in text or "no such file" in text or "filenotfounderror" in text:
        return "failed_dependency"
    if "out of memory" in text or "cuda error" in text and "memory" in text:
        return "failed_oom"
    return "failed_runtime" if returncode else "failed_metric_parse"


def run_native_command(native_command: NativeCommand, *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(
        native_command.command,
        cwd=native_command.cwd,
        env=native_subprocess_env(repo_dir=native_command.cwd.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    train_time = float(time.perf_counter() - start)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    parsed = parse_official_hgb_stdout(native_command.dataset, completed.stdout) if completed.returncode == 0 else {}
    if completed.returncode == 0 and parsed.get("status") == "success":
        status = "success"
        error_message = ""
    else:
        status = _status_from_failure(completed.stdout, completed.stderr, int(completed.returncode))
        error_message = str(parsed.get("error_message") or completed.stderr.strip() or completed.stdout.strip())
    base = {
        "dataset": native_command.dataset,
        "seed": int(native_command.seed),
        "status": status,
        "command": " ".join(native_command.command),
        "train_time_sec": train_time,
        "peak_memory_mb": "",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "error_message": error_message,
    }
    if status == "success":
        base.update(parsed)
        base["status"] = "success"
        base["error_message"] = ""
    else:
        failed_parse = parse_official_hgb_stdout(native_command.dataset, completed.stdout)
        base.update({key: failed_parse.get(key, "") for key in NATIVE_METRIC_FIELDS if key not in base})
        base["status"] = status
        base["error_message"] = error_message
    return base


def summarize_native_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for dataset, group in sorted(grouped.items()):
        successes = [row for row in group if row.get("status") == "success"]
        values = [float(row["test_micro_f1"]) for row in successes if row.get("test_micro_f1") not in {"", None}]
        out.append(
            {
                "dataset": dataset,
                "runs": len(group),
                "success_count": len(successes),
                "failed_count": len(group) - len(successes),
                "test_micro_f1_mean": sum(values) / len(values) if values else "",
                "test_micro_f1_max": max(values) if values else "",
            }
        )
    return out


def run_native_stage(
    *,
    repo_dir: Path,
    data_root: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    device: str,
    out_dir: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    native_dir = out_dir / "native"
    stdout_dir = native_dir / "native_raw_stdout"
    stderr_dir = native_dir / "native_raw_stderr"
    manifest = write_sehgnn_repo_manifest(repo_dir=repo_dir, data_root=data_root, out_dir=out_dir, device=device)
    env = collect_native_environment(repo_dir=repo_dir, data_root=data_root, device=device)
    write_json(native_dir / "native_environment.json", env)
    audits = [audit_native_hgb_data_dir(dataset, data_root, repo_dir) for dataset in datasets]
    write_csv(native_dir / "native_data_audit.csv", audits)
    rows: list[dict[str, Any]] = []
    parser_audit: list[dict[str, Any]] = []
    command_manifest: list[dict[str, Any]] = []
    stop_reason = ""
    for dataset in datasets:
        if stop_reason:
            command_manifest.append({"dataset": dataset, "status": "skipped_after_stop", "reason": stop_reason})
            continue
        dataset_name = supported_sehgnn_hgb_dataset(dataset)
        for seed in seeds:
            native_command = build_official_hgb_command(
                dataset=dataset_name,
                seed=int(seed),
                repo_dir=repo_dir,
                data_root=data_root,
                device=device,
                python_executable=python_executable,
            )
            command_manifest.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "command": native_command.command,
                    "cwd": str(native_command.cwd),
                    "uses_official_main_py": True,
                    "uses_official_preprocess": True,
                    "uses_model_class_adapter_only": False,
                }
            )
            row = run_native_command(
                native_command,
                stdout_path=stdout_dir / f"{dataset_name}_{int(seed)}.log",
                stderr_path=stderr_dir / f"{dataset_name}_{int(seed)}.stderr",
            )
            rows.append(row)
            parser_audit.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "parser_status": "success" if row.get("status") == "success" else "not_parsed",
                    "best_epoch": row.get("best_epoch", ""),
                    "error_message": row.get("error_message", ""),
                }
            )
        if dataset_name == "DBLP" and not all(row.get("status") == "success" for row in rows if row.get("dataset") == "DBLP"):
            stop_reason = "native official DBLP full graph did not reproduce; stopping before ACM/IMDB/export"
    write_csv(native_dir / "native_metrics.csv", rows, fieldnames=NATIVE_METRIC_FIELDS)
    write_csv(native_dir / "native_summary_by_dataset.csv", summarize_native_metrics(rows))
    write_json(native_dir / "native_command_manifest.json", {"commands": command_manifest, "stop_reason": stop_reason})
    write_csv(native_dir / "native_metric_parser_audit.csv", parser_audit)
    return {"manifest": manifest, "rows": rows, "stop_reason": stop_reason}
