from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from hesf_coarsen.eval.official.runner_utils import repo_commit_hash, write_json


MODELCLASS_LABEL = "SeHGNN-modelclass-HeSF-features"
ADAPTER_WARNING = "WARNING: This run uses HeSF-built target feature blocks and is not the official SeHGNN HGB preprocessing pipeline."


def _metadata(export_dir: Path) -> dict[str, Any]:
    path = Path(export_dir) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio_token(value: Any) -> str:
    if value in {"", None}:
        return "none"
    return f"{float(value):.2f}".replace(".", "p")


def _base_result(
    *,
    export_dir: Path,
    repo_dir: Path,
    dataset_name: str,
    target_type: str,
    seed: int,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    meta = _metadata(export_dir)
    method = str(meta.get("method", config.get("method", "")))
    ratio = meta.get("support_ratio", config.get("support_ratio", ""))
    logs_dir = Path(output_dir) / "logs"
    configs_dir = Path(output_dir) / "configs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"sehgnn_official_{dataset_name}_{int(seed)}_{method}_{_ratio_token(ratio)}"
    stdout_path = logs_dir / f"{run_id}.stdout"
    stderr_path = logs_dir / f"{run_id}.stderr"
    config_path = configs_dir / f"{run_id}.json"
    return {
        "model_name": MODELCLASS_LABEL,
        "dataset": str(dataset_name),
        "seed": int(seed),
        "method": method,
        "support_ratio": "" if ratio is None else ratio,
        "target_type": str(target_type),
        "validation_macro_f1": "",
        "validation_micro_f1": "",
        "validation_accuracy": "",
        "test_macro_f1": "",
        "test_micro_f1": "",
        "test_accuracy": "",
        "val_logits_path": "",
        "test_logits_path": "",
        "best_epoch": "",
        "train_time_sec": "",
        "peak_memory_mb": "",
        "command": "",
        "returncode": "",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "config_path": str(config_path),
        "status": "",
        "error_message": "",
        "calibrated": False,
        "calibration_uses_test_labels": False,
        "selector_uses_test_labels": False,
        "uses_hettree_lite": False,
        "bridge_type": "model_class_only",
        "official_pipeline": False,
        "uses_official_preprocess": False,
        "method_label": MODELCLASS_LABEL,
        "warning": ADAPTER_WARNING,
    }


def run_sehgnn_official(
    export_dir: Path,
    repo_dir: Path,
    dataset_name: str,
    target_type: str,
    seed: int,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    start = perf_counter()
    result = _base_result(
        export_dir=Path(export_dir),
        repo_dir=Path(repo_dir),
        dataset_name=dataset_name,
        target_type=target_type,
        seed=int(seed),
        config=config,
        output_dir=Path(output_dir),
    )
    config_dump = {
        "repo_dir": str(repo_dir),
        "repo_commit": repo_commit_hash(repo_dir),
        "python_executable": sys.executable,
        "dataset": dataset_name,
        "seed": int(seed),
        "method": result["method"],
        "support_ratio": result["support_ratio"],
        "target_type": target_type,
        "hyperparameters": dict(config),
        "export_dir": str(export_dir),
    }
    try:
        import torch

        config_dump["torch_version"] = torch.__version__
        config_dump["cuda_available"] = bool(torch.cuda.is_available())
        config_dump["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:  # pragma: no cover - environment dependent.
        config_dump["torch_error"] = str(exc)
    try:
        import dgl  # type: ignore

        config_dump["dgl_version"] = dgl.__version__
    except Exception as exc:  # pragma: no cover - environment dependent.
        config_dump["dgl_error"] = str(exc)
    write_json(Path(result["config_path"]), config_dump)

    stdout_path = Path(result["stdout_path"])
    stderr_path = Path(result["stderr_path"])
    stdout_path.write_text("", encoding="utf-8")
    if not Path(repo_dir).exists():
        result.update(
            {
                "status": "failed_dependency",
                "error_message": f"missing_repo: {repo_dir}",
                "returncode": 127,
                "train_time_sec": float(perf_counter() - start),
            }
        )
        stderr_path.write_text(result["error_message"], encoding="utf-8")
        return result

    runner_result = Path(output_dir) / "runner_results" / f"{Path(result['config_path']).stem}.json"
    logits_dir = Path(output_dir) / "logits"
    command = [
        sys.executable,
        "-m",
        "hesf_coarsen.eval.official.sehgnn_export_runner",
        "--export-dir",
        str(export_dir),
        "--repo-dir",
        str(repo_dir),
        "--dataset-name",
        str(dataset_name),
        "--target-type",
        str(target_type),
        "--seed",
        str(int(seed)),
        "--result-json",
        str(runner_result),
        "--logits-dir",
        str(logits_dir),
        "--epochs",
        str(int(config.get("epochs", config.get("epoch", 12)))),
        "--embed-size",
        str(int(config.get("embed_size", 64))),
        "--hidden",
        str(int(config.get("hidden", 64))),
        "--batch-size",
        str(int(config.get("batch_size", 2048))),
        "--lr",
        str(float(config.get("lr", 0.001))),
        "--weight-decay",
        str(float(config.get("weight_decay", 0.0))),
        "--device",
        str(config.get("device", "cpu")),
    ]
    result["command"] = " ".join(command)
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[3], text=True, capture_output=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result["returncode"] = int(completed.returncode)
    if runner_result.exists():
        payload = json.loads(runner_result.read_text(encoding="utf-8"))
    else:
        payload = {}
    if completed.returncode == 0 and payload.get("status") == "success":
        result.update(payload)
        result.update(
            {
                "model_name": MODELCLASS_LABEL,
                "dataset": str(dataset_name),
                "seed": int(seed),
                "method": result["method"],
                "support_ratio": result["support_ratio"],
                "target_type": str(target_type),
                "command": " ".join(command),
                "returncode": 0,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "config_path": str(result["config_path"]),
                "calibrated": False,
                "calibration_uses_test_labels": False,
                "selector_uses_test_labels": False,
                "uses_hettree_lite": False,
                "bridge_type": "model_class_only",
                "official_pipeline": False,
                "uses_official_preprocess": False,
                "method_label": MODELCLASS_LABEL,
                "warning": ADAPTER_WARNING,
            }
        )
        return result
    status = str(payload.get("status") or ("failed_oom" if "out of memory" in completed.stderr.lower() else "failed_runtime"))
    error_message = str(payload.get("error_message") or completed.stderr.strip() or completed.stdout.strip() or "official SeHGNN runner failed")
    result.update(
        {
            "status": status,
            "error_message": error_message,
            "train_time_sec": float(perf_counter() - start),
        }
    )
    return result
