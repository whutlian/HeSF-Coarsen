from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from hesf_coarsen.eval.official.runner_utils import repo_commit_hash, write_json


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
        "model_name": "SeHGNN-official",
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

    result.update(
        {
            "status": "failed_format_adapter",
            "error_message": "official SeHGNN adapter for Gate21 HGB export is not installed; no lite fallback used",
            "returncode": 2,
            "train_time_sec": float(perf_counter() - start),
            "command": "",
        }
    )
    stderr_path.write_text(str(result["error_message"]), encoding="utf-8")
    return result
