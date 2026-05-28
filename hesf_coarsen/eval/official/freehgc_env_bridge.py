from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


FREEHGC_REPO_URL = "https://github.com/GooLiang/FreeHGC"
REQUIRED_IMPORTS = ("torch", "torch_geometric", "torch_sparse", "torch_scatter", "numpy")
REQUIRED_FILES = (
    "README.md",
    "HGB/train_hgb.py",
    "HGB/data_hgb.py",
    "HGB/model_hgb.py",
    "HGB/model_SeHGNN.py",
)


def resolve_freehgc_root(root: str | Path | None = None) -> Path:
    if root:
        return Path(root)
    env_root = os.environ.get("FREEHGC_ROOT", "")
    if env_root:
        return Path(env_root)
    return Path("external") / "FreeHGC"


def freehgc_preflight(
    *,
    freehgc_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    expected_env: str = "conda:pytorch",
) -> dict[str, Any]:
    root = resolve_freehgc_root(freehgc_root)
    python = str(python_executable or sys.executable)
    missing_files = [
        str(path.relative_to(root)) if root.exists() else str(path)
        for path in [root / required for required in REQUIRED_FILES]
        if not path.exists()
    ]
    import_rows = [_import_status(module) for module in REQUIRED_IMPORTS]
    missing_imports = [row["module"] for row in import_rows if not row["available"]]
    return {
        "freehgc_repo_url": FREEHGC_REPO_URL,
        "freehgc_root": str(root),
        "freehgc_root_exists": root.exists(),
        "expected_env": expected_env,
        "python_executable": python,
        "has_hgb_train": (root / "HGB" / "train_hgb.py").exists(),
        "has_hgb_data_loader": (root / "HGB" / "data_hgb.py").exists(),
        "missing_files": ";".join(missing_files),
        "import_audit": import_rows,
        "missing_dependency": bool(missing_files or missing_imports),
        "missing_dependency_name": ";".join([*missing_files, *missing_imports]),
        "attempted_import": ";".join(REQUIRED_IMPORTS),
        "suggested_install_command": suggested_install_command(missing_imports),
    }


def _import_status(module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    return {"module": module, "available": spec is not None}


def suggested_install_command(missing_imports: Sequence[str]) -> str:
    missing = set(missing_imports)
    commands: list[str] = []
    if "torch_geometric" in missing:
        commands.append("conda run -n pytorch pip install torch-geometric")
    wheel_modules = sorted(missing & {"torch_sparse", "torch_scatter"})
    if wheel_modules:
        commands.append(
            "conda run -n pytorch pip install "
            + " ".join(wheel_modules)
            + " -f https://data.pyg.org/whl/torch-2.8.0+cu128.html"
        )
    if "numpy" in missing:
        commands.append("conda run -n pytorch pip install numpy")
    return " && ".join(commands)


def build_freehgc_command(
    *,
    freehgc_root: str | Path,
    dataset: str,
    data_root: str | Path,
    reduction_rate: float,
    seed: int,
    device: str,
    quick: bool = False,
    python_executable: str | Path | None = None,
) -> list[str]:
    gpu = "0"
    if str(device).lower() == "cpu":
        gpu = "-1"
    command = [
        str(python_executable or sys.executable),
        "train_hgb.py",
        "--dataset",
        str(dataset).upper(),
        "--root",
        str(Path(data_root).resolve()),
        "--method",
        "FreeHGC",
        "--reduction-rate",
        str(float(reduction_rate)),
        "--pr",
        "0.95",
        "--gpu",
        gpu,
        "--seed",
        str(int(seed)),
        "--num-hops",
        "2",
        "--num-hidden",
        "128",
        "--lr",
        "0.001",
        "--dropout",
        "0.5",
        "--ff-layer-2",
        "2",
        "--model",
        "SeHGNN",
    ]
    if quick:
        command.extend(["--num-epochs", "1", "--eval-every", "1", "--batch-size", "10000", "--eval-batch-size", "10000"])
    return command


def missing_dependency_failure_row(
    *,
    dataset: str,
    method: str,
    protocol: str,
    preflight: dict[str, Any],
    reduction_rate: float | None = None,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset).upper(),
        "method": str(method),
        "baseline_name": str(method),
        "method_family": "external_freehgc",
        "protocol": str(protocol),
        "external_baseline": True,
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": False,
        "training_executed": False,
        "eligible_for_tp_main_comparison": False,
        "uses_synthetic_nodes": False,
        "uses_weighted_edges": False,
        "requires_loader_adapter": True,
        "missing_dependency": True,
        "missing_dependency_name": preflight.get("missing_dependency_name", ""),
        "attempted_import": preflight.get("attempted_import", ""),
        "expected_env": preflight.get("expected_env", "conda:pytorch"),
        "suggested_install_command": preflight.get("suggested_install_command", ""),
        "freehgc_repo_url": FREEHGC_REPO_URL,
        "freehgc_root": preflight.get("freehgc_root", ""),
        "failure_type": "missing_external_dependency",
        "failure_message": "FreeHGC upstream preflight failed; row is not READY.",
        "support_node_ratio": "" if reduction_rate is None else float(reduction_rate),
        "test_micro_f1": "",
        "test_macro_f1": "",
        "success": False,
        "success_count": 0,
    }


def run_subprocess(command: Sequence[str], *, cwd: str | Path, timeout_seconds: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=Path(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
