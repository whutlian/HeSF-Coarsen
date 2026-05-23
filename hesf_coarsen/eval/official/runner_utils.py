from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fields: list[str] = [str(name) for name in fieldnames] if fieldnames is not None else []
    seen: set[str] = set(ordered_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                ordered_fields.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered_fields})


def repo_commit_hash(repo_dir: Path | str | None) -> str | None:
    if repo_dir is None:
        return None
    path = Path(repo_dir)
    if not path.exists():
        return None
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def clone_external_repo(url: str, repo_dir: Path) -> dict[str, Any]:
    repo_dir = Path(repo_dir)
    if repo_dir.exists():
        return {"status": "exists", "repo_dir": str(repo_dir), "returncode": 0}
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(["git", "clone", "--depth", "1", url, str(repo_dir)], text=True, capture_output=True, check=False)
    return {
        "status": "success" if completed.returncode == 0 else "failed_clone",
        "repo_dir": str(repo_dir),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def dependency_snapshot(*, sehgnn_repo: Path, openhgnn_repo: Path, hesf_commit: str | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "hesf_coarsen_commit": hesf_commit or "",
        "sehgnn_repo_path": str(sehgnn_repo),
        "sehgnn_repo_exists": Path(sehgnn_repo).exists(),
        "sehgnn_repo_commit": repo_commit_hash(sehgnn_repo),
        "openhgnn_repo_path": str(openhgnn_repo),
        "openhgnn_repo_exists": Path(openhgnn_repo).exists(),
        "openhgnn_repo_commit": repo_commit_hash(openhgnn_repo),
        "hettree_status": "excluded_code_unavailable",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:  # pragma: no cover - depends on environment.
        info["torch_error"] = str(exc)
    try:
        import dgl  # type: ignore

        info["dgl_version"] = dgl.__version__
    except Exception as exc:  # pragma: no cover - depends on environment.
        info["dgl_error"] = str(exc)
    return info


def git_commit_hash(cwd: Path) -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(cwd), text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None
