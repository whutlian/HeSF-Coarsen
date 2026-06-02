from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


REQUIRED_EXTERNAL_REPOS: dict[str, dict[str, Any]] = {
    "FreeHGC": {
        "repo_url": "https://github.com/GooLiang/FreeHGC",
        "dirname": "FreeHGC",
        "required_files": ("README.md", "HGB/train_hgb.py", "HGB/data_hgb.py", "HGB/model_hgb.py"),
    },
    "HGCond": {
        "repo_url": "https://github.com/jianjianGJ/hgcond",
        "dirname": "hgcond",
        "required_files": ("README.md",),
    },
    "GCond": {
        "repo_url": "https://github.com/ChandlerBang/GCond",
        "dirname": "GCond",
        "required_files": ("README.md",),
    },
    "GCondenser": {
        "repo_url": "https://github.com/superallen13/GCondenser",
        "dirname": "GCondenser",
        "required_files": ("README.md",),
    },
}


def audit_required_external_repos(external_repos_dir: str | Path, *, clone_missing: bool = False) -> list[dict[str, Any]]:
    root = Path(external_repos_dir)
    rows: list[dict[str, Any]] = []
    for name, spec in REQUIRED_EXTERNAL_REPOS.items():
        repo_dir = root / str(spec["dirname"])
        clone_attempted = False
        clone_result: subprocess.CompletedProcess[str] | None = None
        if not repo_dir.exists() and clone_missing:
            clone_attempted = True
            root.mkdir(parents=True, exist_ok=True)
            clone_result = subprocess.run(
                ["git", "clone", str(spec["repo_url"]), str(repo_dir)],
                text=True,
                capture_output=True,
                check=False,
            )
        elif repo_dir.exists() and clone_missing and (repo_dir / ".git").exists():
            subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--tags"], text=True, capture_output=True, check=False)

        exists = repo_dir.exists()
        git_repo = (repo_dir / ".git").exists()
        commit_hash = _git_stdout(repo_dir, "rev-parse", "HEAD") if git_repo else ""
        branch_or_tag = _git_stdout(repo_dir, "rev-parse", "--abbrev-ref", "HEAD") if git_repo else ""
        remote_url = _git_stdout(repo_dir, "config", "--get", "remote.origin.url") if git_repo else ""
        missing_files = [rel for rel in spec["required_files"] if not (repo_dir / rel).exists()]
        required_files_present = exists and not missing_files
        clone_success = exists and git_repo and bool(commit_hash)
        failure_type = ""
        failure_reason = ""
        if not clone_success:
            failure_type = "repo_missing" if not clone_attempted else "clone_failed"
            if clone_result is not None and clone_result.returncode != 0:
                failure_reason = (clone_result.stderr or clone_result.stdout).strip()
            else:
                failure_reason = f"{name} repository is not present under {root}."
        elif not required_files_present:
            failure_type = "missing_required_file"
            failure_reason = "Missing required files: " + ";".join(missing_files)

        rows.append(
            {
                "baseline_name": name,
                "repo_url": spec["repo_url"],
                "local_path": str(repo_dir),
                "clone_attempted": clone_attempted,
                "clone_success": clone_success,
                "commit_hash": commit_hash,
                "branch_or_tag": branch_or_tag,
                "remote_url": remote_url,
                "required_files_present": required_files_present,
                "missing_required_files": ";".join(missing_files),
                "adapter_implemented": name in {"FreeHGC"},
                "protocol_supported": False,
                "failure_type": failure_type,
                "failure_reason": failure_reason,
            }
        )
    return rows


def _git_stdout(repo_dir: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo_dir), *args], text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""
