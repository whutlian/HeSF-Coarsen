from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> Path:
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def git_commit_hash(root: Path | None = None) -> str | None:
    root = root or repo_root()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_config_snapshot(path: str | Path, config: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True)
    return path


def write_command_metadata(
    run_dir: str | Path,
    *,
    run_name: str,
    command: Sequence[str] | None = None,
    status: str = "created",
    **extra: Any,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    now = utc_now_iso()
    metadata_path = run_dir / "metadata.json"
    previous: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            previous = read_json(metadata_path)
        except (json.JSONDecodeError, OSError):
            previous = {}
    command_list = list(command or previous.get("command", []))
    end_time = previous.get("end_time")
    if status not in {"created", "running"}:
        end_time = now
    payload: dict[str, Any] = {
        "run_name": run_name,
        "status": status,
        "command": command_list,
        "created_at": previous.get("created_at", now),
        "start_time": previous.get("start_time", now),
        "end_time": end_time,
        "updated_at": now,
        "git_commit": git_commit_hash(),
    }
    payload.update(extra)
    write_json(metadata_path, payload)
    if command_list:
        write_json(
            run_dir / "command.json",
            {
                "command": command_list,
                "cwd": str(repo_root()),
                "git_commit": payload.get("git_commit"),
                "created_at": payload.get("created_at"),
            },
        )
    return payload


def run_subprocess_with_log(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    log_path: str | Path,
    env: Mapping[str, str] | None = None,
    stream_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    if stream_output:
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(map(str, command)) + "\n\n")
            handle.write("OUTPUT\n")
            handle.flush()
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=merged_env,
            )
            output_parts: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                output_parts.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
            returncode = process.wait()
            handle.write(f"\nRETURN_CODE {returncode}\n")
        return subprocess.CompletedProcess(list(command), returncode, "".join(output_parts), "")
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        env=merged_env,
        check=False,
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(map(str, command)) + "\n\n")
        handle.write("STDOUT\n")
        handle.write(completed.stdout)
        handle.write("\nSTDERR\n")
        handle.write(completed.stderr)
        handle.write(f"\nRETURN_CODE {completed.returncode}\n")
    return completed


def disk_usage_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return int(total)


def flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            row.update(flatten_mapping(value, name))
        else:
            row[name] = value
    return row


def diagnostics_row(run_dir: str | Path, diagnostics_path: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    diagnostics_path = Path(diagnostics_path)
    diagnostics = read_json(diagnostics_path)
    row = flatten_mapping(diagnostics)
    row["run_dir"] = str(run_dir)
    row["level"] = diagnostics_path.parent.name.removeprefix("level_")
    row["diagnostics_path"] = str(diagnostics_path)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        for key, value in metadata.items():
            if key not in row and not isinstance(value, (dict, list)):
                row[key] = value
    return row


def discover_run_dirs(inputs: Iterable[str | Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if (path / "metadata.json").exists() or any(path.glob("level_*/diagnostics.json")):
            run_dirs.add(path)
        if path.exists() and path.is_dir():
            for metadata in path.rglob("metadata.json"):
                run_dirs.add(metadata.parent)
            for diagnostics in path.rglob("level_*/diagnostics.json"):
                run_dirs.add(diagnostics.parent.parent)
    return sorted(run_dirs)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])
