from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_gate21_13_icde_manifest(
    *, result_dir: str | Path, summary_files: Sequence[str], decision: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(result_dir)
    files = []
    for name in summary_files:
        path = root / name
        files.append(
            {
                "file": name,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return {
        "gate": "gate21_13",
        "result_dir": str(root),
        "decision_status": decision.get("status", ""),
        "icde_evidence_ready": bool(decision.get("flags", {}).get("ICDE_EVIDENCE_READY")),
        "files": files,
    }


def write_gate21_13_icde_manifest(
    *, result_dir: str | Path, summary_files: Sequence[str], decision: Mapping[str, Any], output_path: str | Path
) -> dict[str, Any]:
    manifest = build_gate21_13_icde_manifest(result_dir=result_dir, summary_files=summary_files, decision=decision)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
