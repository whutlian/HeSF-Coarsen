from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_hgb_metadata_files(export_dir: Path, metadata: Mapping[str, Any]) -> None:
    export_dir = Path(export_dir)
    (export_dir / "hgb").mkdir(parents=True, exist_ok=True)
    node_types = list(metadata.get("node_type_names", []))
    relation_types = list(metadata.get("relation_type_names", []))
    (export_dir / "hgb" / "node_types.json").write_text(json.dumps(node_types, indent=2), encoding="utf-8")
    (export_dir / "hgb" / "relation_types.json").write_text(json.dumps(relation_types, indent=2), encoding="utf-8")
    (export_dir / "hgb" / "dataset_info.json").write_text(json.dumps(dict(metadata), indent=2, default=str), encoding="utf-8")
