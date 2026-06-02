from __future__ import annotations

from pathlib import Path
from typing import Any

from hesf_coarsen.eval.official.acm_consistency_export import repair_acm_official_consistency


def repair_gate21_17_acm_export(export_dir: str | Path, *, mode: str = "conservative_locked") -> dict[str, Any]:
    report = repair_acm_official_consistency(Path(export_dir), mode=mode)
    row = report.as_row()
    return {
        "dataset": "ACM",
        "mode": row.get("mode", mode),
        "P_nonzero_count": row.get("P_nonzero_count", ""),
        "PK_edge_count": row.get("PK_edge_count", ""),
        "P_matches_PK": row.get("P_matches_PK", False),
        "A_matches_AP_PK": row.get("A_matches_AP_PK", False),
        "C_matches_CP_PK": row.get("C_matches_CP_PK", False),
        "PA_AP_reciprocal": row.get("PA_AP_reciprocal", False),
        "PC_CP_reciprocal": row.get("PC_CP_reciprocal", False),
        "PK_KP_reciprocal": row.get("PK_KP_reciprocal", False),
        "official_loader_preflight_pass": row.get("official_loader_preflight_pass", False),
    }
