from __future__ import annotations

from pathlib import Path
from typing import Any

from hesf_coarsen.eval.official.imdb_consistency_export import repair_imdb_official_consistency


def repair_gate21_17_imdb_export(export_dir: str | Path) -> dict[str, Any]:
    report = repair_imdb_official_consistency(Path(export_dir))
    row = report.as_row()
    return {
        "dataset": "IMDB",
        "MD_DM_reciprocal": row.get("MD_DM_reciprocal", False),
        "MA_AM_reciprocal": row.get("MA_AM_reciprocal", False),
        "MK_KM_reciprocal": row.get("MK_KM_reciprocal", False),
        "movie_single_director_constraint_pass": row.get("movie_single_director_constraint_pass", False),
        "num_movies_without_director": row.get("num_movies_without_director", ""),
        "num_movies_multi_director": row.get("num_movies_multi_director", ""),
        "official_loader_preflight_pass": row.get("official_loader_preflight_pass", False),
    }
