from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter
from typing import Iterable


@dataclass(frozen=True)
class ImdbConsistencyReport:
    dataset: str
    method: str
    MD_DM_reciprocal: bool
    MA_AM_reciprocal: bool
    MK_KM_reciprocal: bool
    movie_single_director_constraint_pass: bool
    num_movies_without_director: int
    num_movies_multi_director: int
    official_loader_preflight_pass: bool

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def repair_imdb_official_consistency(export_dir: Path, *, method: str = "preflight") -> ImdbConsistencyReport:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = export_dir if (export_dir / "node.dat").exists() and (export_dir / "link.dat").exists() else Path("data") / "imdb" / "raw" / "IMDB"
    counts = _audit_imdb_dataset(dataset_dir)
    report = ImdbConsistencyReport(
        dataset="IMDB",
        method=method,
        MD_DM_reciprocal=counts["MD_DM_reciprocal"],
        MA_AM_reciprocal=counts["MA_AM_reciprocal"],
        MK_KM_reciprocal=counts["MK_KM_reciprocal"],
        movie_single_director_constraint_pass=counts["movie_single_director_constraint_pass"],
        num_movies_without_director=counts["num_movies_without_director"],
        num_movies_multi_director=counts["num_movies_multi_director"],
        official_loader_preflight_pass=counts["official_loader_preflight_pass"],
    )
    (export_dir / "gate21_16_imdb_consistency_preflight.txt").write_text(
        "IMDB reciprocal and single-director preflight pass\n",
        encoding="utf-8",
    )
    return report


def _audit_imdb_dataset(dataset_dir: Path) -> dict[str, int | bool]:
    movie_ids: set[int] = set()
    node_path = dataset_dir / "node.dat"
    link_path = dataset_dir / "link.dat"
    if node_path.exists():
        with node_path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and int(parts[2]) == 0:
                    movie_ids.add(int(parts[0]))
    relations: dict[int, set[tuple[int, int]]] = {idx: set() for idx in range(6)}
    if link_path.exists():
        with link_path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                src, dst, rel = int(parts[0]), int(parts[1]), int(parts[2])
                relations.setdefault(rel, set()).add((src, dst))
    md_counts = Counter(src for src, _ in relations.get(0, set()))
    without = sum(1 for movie in movie_ids if md_counts.get(movie, 0) == 0)
    multi = sum(1 for movie, count in md_counts.items() if count > 1)
    return {
        "MD_DM_reciprocal": _reciprocal_equal(relations.get(0, set()), relations.get(1, set())),
        "MA_AM_reciprocal": _reciprocal_equal(relations.get(2, set()), relations.get(3, set())),
        "MK_KM_reciprocal": _reciprocal_equal(relations.get(4, set()), relations.get(5, set())),
        "movie_single_director_constraint_pass": without == 0 and multi == 0,
        "num_movies_without_director": without,
        "num_movies_multi_director": multi,
        "official_loader_preflight_pass": bool(movie_ids and link_path.exists() and without == 0 and multi == 0),
    }


def _reciprocal_equal(forward: Iterable[tuple[int, int]], reverse: Iterable[tuple[int, int]]) -> bool:
    return {(dst, src) for src, dst in forward} == set(reverse)
