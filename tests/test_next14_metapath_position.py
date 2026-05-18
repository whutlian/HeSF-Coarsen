from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_metapath_appendix_keeps_path_mass_out_of_main_claims(tmp_path: Path) -> None:
    from experiments.scripts.summarize_next14_metapath_appendix import summarize_next14_metapath_appendix

    source = tmp_path / "next13"
    _write_csv(source / "metapath_mass_by_method.csv", [{"method": "HeSF-LVC-P", "metapath_mass_relative_error_mean": 0.68}, {"method": "flatten-sum", "metapath_mass_relative_error_mean": 0.64}, {"method": "H6-no-spec", "metapath_mass_relative_error_mean": 0.47}])
    _write_csv(source / "metapath_mass_by_dataset.csv", [{"dataset": "ACM", "method": "HeSF-LVC-P", "metapath_mass_relative_error_mean": 0.68}])
    _write_csv(source / "metapath_mass_gap_vs_flatten_h6.csv", [{"method": "HeSF-LVC-P", "delta_vs_flatten_sum": 0.04, "delta_vs_H6": 0.20}])
    output = tmp_path / "out"
    summarize_next14_metapath_appendix(next13_metapath=source, output=output)
    by_method = _read_csv(output / "appendix_metapath_mass_by_method.csv")
    assert {row["method"] for row in by_method} >= {"HeSF-LVC-P", "flatten-sum", "H6-no-spec"}
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "appendix-only" in summary
    assert "P/S preserve metapaths better than flatten-sum/H6" not in summary
    assert "do not support P/S superiority over flatten-sum/H6" in summary
