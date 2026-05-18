from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from experiments.scripts.next11_common import read_csv


def summarize_next14_metapath_appendix(*, next13_metapath: str | Path, output: str | Path) -> dict[str, int]:
    next13_metapath = Path(next13_metapath)
    output = Path(output)
    required = [
        next13_metapath / "metapath_mass_by_method.csv",
        next13_metapath / "metapath_mass_by_dataset.csv",
        next13_metapath / "metapath_mass_gap_vs_flatten_h6.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required metapath inputs: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    by_method = read_csv(required[0])
    by_dataset = read_csv(required[1])
    gaps = read_csv(required[2])
    write_csv(output / "appendix_metapath_mass_by_method.csv", by_method)
    write_csv(output / "appendix_metapath_mass_by_dataset.csv", by_dataset)
    write_csv(output / "appendix_metapath_mass_gap_vs_flatten_h6.csv", gaps)
    lines = [
        "# Next14 Metapath Appendix Position",
        "",
        "Path-mass diagnostics are appendix-only. They measure typed transition mass, not held-out fused low-frequency operator preservation.",
        "Next13 results are method-sensitive but do not support P/S superiority over flatten-sum/H6.",
        "They can be cited as secondary evidence against weaker baselines such as H0, random, GraphZoom-style, ConvMatch-style, and TypedHash.",
        "",
        markdown_table(by_method, list(by_method[0].keys()) if by_method else ["method"]),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"method_rows": len(by_method), "dataset_rows": len(by_dataset)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next13-metapath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_next14_metapath_appendix(next13_metapath=args.next13_metapath, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
