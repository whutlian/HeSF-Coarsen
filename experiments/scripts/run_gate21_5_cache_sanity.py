from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.metapath_channel_audit import cache_sanity_row
from hesf_coarsen.eval.official.runner_utils import write_csv


def _default_export_dir() -> Path:
    cache_audit = Path("results/gate21_4_apv_skeleton_validation/gate21_4_cache_audit.csv")
    if cache_audit.exists():
        import csv

        with cache_audit.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") == "H6-APV-skeleton" and row.get("graph_seed") == "1":
                    path = Path(row.get("export_dir", ""))
                    if path.exists():
                        return path
    return Path("external/SeHGNN/data/DBLP")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/gate21_5_directed_apv_feature_adapter"))
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir) if args.export_dir is not None else _default_export_dir()
    row = cache_sanity_row(export_dir)
    write_csv(out / "gate21_5_cache_sanity.csv", [row])
    print(json.dumps({"export_dir": str(export_dir), "cache_sanity_pass": row.get("cache_sanity_pass", False)}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
