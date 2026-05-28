from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import add_gate21_7_common_args, ensure_layout, read_csv
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.eval.official.storage_system_costs import build_gate21_7_storage_rows, build_gate21_7_system_resource_rows


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["storage_system_costs"]
    cache_rows = read_csv(Path(args.gate21_5_dir) / "gate21_5_cache_audit.csv")
    full = _export_for(cache_rows, "H6-APV-skeleton") or _export_for(cache_rows, "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00")
    compressed = []
    for name, method in [
        ("APV12_official_text", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"),
        ("APV16_official_text", "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00"),
    ]:
        path = _export_for(cache_rows, method)
        if path:
            compressed.append((name, path))
    if full is None:
        storage_rows = []
        system_rows = []
    else:
        storage_rows = build_gate21_7_storage_rows(dataset=args.dataset, full_export_dir=full, compressed_exports=compressed)
        system_rows = build_gate21_7_system_resource_rows(output_root=out, source_paths=[full, *[path for _name, path in compressed]])
    write_csv(out / "gate21_7_storage_only_baselines.csv", storage_rows)
    write_csv(out / "gate21_7_storage_system_costs.csv", storage_rows)
    write_csv(out / "gate21_7_system_resource_by_stage.csv", system_rows)
    write_json(out / "gate21_7_storage_system_costs_plan.json", {"gate21_5_dir": str(args.gate21_5_dir), "full_export_dir": "" if full is None else str(full)})
    return {"storage_rows": len(storage_rows), "system_rows": len(system_rows)}


def _export_for(rows: list[dict[str, str]], method: str) -> Path | None:
    for row in rows:
        if row.get("method") == method and row.get("training_seed") == "1":
            path = Path(row.get("export_dir", ""))
            if path.exists():
                return path
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--gate21-5-dir", type=Path, default=Path("results/gate21_5_directed_apv_feature_adapter"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
