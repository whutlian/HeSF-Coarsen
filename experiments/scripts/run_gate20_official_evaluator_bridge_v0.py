from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import git_commit_hash, write_csv


def _dependency_status() -> tuple[str, list[str]]:
    missing: list[str] = []
    for name in ("hettree", "SeHGNN", "openhgnn"):
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if len(missing) == 3:
        return "unavailable", missing
    return "partial_dependency_probe_only", missing


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status, missing = _dependency_status()
    rows = []
    for dataset in [str(item).upper() for item in args.datasets]:
        rows.append(
            {
                "dataset": dataset,
                "evaluator_name": "official_or_faithful_full_graph_HETTREE_or_SeHGNN",
                "evaluator_source": "not_available_locally",
                "official_or_faithful": False,
                "full_graph_accuracy": "",
                "full_graph_macro_f1": "",
                "compressed_accuracy": "",
                "compressed_macro_f1": "",
                "can_reach_095_accuracy": "",
                "missing_dependencies": ";".join(missing),
                "official_bridge_status": status,
                "git_commit": git_commit_hash() or "",
            }
        )
    write_csv(output_dir / "gate20_official_bridge_results.csv", rows)
    status_lines = [
        "# Gate20 Official Evaluator Bridge v0",
        "",
        f"- official_bridge_status: {status}",
        f"- missing_dependency_or_checkpoint: {';'.join(missing) if missing else 'none_detected_by_import_probe'}",
        "- lite evaluator results are not reported as official or faithful results.",
        "- Gate21 should port HeSF-CAL only after an official/faithful full-graph evaluator is installed.",
    ]
    (output_dir / "gate20_official_bridge_status.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    missing_lines = [
        "# Gate20 Official Bridge Missing Items",
        "",
        "- Official HETTREE or SeHGNN code/checkpoint is not vendored in this repository.",
        "- No local official evaluator adapter is configured.",
        "- No official full-graph 0.95 ceiling can be claimed from the lite evaluator.",
        "",
        "## Missing Dependencies",
        *[f"- {item}" for item in missing],
    ]
    (output_dir / "gate20_official_bridge_missing_items.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
    return {"official_bridge_status": status, "missing_dependencies": missing}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe official/faithful evaluator availability for Gate20.")
    parser.add_argument("--datasets", nargs="*", default=["DBLP", "ACM"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate20_official_bridge_v0"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
