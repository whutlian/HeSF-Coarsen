from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv, write_json


def estimate_scale_bytes(nodes: int, edges: int, feature_dim: int = 32, candidate_k: int = 16) -> dict[str, int]:
    nodes = int(nodes)
    edges = int(edges)
    feature_dim = int(feature_dim)
    candidate_k = int(candidate_k)
    relation_arrays_bytes = edges * (8 + 8 + 4)
    node_type_bytes = nodes * 4
    feature_bytes_fp32 = nodes * feature_dim * 4
    sketch_bytes_fp16 = nodes * feature_dim * 2
    candidate_store_bytes = nodes * candidate_k * (8 + 4 + 2)
    diagnostics_bytes = max(4096, nodes // 128)
    expected_disk_footprint_bytes = (
        relation_arrays_bytes
        + node_type_bytes
        + feature_bytes_fp32
        + sketch_bytes_fp16
        + candidate_store_bytes
        + diagnostics_bytes
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "feature_dim": feature_dim,
        "candidate_k": candidate_k,
        "relation_arrays_bytes": relation_arrays_bytes,
        "node_type_bytes": node_type_bytes,
        "feature_bytes_fp32": feature_bytes_fp32,
        "sketch_bytes_fp16": sketch_bytes_fp16,
        "candidate_store_bytes": candidate_store_bytes,
        "diagnostics_bytes": diagnostics_bytes,
        "expected_disk_footprint_bytes": expected_disk_footprint_bytes,
    }


def parse_scale_label(label: str) -> tuple[int, int]:
    parts = label.lower().split("_")
    if len(parts) != 2:
        raise ValueError(f"scale must be formatted like 1m_10m, got {label!r}")
    return _parse_count(parts[0]), _parse_count(parts[1])


def _parse_count(value: str) -> int:
    value = value.strip().lower().replace(",", "")
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("m"):
        multiplier = 1_000_000
        value = value[:-1]
    elif value.endswith("b"):
        multiplier = 1_000_000_000
        value = value[:-1]
    return int(float(value) * multiplier)


def write_scale_manifests(output: Path, scales: list[str], feature_dim: int, candidate_k: int, seed: int) -> list[dict[str, int | str]]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str]] = []
    for label in scales:
        nodes, edges = parse_scale_label(label)
        scale_dir = output / label
        estimate = estimate_scale_bytes(nodes, edges, feature_dim, candidate_k)
        payload = {"scale": label, "seed": int(seed), **estimate}
        write_json(scale_dir / "estimate.json", payload)
        rows.append({"scale": label, "scale_dir": str(scale_dir), **payload})
    write_csv(output / "summary.csv", rows)
    report = ["# Synthetic Scale Manifest", "", markdown_table(rows, ["scale", "nodes", "edges", "expected_disk_footprint_bytes", "scale_dir"])]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate synthetic scale experiment storage.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_scale"))
    parser.add_argument("--scales", nargs="+", help="Scale labels such as 1m_10m, 5m_50m, 10m_100m.")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--edges", type=int)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scales:
        rows = write_scale_manifests(args.output, args.scales, args.feature_dim, args.candidate_k, args.seed)
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if args.nodes is None or args.edges is None:
        parser.error("--nodes and --edges are required when --scales is not provided")
    print(json.dumps(estimate_scale_bytes(args.nodes, args.edges, args.feature_dim, args.candidate_k), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
