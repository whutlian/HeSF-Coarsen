from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from hesf_coarsen.io.edge_list import load_graph, save_graph
from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, nodes_of_type


def sample_relation_aware_subset(
    input_dir: str | Path,
    output_dir: str | Path,
    target_nodes: int,
    edge_budget: int | None = None,
    seed: int = 12345,
) -> Path:
    graph = load_graph(input_dir)
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    target_nodes = max(1, min(int(target_nodes), graph.num_nodes))
    selected_parts: list[np.ndarray] = []
    unique_types = sorted(int(t) for t in np.unique(graph.node_type))
    remaining = target_nodes
    for index, type_id in enumerate(unique_types):
        nodes = nodes_of_type(graph, type_id)
        types_left = len(unique_types) - index
        quota = min(len(nodes), max(1, remaining // types_left))
        if index == len(unique_types) - 1:
            quota = min(len(nodes), remaining)
        chosen = rng.choice(nodes, size=quota, replace=False) if quota < len(nodes) else nodes
        selected_parts.append(np.sort(chosen.astype(np.int64)))
        remaining -= int(len(chosen))
    selected = np.sort(np.concatenate(selected_parts)[:target_nodes])
    old_to_new = {int(old): int(new) for new, old in enumerate(selected)}
    keep_mask = np.zeros(graph.num_nodes, dtype=bool)
    keep_mask[selected] = True

    relations: dict[int, RelationAdj] = {}
    total_edges = 0
    for relation_id, rel in sorted(graph.relations.items()):
        mask = keep_mask[rel.src] & keep_mask[rel.dst]
        src = rel.src[mask]
        dst = rel.dst[mask]
        weight = rel.weight[mask] if rel.weight is not None else None
        if edge_budget is not None:
            remaining_edges = max(0, int(edge_budget) - total_edges)
            src = src[:remaining_edges]
            dst = dst[:remaining_edges]
            weight = weight[:remaining_edges] if weight is not None else None
        remap = np.vectorize(old_to_new.__getitem__, otypes=[np.int64])
        relations[relation_id] = RelationAdj(remap(src), remap(dst), weight, rel.src_type, rel.dst_type, relation_id)
        total_edges += len(src)

    features = None
    if graph.features is not None:
        features = {}
        for type_id, matrix in graph.features.items():
            old_type_nodes = nodes_of_type(graph, type_id)
            old_local = {int(node): pos for pos, node in enumerate(old_type_nodes)}
            selected_type_nodes = selected[graph.node_type[selected] == int(type_id)]
            local_idx = [old_local[int(node)] for node in selected_type_nodes]
            features[int(type_id)] = matrix[local_idx]

    subset = HeteroGraph(
        num_nodes=len(selected),
        node_type=graph.node_type[selected],
        relations=relations,
        relation_specs=graph.relation_specs,
        features=features,
        labels=graph.labels[selected] if graph.labels is not None else None,
        partitions=graph.partitions[selected] if graph.partitions is not None else None,
    )
    save_graph(subset, output_dir)
    diagnostics = {
        "input_dir": str(input_dir),
        "target_nodes": int(target_nodes),
        "actual_nodes": int(subset.num_nodes),
        "edge_budget": edge_budget,
        "actual_edges": int(sum(rel.num_edges for rel in subset.relations.values())),
        "seed": int(seed),
        "node_count_by_type": {str(t): int(np.sum(subset.node_type == t)) for t in sorted(np.unique(subset.node_type))},
        "edge_count_by_relation": {str(rid): int(rel.num_edges) for rid, rel in sorted(subset.relations.items())},
    }
    (output_dir / "subset_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir


def _size_label(size: int) -> str:
    if size % 1_000_000 == 0:
        return f"{size // 1_000_000}m"
    if size % 1_000 == 0:
        return f"{size // 1_000}k"
    return str(size)


def _ensure_input_graph(args: argparse.Namespace, input_dir: Path) -> None:
    if (input_dir / "schema.json").exists():
        return
    if not args.import_if_missing:
        raise FileNotFoundError(f"{input_dir} is missing; pass --import-if-missing to import")
    from hesf_coarsen.io.dataset_importers import import_ogbn_mag_dataset

    import_ogbn_mag_dataset(root=args.ogb_root, output=input_dir)


def run_subset_sweep(
    *,
    input_dir: Path,
    output: Path,
    subset_root: Path,
    sizes: list[int],
    edge_budgets: list[int] | None,
    seed: int,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failed = False
    for index, size in enumerate(sizes):
        budget = edge_budgets[index] if edge_budgets and index < len(edge_budgets) else None
        subset_dir = subset_root / f"subset_{_size_label(int(size))}"
        row: dict[str, object] = {
            "run_name": f"ogbn_mag_subset_{_size_label(int(size))}",
            "status": "running",
            "target_nodes": int(size),
            "edge_budget": budget or "",
            "subset_dir": str(subset_dir),
        }
        try:
            sample_relation_aware_subset(input_dir, subset_dir, int(size), budget, seed)
            diagnostics = json.loads((subset_dir / "subset_diagnostics.json").read_text(encoding="utf-8"))
            row.update(
                {
                    "status": "success",
                    "actual_nodes": diagnostics.get("actual_nodes", ""),
                    "actual_edges": diagnostics.get("actual_edges", ""),
                }
            )
        except Exception as exc:
            failed = True
            row.update({"status": "failed", "failure_reason": str(exc)})
        rows.append(row)
    write_csv(output / "summary.csv", rows)
    failures = [row for row in rows if row.get("status") == "failed"]
    write_csv(output / "failures.csv", failures)
    report = [
        "# OGBN-MAG Subset Report",
        "",
        markdown_table(rows, ["run_name", "status", "target_nodes", "actual_nodes", "actual_edges", "subset_dir", "failure_reason"]),
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a deterministic relation-aware OGBN-MAG subset.")
    parser.add_argument("--input", dest="input_alias", type=Path, help="Input graph directory; plan command alias.")
    parser.add_argument("--output", type=Path, help="Sweep output directory; plan command alias.")
    parser.add_argument("--sizes", nargs="+", type=int, help="Run a deterministic subset sweep for these target node counts.")
    parser.add_argument("--subset-root", type=Path, default=Path("data/ogbn_mag_subsets"))
    parser.add_argument("--edge-budgets", nargs="*", type=int)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-nodes", type=int)
    parser.add_argument("--edge-budget", type=int)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--import-if-missing", action="store_true")
    parser.add_argument("--ogb-root", type=Path, default=Path("data/ogb"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dir = args.input_dir or args.input_alias
    if input_dir is None:
        parser.error("--input or --input-dir is required")
    _ensure_input_graph(args, input_dir)
    if args.sizes:
        return run_subset_sweep(
            input_dir=input_dir,
            output=args.output or Path("outputs/experiments/ogbn_mag_subset"),
            subset_root=args.subset_root,
            sizes=[int(size) for size in args.sizes],
            edge_budgets=args.edge_budgets,
            seed=args.seed,
        )
    output_dir = args.output_dir or args.output
    if output_dir is None:
        parser.error("--output or --output-dir is required in single-subset mode")
    if args.target_nodes is None:
        parser.error("--target-nodes is required in single-subset mode")
    sample_relation_aware_subset(input_dir, output_dir, args.target_nodes, args.edge_budget, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
