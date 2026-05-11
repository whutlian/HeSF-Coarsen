from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hesf_coarsen.config import load_config
from hesf_coarsen.coarsen.aggregate_edges import coarsen_graph_chunked
from hesf_coarsen.coarsen.assignment import Assignment
from hesf_coarsen.coarsen.multilevel import discover_completed_levels, run_multilevel_coarsening
from hesf_coarsen.io.edge_list import generate_synthetic_graph, load_graph, save_graph
from hesf_coarsen.io.dataset_importers import import_hgb_dataset, import_ogbn_mag_dataset
from hesf_coarsen.io.memmap_csr import load_memmap_graph, memmap_summary, save_memmap_graph
from hesf_coarsen.io.schema import validate_schema


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_generate_synthetic(args: argparse.Namespace) -> None:
    graph = generate_synthetic_graph(
        num_users=args.num_users,
        num_items=args.num_items,
        num_tags=args.num_tags,
        seed=args.seed,
    )
    save_graph(graph, args.output)
    _print_json(
        {
            "output": str(args.output),
            "num_nodes": graph.num_nodes,
            "relations": {str(k): rel.num_edges for k, rel in graph.relations.items()},
            "seed": args.seed,
        }
    )


def cmd_coarsen(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.progress:
        config.setdefault("progress", {})["enabled"] = True
    if args.progress_backend is not None:
        config.setdefault("progress", {})["backend"] = args.progress_backend
    if args.progress_interval is not None:
        config.setdefault("progress", {})["min_interval_seconds"] = args.progress_interval
    if args.resume:
        config.setdefault("resume", {})["enabled"] = True
    if args.allow_legacy_checkpoints:
        config.setdefault("resume", {})["allow_legacy_checkpoints"] = True
    config.setdefault("output", {})["dir"] = str(args.output)
    resume_cfg = config.get("resume", {})
    completed_before = (
        discover_completed_levels(
            args.output,
            allow_legacy_checkpoints=bool(resume_cfg.get("allow_legacy_checkpoints", False)),
        )
        if bool(resume_cfg.get("enabled", False))
        else []
    )
    graph = load_graph(args.input)
    validate_schema(graph)
    results = run_multilevel_coarsening(graph, config)
    completed_after = discover_completed_levels(
        args.output,
        allow_legacy_checkpoints=bool(resume_cfg.get("allow_legacy_checkpoints", False)),
    )
    final_nodes = (
        results[-1].graph.num_nodes
        if results
        else completed_after[-1].num_nodes
        if completed_after
        else graph.num_nodes
    )
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "levels": len(results),
        "completed_levels": len(completed_after),
        "resumed_from_level": completed_before[-1].level if completed_before else 0,
        "final_nodes": final_nodes,
    }
    if results:
        summary["last_level_diagnostics"] = results[-1].diagnostics
    _print_json(summary)


def cmd_diagnose(args: argparse.Namespace) -> None:
    graph = load_graph(args.input)
    diagnostics_path = Path(args.input) / "diagnostics.json"
    if diagnostics_path.exists():
        with diagnostics_path.open("r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
    else:
        validate_schema(graph)
        diagnostics = {
            "nodes": graph.num_nodes,
            "relations": {str(k): rel.num_edges for k, rel in graph.relations.items()},
        }
    _print_json(diagnostics)


def cmd_export_memmap(args: argparse.Namespace) -> None:
    graph = load_graph(args.input)
    save_memmap_graph(graph, args.output, chunk_size=args.chunk_size)
    _print_json(memmap_summary(args.output))


def _load_assignment(path: Path) -> Assignment:
    payload = np.load(path)
    return Assignment(
        assignment=payload["assignment"],
        supernode_type=payload["supernode_type"],
    )


def cmd_chunked_aggregate(args: argparse.Namespace) -> None:
    graph = load_memmap_graph(args.input) if args.memmap_input else load_graph(args.input)
    assignment = _load_assignment(args.assignment)
    coarse = coarsen_graph_chunked(
        graph,
        assignment,
        chunk_size=args.chunk_size,
        output_dir=args.output,
        reducer=args.reducer,
    )
    save_graph(coarse, args.output)
    _print_json(
        {
            "input": str(args.input),
            "output": str(args.output),
            "memmap_input": bool(args.memmap_input),
            "chunk_size": int(args.chunk_size),
            "reducer": args.reducer,
            "num_nodes": coarse.num_nodes,
            "relations": {str(k): rel.num_edges for k, rel in coarse.relations.items()},
        }
    )


def _graph_summary(graph, output: Path) -> dict:
    return {
        "output": str(output),
        "num_nodes": graph.num_nodes,
        "node_count_by_type": {
            str(int(type_id)): int((graph.node_type == type_id).sum())
            for type_id in sorted(np.unique(graph.node_type))
        },
        "relations": {
            str(relation_id): {
                "name": graph.relation_specs[relation_id].name,
                "edges": rel.num_edges,
                "src_type": rel.src_type,
                "dst_type": rel.dst_type,
            }
            for relation_id, rel in sorted(graph.relations.items())
        },
    }


def cmd_import_hgb(args: argparse.Namespace) -> None:
    graph = import_hgb_dataset(
        name=args.name,
        root=args.root,
        output=args.output,
        force_reload=args.force_reload,
    )
    _print_json(_graph_summary(graph, args.output))


def cmd_import_ogbn_mag(args: argparse.Namespace) -> None:
    graph = import_ogbn_mag_dataset(root=args.root, output=args.output)
    _print_json(_graph_summary(graph, args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hesf-coarsen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-synthetic")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--num-users", type=int, default=1000)
    generate.add_argument("--num-items", type=int, default=500)
    generate.add_argument("--num-tags", type=int, default=100)
    generate.add_argument("--seed", type=int, default=12345)
    generate.set_defaults(func=cmd_generate_synthetic)

    coarsen = subparsers.add_parser("coarsen")
    coarsen.add_argument("--config", type=Path, required=True)
    coarsen.add_argument("--input", type=Path, required=True)
    coarsen.add_argument("--output", type=Path, required=True)
    coarsen.add_argument("--progress", action="store_true", help="emit progress updates to stderr")
    coarsen.add_argument("--progress-backend", choices=["auto", "plain", "tqdm"])
    coarsen.add_argument("--progress-interval", type=float)
    coarsen.add_argument("--resume", action="store_true", help="continue from completed level outputs")
    coarsen.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help="treat loadable pre-checkpoint level outputs as completed when resuming",
    )
    coarsen.set_defaults(func=cmd_coarsen)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--input", type=Path, required=True)
    diagnose.set_defaults(func=cmd_diagnose)

    export_memmap = subparsers.add_parser("export-memmap")
    export_memmap.add_argument("--input", type=Path, required=True)
    export_memmap.add_argument("--output", type=Path, required=True)
    export_memmap.add_argument("--chunk-size", type=int, default=1_000_000)
    export_memmap.set_defaults(func=cmd_export_memmap)

    chunked = subparsers.add_parser("chunked-aggregate")
    chunked.add_argument("--input", type=Path, required=True)
    chunked.add_argument("--assignment", type=Path, required=True)
    chunked.add_argument("--output", type=Path, required=True)
    chunked.add_argument("--chunk-size", type=int, default=1_000_000)
    chunked.add_argument("--reducer", choices=["sort", "hash"], default="sort")
    chunked.add_argument("--memmap-input", action="store_true")
    chunked.set_defaults(func=cmd_chunked_aggregate)

    import_hgb = subparsers.add_parser("import-hgb")
    import_hgb.add_argument("--name", choices=["ACM", "DBLP", "IMDB", "Freebase"], required=True)
    import_hgb.add_argument("--root", type=Path, default=Path("data"))
    import_hgb.add_argument("--output", type=Path, required=True)
    import_hgb.add_argument("--force-reload", action="store_true")
    import_hgb.set_defaults(func=cmd_import_hgb)

    import_mag = subparsers.add_parser("import-ogbn-mag")
    import_mag.add_argument("--root", type=Path, default=Path("data/ogb"))
    import_mag.add_argument("--output", type=Path, required=True)
    import_mag.set_defaults(func=cmd_import_ogbn_mag)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
