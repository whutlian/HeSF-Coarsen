from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.candidates.array_store import ArrayCandidateStore
from hesf_coarsen.io.edge_list import generate_synthetic_graph
from hesf_coarsen.task_first.config import TaskFirstConfig
from hesf_coarsen.task_first.pipeline import build_support_only_task_first_coarsening


def run_smoke(output: Path, seed: int = 12345) -> dict:
    graph = generate_synthetic_graph(num_users=8, num_items=8, num_tags=5, seed=seed)
    labels = np.asarray(graph.labels)
    train_mask = np.zeros(graph.num_nodes, dtype=bool)
    target_nodes = np.flatnonzero(graph.node_type == 0).astype(np.int64)
    train_mask[target_nodes[: max(2, len(target_nodes) // 2)]] = True
    cfg = TaskFirstConfig(target_node_type=0)
    store = ArrayCandidateStore(graph.node_type, K=8, same_type_only=True)
    for type_id in sorted(set(int(value) for value in graph.node_type.tolist()) - {0}):
        nodes = np.flatnonzero(graph.node_type == type_id).astype(np.int64)
        for left, right in zip(nodes[::2], nodes[1::2]):
            store.add(int(left), int(right), 0.0, "smoke")
    result = build_support_only_task_first_coarsening(graph, store, labels, train_mask, cfg)
    payload = {
        "original_nodes": int(graph.num_nodes),
        "coarse_nodes": int(result.graph.num_nodes),
        "target_nodes": int(len(target_nodes)),
        "diagnostics": result.diagnostics,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny HeSF-TC support-only smoke test.")
    parser.add_argument("--output", type=Path, default=Path("outputs/task_first_smoke"))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    payload = run_smoke(args.output, seed=int(args.seed))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
