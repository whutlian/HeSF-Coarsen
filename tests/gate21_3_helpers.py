from __future__ import annotations

from pathlib import Path

import numpy as np

from hesf_coarsen.io.schema import HeteroGraph, RelationAdj, RelationSpec


def tiny_dblp_graph() -> HeteroGraph:
    node_type = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    relations = {
        0: RelationAdj(np.array([0, 1, 0]), np.array([2, 3, 3]), np.ones(3, dtype=np.float32), 0, 1, 0),
        1: RelationAdj(np.array([2, 3, 3]), np.array([0, 1, 0]), np.ones(3, dtype=np.float32), 1, 0, 1),
        2: RelationAdj(np.array([2, 3, 2]), np.array([4, 5, 5]), np.ones(3, dtype=np.float32), 1, 2, 2),
        3: RelationAdj(np.array([2, 3, 3]), np.array([6, 7, 7]), np.ones(3, dtype=np.float32), 1, 3, 3),
        4: RelationAdj(np.array([4, 5, 5]), np.array([2, 3, 2]), np.ones(3, dtype=np.float32), 2, 1, 4),
        5: RelationAdj(np.array([6, 7, 7]), np.array([2, 3, 2]), np.ones(3, dtype=np.float32), 3, 1, 5),
    }
    relation_specs = {
        0: RelationSpec(0, "AP", 0, 1),
        1: RelationSpec(1, "PA", 1, 0),
        2: RelationSpec(2, "PT", 1, 2),
        3: RelationSpec(3, "PV", 1, 3),
        4: RelationSpec(4, "TP", 2, 1),
        5: RelationSpec(5, "VP", 3, 1),
    }
    return HeteroGraph(
        num_nodes=8,
        node_type=node_type,
        relations=relations,
        relation_specs=relation_specs,
        features={
            0: np.eye(2, dtype=np.float32),
            1: np.eye(2, dtype=np.float32),
            2: np.eye(2, dtype=np.float32),
            3: np.eye(2, dtype=np.float32),
        },
        labels=np.array([0, 1, -1, -1, -1, -1, -1, -1], dtype=np.int64),
    )


def write_minimal_hgb_dir(path: Path, *, relation_counts: dict[int, int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "node.dat").write_text(
        "\n".join(
            [
                "0\t0\t0\t1,0",
                "1\t1\t0\t0,1",
                "2\t2\t1\t1,0",
                "3\t3\t1\t0,1",
                "4\t4\t2\t1,0",
                "5\t5\t2\t0,1",
                "6\t6\t3\t1,0",
                "7\t7\t3\t0,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    link_lines: list[str] = []
    for relation_id, count in sorted(relation_counts.items()):
        for idx in range(int(count)):
            link_lines.append(f"{idx % 2}\t{2 + (idx % 2)}\t{relation_id}\t1.0")
    (path / "link.dat").write_text("\n".join(link_lines) + "\n", encoding="utf-8")
    (path / "label.dat").write_text("0\t0\t0\t0\n", encoding="utf-8")
    (path / "label.dat.test").write_text("1\t1\t0\t1\n", encoding="utf-8")
