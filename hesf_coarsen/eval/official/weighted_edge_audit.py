from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from hesf_coarsen.eval.official.runner_utils import write_csv
from hesf_coarsen.eval.official.sehgnn_hgb_format import supported_sehgnn_hgb_dataset


def _read_link_stats(link_dat: Path) -> dict[str, Any]:
    count_by_relation: dict[str, int] = defaultdict(int)
    sum_by_relation: dict[str, float] = defaultdict(float)
    min_by_relation: dict[str, float] = {}
    max_by_relation: dict[str, float] = {}
    nonunit = 0
    total = 0
    with Path(link_dat).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            relation = str(int(parts[2]))
            weight = float(parts[3])
            total += 1
            count_by_relation[relation] += 1
            sum_by_relation[relation] += weight
            min_by_relation[relation] = min(weight, min_by_relation.get(relation, weight))
            max_by_relation[relation] = max(weight, max_by_relation.get(relation, weight))
            if abs(weight - 1.0) > 1.0e-8:
                nonunit += 1
    mean_by_relation = {
        key: float(sum_by_relation[key] / max(count_by_relation[key], 1))
        for key in count_by_relation
    }
    return {
        "exported_edge_count": int(total),
        "exported_link_weight_nonunit_count": int(nonunit),
        "exported_link_weight_nonunit_fraction": float(nonunit / max(total, 1)),
        "exported_weight_sum_by_relation": dict(sorted(sum_by_relation.items())),
        "exported_edge_count_by_relation": dict(sorted(count_by_relation.items())),
        "exported_weight_min_by_relation": dict(sorted(min_by_relation.items())),
        "exported_weight_max_by_relation": dict(sorted(max_by_relation.items())),
        "exported_weight_mean_by_relation": dict(sorted(mean_by_relation.items())),
    }


def _load_data_loader_weight_sums(export_dir: Path, sehgnn_repo_dir: Path) -> dict[str, float]:
    data_dir = Path(sehgnn_repo_dir) / "data"
    if not (data_dir / "data_loader.py").exists():
        return {}
    sys.path.insert(0, str(data_dir))
    try:
        from data_loader import data_loader  # type: ignore

        dl = data_loader(str(export_dir))
        return {str(int(rid)): float(mat.sum()) for rid, mat in dl.links["data"].items()}
    finally:
        try:
            sys.path.remove(str(data_dir))
        except ValueError:
            pass


def _official_utils_uses_sparse_values(sehgnn_repo_dir: Path) -> bool:
    utils_path = Path(sehgnn_repo_dir) / "hgb" / "utils.py"
    if not utils_path.exists():
        return False
    text = utils_path.read_text(encoding="utf-8", errors="ignore")
    marker = "SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)"
    if marker in text:
        return False
    return "SparseTensor(" in text and ("value=" in text or "value =" in text)


def audit_sehgnn_edge_weight_semantics(
    *,
    export_dir: Path,
    dataset_name: str,
    sehgnn_repo_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = supported_sehgnn_hgb_dataset(dataset_name)
    export_dir = Path(export_dir)
    output_dir = Path(output_dir)
    stats = _read_link_stats(export_dir / "link.dat")
    data_loader_weight_sum = _load_data_loader_weight_sums(export_dir, Path(sehgnn_repo_dir))
    official_uses_values = _official_utils_uses_sparse_values(Path(sehgnn_repo_dir))
    loaded_sparse_sum = (
        data_loader_weight_sum
        if official_uses_values
        else {key: float(value) for key, value in stats["exported_edge_count_by_relation"].items()}
    )
    exported_sums = {str(key): float(value) for key, value in stats["exported_weight_sum_by_relation"].items()}
    preserves = bool(official_uses_values and exported_sums == {str(k): float(v) for k, v in loaded_sparse_sum.items()})
    drops = bool(not official_uses_values and stats["exported_link_weight_nonunit_count"] > 0)
    row = {
        "dataset": dataset,
        "method": export_dir.parent.parent.name if export_dir.parent.parent.exists() else export_dir.name,
        "seed": "",
        "exported_edge_count": int(stats["exported_edge_count"]),
        "exported_link_weight_nonunit_count": int(stats["exported_link_weight_nonunit_count"]),
        "exported_link_weight_nonunit_fraction": float(stats["exported_link_weight_nonunit_fraction"]),
        "exported_weight_sum_by_relation": json.dumps(exported_sums, sort_keys=True),
        "exported_edge_count_by_relation": json.dumps(stats["exported_edge_count_by_relation"], sort_keys=True),
        "exported_weight_min_by_relation": json.dumps(stats["exported_weight_min_by_relation"], sort_keys=True),
        "exported_weight_max_by_relation": json.dumps(stats["exported_weight_max_by_relation"], sort_keys=True),
        "exported_weight_mean_by_relation": json.dumps(stats["exported_weight_mean_by_relation"], sort_keys=True),
        "loaded_weight_sum_by_relation": json.dumps(loaded_sparse_sum, sort_keys=True),
        "data_loader_weight_sum_by_relation": json.dumps(data_loader_weight_sum, sort_keys=True),
        "official_preprocess_accepts_edge_values": bool(official_uses_values),
        "official_preprocess_preserves_edge_values": bool(preserves),
        "official_preprocess_drops_edge_values": bool(drops),
        "weighted_superedge_main_table_allowed": bool(preserves),
        "recommendation": "Family B weighted coarse graphs are diagnostic only for unmodified official SeHGNN." if drops else "No nonunit edge weights observed or official path preserves values.",
    }
    write_csv(output_dir / "gate21_1_weighted_edge_audit.csv", [row])
    md = [
        "# Gate21.1 Weighted Edge Audit",
        "",
        f"- dataset: `{dataset}`",
        f"- exported_edge_count: `{row['exported_edge_count']}`",
        f"- exported_link_weight_nonunit_fraction: `{row['exported_link_weight_nonunit_fraction']}`",
        f"- official_preprocess_accepts_edge_values: `{row['official_preprocess_accepts_edge_values']}`",
        f"- official_preprocess_preserves_edge_values: `{row['official_preprocess_preserves_edge_values']}`",
        f"- official_preprocess_drops_edge_values: `{row['official_preprocess_drops_edge_values']}`",
        f"- weighted_superedge_main_table_allowed: `{row['weighted_superedge_main_table_allowed']}`",
        "",
        str(row["recommendation"]),
    ]
    (output_dir / "gate21_1_weighted_edge_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate21_1_weighted_edge_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return row
