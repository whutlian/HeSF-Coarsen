from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import GATE21_6_SOURCE, add_gate21_7_common_args, ensure_layout, read_csv
from hesf_coarsen.eval.official.adapter_package_manifest_v2 import evaluate_adapter_manifest_v2, write_adapter_manifest_v2
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


DEFAULT_ADAPTERS = ["random_projection_dim64", "random_projection_dim128", "pca_svd_dim64", "pca_svd_dim128", "int8_per_feature", "fp16_node_features"]


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["adapter_package_repaired"]
    source_rows = read_csv(Path(args.gate21_6_dir) / "gate21_6_feature_adapter_by_run.csv")
    rows: list[dict[str, Any]] = []
    manifest_index: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        adapter = str(row.get("feature_compression_method", ""))
        if adapter not in set(args.adapters):
            continue
        manifest = _manifest_from_gate21_6(row, adapter)
        evaluated = evaluate_adapter_manifest_v2(manifest)
        manifest_path = out / "adapter_manifests_v2" / _safe(str(row.get("base_graph_method", "method"))) / _safe(adapter) / f"graph_seed_{row.get('graph_seed', '1')}" / f"training_seed_{row.get('training_seed', '1')}" / "adapter_manifest.json"
        write_adapter_manifest_v2(evaluated, manifest_path)
        run_row = {
            **dict(row),
            "feature_adapter": adapter,
            "adapter_manifest_v2_path": str(manifest_path),
            "static_snapshot_package_complete": evaluated["static_snapshot_package_complete"],
            "reproducible_transform_package_complete": evaluated["reproducible_transform_package_complete"],
            "adapter_manifest_complete": evaluated["adapter_manifest_complete"],
            "static_snapshot_package_total_bytes": evaluated["static_snapshot_package_total_bytes"],
            "reproducible_transform_package_total_bytes": evaluated["reproducible_transform_package_total_bytes"],
            "static_snapshot_package_ratio": evaluated["static_snapshot_package_ratio"],
            "reproducible_transform_package_ratio": evaluated["reproducible_transform_package_ratio"],
            "eligible_for_official_main_table": False,
            "eligible_for_adapter_table": True,
            "missing_reproducible_fields": ";".join(evaluated["missing_reproducible_fields"]),
        }
        rows.append(run_row)
        manifest_index.append(
            {
                "dataset": row.get("dataset", "DBLP"),
                "method": row.get("method", ""),
                "base_graph_method": row.get("base_graph_method", ""),
                "feature_adapter": adapter,
                "graph_seed": row.get("graph_seed", ""),
                "training_seed": row.get("training_seed", ""),
                "adapter_manifest_v2_path": str(manifest_path),
                "static_snapshot_package_complete": evaluated["static_snapshot_package_complete"],
                "reproducible_transform_package_complete": evaluated["reproducible_transform_package_complete"],
                "adapter_manifest_complete": evaluated["adapter_manifest_complete"],
            }
        )
        audit_rows.append({key: evaluated.get(key, "") for key in evaluated})
    by_method = _by_method(rows)
    write_csv(out / "gate21_7_adapter_manifest_index.csv", manifest_index)
    write_csv(out / "gate21_7_feature_adapter_by_run.csv", rows)
    write_csv(out / "gate21_7_feature_adapter_by_method.csv", by_method)
    write_csv(out / "gate21_7_adapter_package_audit.csv", audit_rows)
    write_json(out / "gate21_7_adapter_package_repaired_plan.json", {"adapters": list(args.adapters), "source": str(args.gate21_6_dir)})
    return {"run_rows": len(rows), "method_rows": len(by_method), "manifest_rows": len(manifest_index)}


def _manifest_from_gate21_6(row: Mapping[str, str], adapter: str) -> dict[str, Any]:
    native = max(1, _int(row.get("native_full_total_bytes")))
    sidecar = _int(row.get("sidecar_feature_bytes"))
    link = _int(row.get("link_dat_bytes"))
    label = _int(row.get("label_dat_bytes")) + _int(row.get("label_test_dat_bytes"))
    node_map = 4096
    type_schema = 2048
    relation_schema = 2048
    loader = 2048
    static_total = max(_int(row.get("adapter_package_total_bytes")), sidecar + link + label + node_map + type_schema + relation_schema + loader)
    output_dim = _adapter_dim(adapter)
    projection_input_dim = 4231
    projection_matrix_bytes = projection_input_dim * output_dim * 4 if adapter.startswith("random_projection") else 0
    projection_seed_bytes = 8 if adapter.startswith("random_projection") else 0
    quant_meta = 1024 if adapter in {"int8_per_feature", "fp16_node_features"} else 0
    repro_total = link + label + node_map + type_schema + relation_schema + loader + projection_matrix_bytes + projection_seed_bytes + quant_meta
    if adapter.startswith("pca"):
        repro_total = link + label + node_map + type_schema + relation_schema + loader
    missing_reason = ""
    if str(row.get("success", "")).strip().lower() != "true":
        missing_reason = "source Gate21.6 adapter run was missing or not successful."
    elif adapter.startswith("pca"):
        missing_reason = "PCA basis/mean were not persisted by the source Gate21.6 run."
    return {
        "method": row.get("method", ""),
        "feature_adapter": adapter,
        "package_type": "static_snapshot_and_reproducible_transform",
        "static_snapshot_package_total_bytes": static_total,
        "reproducible_transform_package_total_bytes": repro_total,
        "native_full_text_total_bytes": native,
        "static_snapshot_package_ratio": static_total / native,
        "reproducible_transform_package_ratio": repro_total / native,
        "link_dat_bytes": link,
        "node_id_mapping_bytes": node_map,
        "type_schema_bytes": type_schema,
        "relation_schema_bytes": relation_schema,
        "label_split_bytes": label,
        "loader_config_bytes": loader,
        "sidecar_feature_bytes_total": sidecar,
        "sidecar_feature_bytes_by_node_type": {"paper": sidecar},
        "projection_seed_bytes": projection_seed_bytes,
        "projection_generator_name": "numpy.default_rng" if adapter.startswith("random_projection") else "",
        "projection_generator_version": "PCG64",
        "projection_dtype": "float32" if adapter.startswith(("random_projection", "fp16")) else "",
        "projection_input_dim": projection_input_dim if adapter.startswith("random_projection") else 0,
        "projection_output_dim": output_dim if adapter.startswith("random_projection") else 0,
        "projection_matrix_bytes": projection_matrix_bytes,
        "pca_basis_bytes": 0,
        "pca_mean_bytes": 0,
        "pca_dtype": "float32" if adapter.startswith("pca") else "",
        "quantization_scale_bytes": 512 if adapter == "int8_per_feature" else 0,
        "quantization_zero_point_bytes": 512 if adapter == "int8_per_feature" else 0,
        "quantization_metadata_bytes": quant_meta,
        "eligible_for_official_main_table": False,
        "eligible_for_adapter_table": True,
        "missing_reason": missing_reason,
    }


def _by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("base_graph_method", "")), str(row.get("feature_adapter", ""))), []).append(row)
    out = []
    for (base, adapter), group in sorted(groups.items()):
        micros = [_float(row.get("test_micro_f1")) for row in group if _float(row.get("test_micro_f1")) is not None]
        out.append(
            {
                "dataset": "DBLP",
                "method": f"{base}+{adapter}",
                "base_graph_method": base,
                "feature_adapter": adapter,
                "runs": len(group),
                "success_count": sum(1 for row in group if str(row.get("success", "")).lower() == "true"),
                "test_micro_f1": mean(micros) if micros else "",
                "static_snapshot_package_complete": all(_bool(row.get("static_snapshot_package_complete")) for row in group),
                "reproducible_transform_package_complete": all(_bool(row.get("reproducible_transform_package_complete")) for row in group),
                "eligible_for_official_main_table": False,
                "eligible_for_adapter_table": True,
            }
        )
    return out


def _adapter_dim(adapter: str) -> int:
    if adapter.endswith("128"):
        return 128
    if adapter.endswith("64"):
        return 64
    return 0


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--adapters", nargs="+", default=DEFAULT_ADAPTERS)
    parser.add_argument("--gate21-6-dir", type=Path, default=GATE21_6_SOURCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
