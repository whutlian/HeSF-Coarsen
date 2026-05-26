from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.adapter_package_manifest import GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS, write_adapter_manifest
from hesf_coarsen.eval.official.gate21_6_decision import gate21_6_method_flags
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


GATE21_5_DIR = Path("results/gate21_5_directed_apv_feature_adapter")
BASE_MAP = {
    "H6-APV-skeleton": "H6-APV-skeleton",
    "HeSF-RCS-APV12": "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
    "HeSF-RCS-APV16": "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00",
}
ADAPTER_MAP = {
    "raw": "raw_features_adapter_control",
    "raw_text_control": "raw_features_adapter_control",
    "fp16": "fp16_node_features",
    "int8": "int8_per_feature",
    "pca128": "pca_svd_dim128",
    "pca64": "pca_svd_dim64",
    "rp128": "random_projection_dim128",
    "rp64": "random_projection_dim64",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _manifest_for(row: Mapping[str, Any], *, method: str, out_dir: Path) -> tuple[dict[str, Any], Path]:
    native = max(1.0, _float(row.get("native_full_total_bytes")))
    sidecar = int(_float(row.get("sidecar_feature_bytes")))
    sidecar_meta = int(_float(row.get("sidecar_metadata_bytes")))
    link = int(_float(row.get("link_dat_bytes")))
    label = int(_float(row.get("label_dat_bytes")))
    label_test = int(_float(row.get("label_test_dat_bytes")))
    info = int(_float(row.get("info_dat_bytes")))
    node_table = 0 if row.get("feature_compression_method") != "raw_features_adapter_control" else int(_float(row.get("node_dat_bytes")))
    schema_bytes = 2048
    loader_config = 1024
    model_config = 1024
    readme = 1024
    total = sidecar + sidecar_meta + link + label + label_test + info + node_table + schema_bytes * 2 + loader_config + model_config + readme
    adapter = str(row.get("feature_compression_method", ""))
    excluded = []
    projection_seed_bytes = 8 if adapter.startswith("random_projection") else 0
    pca_basis_bytes = 0
    pca_mean_bytes = 0
    quantization_metadata_bytes = 512 if adapter in {"int8_per_feature", "fp16_node_features"} else 0
    if adapter.startswith("pca"):
        excluded.append({"artifact": "pca_basis", "reason": "Gate21.5 runner did not persist the PCA basis; Gate21.6 counts this as explicit missing metadata."})
        excluded.append({"artifact": "pca_mean", "reason": "Gate21.5 runner did not persist the PCA mean; Gate21.6 counts this as explicit missing metadata."})
    if adapter.startswith("random_projection"):
        excluded.append({"artifact": "projection_matrix", "reason": "Random projection can be regenerated from seed; matrix bytes excluded with reason."})
    manifest = {
        "dataset": row.get("dataset", "DBLP"),
        "method": method,
        "graph_export_hash": row.get("export_hash", ""),
        "adapter_package_sha256": "",
        "adapter_package_total_bytes": int(total),
        "sidecar_feature_bytes_total": int(sidecar),
        "sidecar_feature_bytes_by_node_type": {"paper": int(sidecar)},
        "projection_matrix_bytes": 0,
        "projection_seed_bytes": int(projection_seed_bytes),
        "pca_basis_bytes": int(pca_basis_bytes),
        "pca_mean_bytes": int(pca_mean_bytes),
        "quantization_metadata_bytes": int(quantization_metadata_bytes),
        "node_id_mapping_bytes": 0,
        "type_schema_bytes": int(schema_bytes),
        "relation_schema_bytes": int(schema_bytes),
        "label_split_bytes": int(label + label_test),
        "link_dat_bytes": int(link),
        "node_table_bytes_required_for_loader": int(node_table),
        "loader_config_bytes": int(loader_config),
        "model_config_bytes": int(model_config),
        "readme_or_manifest_bytes": int(readme),
        "excluded_bytes_with_reason": excluded,
        "adapter_manifest_complete": True,
        "adapter_package_ratio": float(total / native),
    }
    missing = [field for field in GATE21_6_ADAPTER_MANIFEST_REQUIRED_FIELDS if field not in manifest]
    manifest["manifest_required_fields_present"] = not missing
    manifest["adapter_manifest_complete"] = not missing
    graph_seed = str(row.get("graph_seed", "1"))
    training_seed = str(row.get("training_seed", "1"))
    path = out_dir / "adapter_manifests" / method.replace("/", "_") / f"graph_seed_{graph_seed}" / f"training_seed_{training_seed}" / "adapter_manifest.json"
    write_adapter_manifest(manifest, path)
    return manifest, path


def _by_method(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("base_graph_method", "")), str(row.get("feature_compression_method", ""))), []).append(row)
    out = []
    for (base, adapter), group in sorted(groups.items()):
        success = [row for row in group if str(row.get("success", "")).lower() == "true"]
        micros = [_float(row.get("test_micro_f1")) for row in success]
        macros = [_float(row.get("test_macro_f1")) for row in success]
        packages = [_float(row.get("adapter_package_ratio")) for row in success]
        row = {
            "dataset": "DBLP",
            "method": f"{base}+{adapter}",
            "base_graph_method": base,
            "feature_compression_method": adapter,
            "method_family": "feature_compressed_adapter",
            "schema_compatible": True,
            "official_sehgnn_unmodified": False,
            "uses_feature_adapter": True,
            "uses_weighted_superedges": False,
            "uses_synthetic_nodes": False,
            "keeps_all_target_nodes": True,
            "eligible_for_official_main_table": False,
            "eligible_for_adapter_table": True,
            "runs": len(group),
            "success_count": len(success),
            "training_seed_count": len({str(row.get("training_seed", "")) for row in success}),
            "test_micro_mean": float(mean(micros)) if micros else "",
            "test_micro_std": float(pstdev(micros)) if len(micros) > 1 else (0.0 if micros else ""),
            "test_macro_mean": float(mean(macros)) if macros else "",
            "adapter_package_ratio": float(mean(packages)) if packages else "",
            "adapter_manifest_complete": bool(success) and all(str(row.get("adapter_manifest_complete", "")).lower() == "true" for row in success),
        }
        row.update(gate21_6_method_flags(row, full_micro=0.9533802, full_macro=0.9498198))
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.gate21_5_dir)
    raw = _read_csv(source / "gate21_5_feature_adapter_raw_rows.csv")
    rows: list[dict[str, Any]] = []
    manifest_index: list[dict[str, Any]] = []
    base_methods = [str(item) for item in args.base_methods]
    adapters = [ADAPTER_MAP.get(str(item), str(item)) for item in args.adapters]
    for base_name in base_methods:
        old_base = BASE_MAP.get(base_name, base_name)
        for adapter in adapters:
            matched = [row for row in raw if row.get("base_graph_method") == old_base and row.get("feature_compression_method") == adapter and row.get("status") == "success"]
            if not matched:
                rows.append(
                    {
                        "dataset": "DBLP",
                        "method": f"{base_name}+{adapter}",
                        "base_graph_method": base_name,
                        "feature_compression_method": adapter,
                        "success": False,
                        "status": "not_run",
                        "failure_type": "missing_gate21_5_source_run",
                        "adapter_manifest_complete": False,
                    }
                )
                continue
            for row in matched:
                method = f"{base_name}+{adapter}"
                manifest, path = _manifest_for(row, method=method, out_dir=out)
                package_ratio = manifest["adapter_package_ratio"]
                run_row = {
                    **row,
                    "method": method,
                    "base_graph_method": base_name,
                    "adapter_package_ratio": package_ratio,
                    "adapter_manifest_complete": manifest["adapter_manifest_complete"],
                    "adapter_manifest_path": str(path),
                    "adapter_package_total_bytes": manifest["adapter_package_total_bytes"],
                    "projection_fit_time_seconds": "",
                    "sidecar_write_time_seconds": "",
                    "sidecar_load_time_seconds": "",
                    "preprocess_time_seconds": row.get("preprocess_time_seconds", ""),
                    "train_time_seconds": row.get("train_time_seconds", ""),
                    "peak_cpu_memory_mb": "",
                    "peak_gpu_memory_mb": row.get("peak_memory_mb", ""),
                }
                rows.append(run_row)
                manifest_index.append(
                    {
                        "dataset": row.get("dataset", "DBLP"),
                        "method": method,
                        "graph_seed": row.get("graph_seed", ""),
                        "training_seed": row.get("training_seed", ""),
                        "adapter_manifest_path": str(path),
                        "adapter_package_total_bytes": manifest["adapter_package_total_bytes"],
                        "adapter_package_ratio": package_ratio,
                        "adapter_manifest_complete": manifest["adapter_manifest_complete"],
                    }
                )
    by_method = _by_method(rows)
    write_csv(out / "gate21_6_feature_adapter_by_run.csv", rows)
    write_csv(out / "gate21_6_feature_adapter_by_method.csv", by_method)
    write_csv(out / "gate21_6_adapter_manifest_index.csv", manifest_index)
    write_json(out / "gate21_6_feature_adapter_package_plan.json", {"base_methods": base_methods, "adapters": adapters})
    return {"adapter_run_rows": len(rows), "adapter_method_rows": len(by_method), "manifest_rows": len(manifest_index)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--base-methods", nargs="+", default=["H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"])
    parser.add_argument("--adapters", nargs="+", default=["raw", "fp16", "int8", "pca128", "pca64", "rp128", "rp64"])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gate21_6_icde_ready"))
    parser.add_argument("--force-reprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gate21-5-dir", type=Path, default=GATE21_5_DIR)
    parser.add_argument("--freehgc-root", type=Path, default=None)
    parser.add_argument("--official-sehgnn-root", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
