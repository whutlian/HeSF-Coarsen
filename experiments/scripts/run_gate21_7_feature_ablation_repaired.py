from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate21_7_common import GATE21_6_SOURCE, add_gate21_7_common_args, ensure_layout, read_csv
from hesf_coarsen.eval.official.feature_ablation_repaired import REQUIRED_FEATURE_TRANSFORMS, apply_repaired_feature_transform
from hesf_coarsen.eval.official.runner_utils import write_csv, write_json
from hesf_coarsen.io.schema import HeteroGraph


DEFAULT_METHODS = ["full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"]
METHOD_ALIASES = {
    "full": "H6-APV-skeleton",
    "HeSF-RCS-APV12": "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
    "HeSF-RCS-APV16": "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00",
}
LABEL_GRAPH_SETTINGS = [
    "default",
    "no_label_feats",
    "num_label_hops_0",
    "num_feature_hops_0",
    "no_label_feats+zero-all-support",
    "feature_only_mlp_if_supported",
]


def run(args: argparse.Namespace) -> dict[str, int]:
    paths = ensure_layout(Path(args.output_root))
    out = paths["feature_ablation_repaired"]
    gate5 = Path(args.gate21_5_dir)
    cache_rows = read_csv(gate5 / "gate21_5_cache_audit.csv")
    loader_rows = read_csv(gate5 / "gate21_5_feature_loader_audit.csv")
    result_rows: list[dict[str, Any]] = []
    transform_audit_rows: list[dict[str, Any]] = []
    assertion_rows: list[dict[str, Any]] = []
    for method in args.methods:
        source_method = METHOD_ALIASES.get(str(method), str(method))
        cache_row = next((row for row in cache_rows if row.get("method") == source_method and row.get("training_seed") == "1"), None)
        if cache_row is None:
            result_rows.append(_failure_row(method, "missing_export_dir", "No HGB export found for repaired feature ablation."))
            continue
        graph = _representative_feature_graph_from_hgb(Path(cache_row.get("export_dir", "")))
        for transform_name in REQUIRED_FEATURE_TRANSFORMS:
            try:
                transformed = apply_repaired_feature_transform(graph, transform_name, seed=int(args.seeds[0]))
                audit = dict(transformed._gate21_7_transform_audit)
                result_rows.append(
                    {
                        "dataset": str(args.dataset).upper(),
                        "method": method,
                        "feature_setting": transform_name,
                        "label_graph_setting": "default",
                        "training_executed": False,
                        "test_micro_f1": "",
                        "test_macro_f1": "",
                        "success": False,
                        "failure_type": "shape_audit_only",
                        "failure_message": "Gate21.7 repaired transform was shape-audited; official SeHGNN retraining is not claimed by this row.",
                        "shape_safe_pass": bool(audit["FEATURE_ABLATION_SHAPE_SAFE_PASS"]),
                        "label_graph_setting_ready": True,
                        "eligible_for_mechanism_claim": False,
                        "shape_audit_scope": "representative_shape_from_hgb_node_dat",
                    }
                )
                transform_audit_rows.append(
                    {
                        "dataset": str(args.dataset).upper(),
                        "method": method,
                        "feature_transform_name": transform_name,
                        **{key: audit.get(key, "") for key in audit},
                    }
                )
                for assertion in audit["shape_assertion_rows"]:
                    assertion_rows.append({"dataset": str(args.dataset).upper(), "method": method, **dict(assertion)})
            except Exception as exc:
                result_rows.append(_failure_row(method, "feature_transform_failure", str(exc), transform_name=transform_name))
        for label_setting in LABEL_GRAPH_SETTINGS:
            if label_setting == "default":
                continue
            result_rows.append(
                {
                    "dataset": str(args.dataset).upper(),
                    "method": method,
                    "feature_setting": "raw",
                    "label_graph_setting": label_setting,
                    "success": False,
                    "training_executed": False,
                    "failure_type": "unsupported_by_official_pipeline",
                    "failure_message": f"{label_setting} is recorded as an explicit official-pipeline unsupported setting.",
                    "shape_safe_pass": True,
                    "label_graph_setting_ready": True,
                    "eligible_for_mechanism_claim": False,
                }
            )
    write_csv(out / "gate21_7_feature_ablation_repaired.csv", result_rows)
    write_csv(out / "gate21_7_feature_loader_audit_repaired.csv", loader_rows)
    write_csv(out / "gate21_7_feature_transform_audit_repaired.csv", transform_audit_rows)
    write_csv(out / "gate21_7_feature_shape_assertions.csv", assertion_rows)
    write_json(out / "gate21_7_feature_ablation_repaired_plan.json", {"methods": list(args.methods), "feature_transforms": REQUIRED_FEATURE_TRANSFORMS, "label_graph_settings": LABEL_GRAPH_SETTINGS})
    return {"feature_rows": len(result_rows), "transform_audit_rows": len(transform_audit_rows), "shape_assertion_rows": len(assertion_rows)}


def _representative_feature_graph_from_hgb(export_dir: Path) -> HeteroGraph:
    type_dims: dict[int, int] = {}
    type_counts: dict[int, int] = {}
    node_path = Path(export_dir) / "node.dat"
    with node_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            type_id = int(parts[2])
            type_counts[type_id] = type_counts.get(type_id, 0) + 1
            if type_id not in type_dims:
                type_dims[type_id] = len([item for item in (parts[3] if len(parts) >= 4 else "").split(",") if item != ""])
    capped_counts = {type_id: min(count, 4) for type_id, count in type_counts.items()}
    node_type = np.concatenate([np.full(count, type_id, dtype=np.int32) for type_id, count in sorted(capped_counts.items())])
    features = {
        type_id: np.arange(max(count * type_dims.get(type_id, 0), 0), dtype=np.float32).reshape(count, type_dims.get(type_id, 0))
        for type_id, count in capped_counts.items()
    }
    return HeteroGraph(num_nodes=int(node_type.size), node_type=node_type, relations={}, features=features)


def _failure_row(method: str, failure_type: str, failure_message: str, *, transform_name: str = "") -> dict[str, Any]:
    return {
        "dataset": "DBLP",
        "method": method,
        "feature_setting": transform_name,
        "label_graph_setting": "default",
        "success": False,
        "training_executed": False,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "shape_safe_pass": False,
        "eligible_for_mechanism_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = add_gate21_7_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--gate21-6-dir", type=Path, default=GATE21_6_SOURCE)
    parser.add_argument("--gate21-5-dir", type=Path, default=Path("results/gate21_5_directed_apv_feature_adapter"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
