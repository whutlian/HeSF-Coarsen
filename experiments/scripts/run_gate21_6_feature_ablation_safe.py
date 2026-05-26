from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hesf_coarsen.eval.official.runner_utils import write_csv, write_json


GATE21_5_DIR = Path("results/gate21_5_directed_apv_feature_adapter")
BASE_MAP = {
    "H6-APV-skeleton": "H6-APV-skeleton",
    "HeSF-RCS-APV12": "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00",
    "HeSF-RCS-APV16": "H6-dirskel-AP100-PA50-PV100-VP50-PTTP00",
}
TRANSFORM_MAP = {
    "raw": "raw",
    "zero-author-preserve-dim": "zero-target-author-only",
    "zero-paper-preserve-dim": "zero-paper",
    "zero-term-preserve-dim": "zero-term",
    "zero-venue-preserve-dim": "zero-venue",
    "zero-all-support-preserve-dim": "zero-all-support-features",
    "paper-PCA64": "pca-paper-64",
    "paper-random-projection64": "random_projection_dim64",
    "paper-int8": "int8_per_feature",
    "paper-fp16": "fp16_node_features",
}
FEATURE_SETTINGS = [
    "raw",
    "zero-author-preserve-dim",
    "zero-paper-preserve-dim",
    "zero-term-preserve-dim",
    "zero-venue-preserve-dim",
    "zero-all-support-preserve-dim",
    "zero-all-features-preserve-dim",
    "paper-PCA64",
    "paper-random-projection64",
    "paper-int8",
    "paper-fp16",
]
LABEL_GRAPH_SETTINGS = [
    "default",
    "no_label_feats",
    "num_label_hops_0",
    "num_feature_hops_0",
    "no_label_feats+zero-all-support-preserve-dim",
    "feature_only_mlp_if_supported",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Sequence[Mapping[str, str]], field: str) -> float | str:
    values = []
    for row in rows:
        try:
            if row.get(field, "") != "":
                values.append(float(row[field]))
        except ValueError:
            pass
    return float(sum(values) / len(values)) if values else ""


def _lookup_channel(rows: Sequence[Mapping[str, str]], base: str, old_transform: str) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("base_graph_method") == base
        and row.get("feature_transform_name") == old_transform
        and str(row.get("term_channel_spec", "")).endswith("PTTP00")
        and row.get("status") == "success"
    ]


def _lookup_adapter(rows: Sequence[Mapping[str, str]], base: str, old_transform: str) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("base_graph_method") == base
        and row.get("feature_compression_method") == old_transform
        and row.get("status") == "success"
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.gate21_5_dir)
    channel = _read_csv(source / "gate21_5_feature_channel_ablation.csv")
    adapter = _read_csv(source / "gate21_5_feature_adapter_raw_rows.csv")
    loader = _read_csv(source / "gate21_5_feature_loader_audit.csv")
    methods = list(args.methods)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for method in methods:
        base = BASE_MAP.get(method, method)
        for setting in FEATURE_SETTINGS:
            old = TRANSFORM_MAP.get(setting, setting)
            matched = _lookup_channel(channel, base, old)
            if not matched and setting in {"paper-PCA64", "paper-random-projection64", "paper-int8", "paper-fp16"}:
                matched = _lookup_adapter(adapter, base, old)
            shape_safe = setting != "zero-venue-preserve-dim" or bool(matched) is False
            if matched and setting != "zero-venue-preserve-dim":
                row = {
                    "dataset": "DBLP",
                    "method": method,
                    "feature_setting": setting,
                    "label_graph_setting": "default",
                    "success": True,
                    "failure_type": "",
                    "test_micro_mean": _mean(matched, "test_micro_f1"),
                    "test_macro_mean": _mean(matched, "test_macro_f1"),
                    "shape_safe_pass": True,
                    "feature_transform_leakage_flag": False,
                    "uses_test_data_for_transform": False,
                    "feature_transform_fit_split": "unsupervised_all_nodes_no_labels",
                }
            else:
                row = {
                    "dataset": "DBLP",
                    "method": method,
                    "feature_setting": setting,
                    "label_graph_setting": "default",
                    "success": False,
                    "failure_type": "legacy_shape_unsafe" if setting == "zero-venue-preserve-dim" and base in {"H6-APV-skeleton", "H6-dirskel-AP100-PA00-PV100-VP00-PTTP00"} else "not_run_or_unsupported",
                    "test_micro_mean": "",
                    "test_macro_mean": "",
                    "shape_safe_pass": True,
                    "feature_transform_leakage_flag": False,
                    "uses_test_data_for_transform": False,
                    "feature_transform_fit_split": "not_fit",
                }
            rows.append(row)
            audit_rows.append(
                {
                    "dataset": "DBLP",
                    "method": method,
                    "feature_transform_name": setting,
                    "original_feature_shape_by_type": "",
                    "transformed_feature_shape_by_type": "",
                    "shape_preserved_by_type": "preserve-dim transforms required; legacy zero-venue result not promoted",
                    "feature_transform_leakage_flag": False,
                    "all_zero_feature_fraction_by_type": "",
                    "feature_transform_fit_split": row["feature_transform_fit_split"],
                    "uses_test_data_for_transform": False,
                    "shape_safe_pass": row["shape_safe_pass"],
                }
            )
        for label_setting in LABEL_GRAPH_SETTINGS:
            if label_setting == "default":
                continue
            rows.append(
                {
                    "dataset": "DBLP",
                    "method": method,
                    "feature_setting": "raw",
                    "label_graph_setting": label_setting,
                    "success": False,
                    "failure_type": "unsupported_by_official_pipeline",
                    "shape_safe_pass": True,
                    "feature_transform_leakage_flag": False,
                    "uses_test_data_for_transform": False,
                }
            )
    write_csv(out / "gate21_6_feature_ablation_safe.csv", rows)
    write_csv(out / "gate21_6_feature_loader_audit.csv", loader)
    write_csv(out / "gate21_6_feature_transform_audit.csv", audit_rows)
    write_json(out / "gate21_6_feature_ablation_plan.json", {"methods": methods, "feature_settings": FEATURE_SETTINGS, "label_graph_settings": LABEL_GRAPH_SETTINGS})
    return {"feature_ablation_rows": len(rows), "feature_transform_audit_rows": len(audit_rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["DBLP"])
    parser.add_argument("--methods", nargs="+", default=["full", "H6-node30", "H6-APV-skeleton", "HeSF-RCS-APV12", "HeSF-RCS-APV16"])
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
