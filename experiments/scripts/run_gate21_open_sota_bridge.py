from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.gate13_task_first_common import load_hgb_graph, run_support_baseline
from experiments.scripts.run_gate17_1_support_sensitivity import _target_only_empty_support_graph
from experiments.scripts.summarize_gate21_open_sota import summarize
from hesf_coarsen.eval.hettree_task import infer_target_node_type
from hesf_coarsen.eval.official.calibration_adapter import calibrate_logits_nested
from hesf_coarsen.eval.official.graph_export import export_hgb_graph
from hesf_coarsen.eval.official.metrics import (
    classification_metrics_from_logits,
    confusion_rows,
    per_class_metric_rows,
)
from hesf_coarsen.eval.official.openhgnn_bridge import run_openhgnn_model
from hesf_coarsen.eval.official.runner_utils import (
    clone_external_repo,
    dependency_snapshot,
    git_commit_hash,
    write_csv,
    write_json,
)
from hesf_coarsen.eval.official.sehgnn_bridge import run_sehgnn_official
from hesf_coarsen.eval.task_gnn import select_task_protocol_split
from hesf_coarsen.io.schema import HeteroGraph, nodes_of_type


DEFAULT_DATASET_SEEDS = {"DBLP": (23456, 56789), "ACM": (23456,), "IMDB": (45678,)}
METHOD_TO_BASELINE = {
    "H6": "H6-no-spec-support-only",
    "flatten": "flatten-sum-support-only",
    "typedhash": "TypedHash-ChebHeat-support-only",
}
CAL_METHOD_NAMES = {"H6": "HeSF-CAL-H6", "flatten": "HeSF-CAL-flatten", "typedhash": "HeSF-CAL-TypedHash"}
MODEL_NAMES = {"sehgnn_official", "openhgnn_sehgnn", "openhgnn_hgt", "openhgnn_simplehgn"}
SEHGNN_URL = "https://github.com/ICT-GIMLab/SeHGNN.git"
OPENHGNN_URL = "https://github.com/BUPT-GAMMA/OpenHGNN.git"
CALIBRATION_FIELDS = (
    "dataset",
    "seed",
    "model_name",
    "method",
    "support_ratio",
    "temperature",
    "class_bias_vector",
    "delta_macro_from_calibration",
    "delta_accuracy_from_calibration",
    "ece_before",
    "ece_after",
    "nll_before",
    "nll_after",
    "brier_before",
    "brier_after",
    "constraint_satisfied_rate",
    "calibration_uses_test_labels",
)
PER_CLASS_FIELDS = (
    "dataset",
    "model",
    "method",
    "seed",
    "ratio",
    "calibrated",
    "class_id",
    "precision",
    "recall",
    "f1",
    "support",
    "delta_precision_vs_uncalibrated",
    "delta_recall_vs_uncalibrated",
    "delta_f1_vs_uncalibrated",
)
CONFUSION_FIELDS = ("dataset", "model", "method", "seed", "ratio", "calibrated", "true_class", "pred_class", "count")


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_dataset_seeds(values: Sequence[str] | None) -> list[tuple[str, int]]:
    if not values:
        return [(dataset, seed) for dataset, seeds in DEFAULT_DATASET_SEEDS.items() for seed in seeds]
    pairs: list[tuple[str, int]] = []
    current_dataset = ""
    for raw in values:
        for token in str(raw).replace(",", " ").split():
            if ":" in token:
                dataset, seed_text = token.split(":", 1)
                current_dataset = dataset.strip().upper()
                if seed_text:
                    pairs.append((current_dataset, int(seed_text)))
            elif current_dataset:
                pairs.append((current_dataset, int(token)))
            else:
                raise ValueError(f"dataset seed token must start with DATASET:SEED, got {token!r}")
    return pairs


def _edge_count(graph: HeteroGraph) -> int:
    return int(sum(rel.num_edges for rel in graph.relations.values()))


def _support_count(graph: HeteroGraph, target_type: int) -> int:
    return int(np.sum(np.asarray(graph.node_type) != int(target_type)))


def _remote_available(url: str) -> dict[str, Any]:
    completed = subprocess.run(["git", "ls-remote", "--heads", url], text=True, capture_output=True, check=False)
    first = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    return {
        "url": url,
        "available": bool(completed.returncode == 0 and first),
        "returncode": int(completed.returncode),
        "first_head": first,
        "stderr": completed.stderr.strip(),
    }


def _cost_fields(original: HeteroGraph, graph: HeteroGraph, *, target_type: int, requested_ratio: float | None) -> dict[str, Any]:
    original_support = _support_count(original, target_type)
    support = _support_count(graph, target_type)
    original_edges = _edge_count(original)
    edges = _edge_count(graph)
    original_storage = max(int(original.num_nodes) + int(original_edges), 1)
    storage = int(graph.num_nodes) + int(edges)
    return {
        "support_ratio_requested": "" if requested_ratio is None else float(requested_ratio),
        "support_ratio_realized": float(support / max(original_support, 1)),
        "support_node_ratio": float(support / max(original_support, 1)),
        "support_edge_ratio": float(edges / max(original_edges, 1)),
        "total_storage_ratio_vs_full_graph": float(storage / original_storage),
        "total_storage_ratio_vs_full_stc": float(storage / original_storage),
    }


def dry_run_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    support_ratio: float | None,
    model: str,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "stage": "Gate21-OpenSOTA",
        "dataset": dataset,
        "seed": int(seed),
        "model_name": model,
        "method": method,
        "support_ratio": "" if support_ratio is None else float(support_ratio),
        "primary_eval_mode": "compressed_projected",
        "validation_macro_f1": "",
        "validation_micro_f1": "",
        "validation_accuracy": "",
        "test_macro_f1": "",
        "test_micro_f1": "",
        "test_accuracy": "",
        "val_logits_path": "",
        "test_logits_path": "",
        "status": status,
        "error_message": error_message,
        "calibrated": False,
        "calibration_uses_test_labels": False,
        "selector_uses_test_labels": False,
        "uses_hettree_lite": False,
    }


def _build_method_graph(
    original: HeteroGraph,
    *,
    method: str,
    ratio: float | None,
    seed: int,
    candidate_k: int,
    target_type: int,
) -> tuple[HeteroGraph, np.ndarray, dict[str, Any]]:
    if method == "full":
        assignment = np.arange(original.num_nodes, dtype=np.int64)
        return original, assignment, {"method_build_status": "success", "method_source": "original_full_graph"}
    if method == "target-only":
        graph, assignment = _target_only_empty_support_graph(original, int(target_type))
        return graph, np.asarray(assignment, dtype=np.int64), {"method_build_status": "success", "method_source": "target_only_empty_support"}
    if method in METHOD_TO_BASELINE:
        graph, assignment, diag = run_support_baseline(
            original,
            baseline=METHOD_TO_BASELINE[method],
            ratio=float(ratio if ratio is not None else 0.30),
            seed=int(seed),
            candidate_k=int(candidate_k),
        )
        return graph, np.asarray(assignment, dtype=np.int64), {"method_build_status": "success", **diag}
    raise ValueError(f"unsupported Gate21 method: {method}")


def _run_model(
    *,
    model: str,
    export_dir: Path,
    dataset: str,
    seed: int,
    target_type: str,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if model == "sehgnn_official":
        return run_sehgnn_official(export_dir, Path(args.sehgnn_repo), dataset, target_type, int(seed), config, Path(args.output_dir))
    if model.startswith("openhgnn_"):
        return run_openhgnn_model(export_dir, Path(args.openhgnn_repo), model, dataset, target_type, int(seed), config, Path(args.output_dir))
    raise ValueError(f"unsupported model: {model}")


def _maybe_calibrate(
    row: Mapping[str, Any],
    *,
    export_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if str(row.get("status")) != "success" or not row.get("val_logits_path") or not row.get("test_logits_path"):
        return [], [], [], []
    val_logits = np.load(str(row["val_logits_path"]))
    test_logits = np.load(str(row["test_logits_path"]))
    labels = np.load(export_dir / "labels.npy")
    val_idx = np.load(export_dir / "splits" / "val_idx.npy")
    test_idx = np.load(export_dir / "splits" / "test_idx.npy")
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    cal = calibrate_logits_nested(val_logits, val_labels, test_logits)
    calibrated_test = np.asarray(cal["calibrated_test_logits"], dtype=np.float32)
    logits_dir = output_dir / "calibrated_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    token = f"{row.get('model_name')}_{row.get('dataset')}_{row.get('seed')}_{row.get('method')}_{row.get('support_ratio')}".replace(" ", "_").replace("/", "_")
    calibrated_path = logits_dir / f"{token}_test_logits.npy"
    np.save(calibrated_path, calibrated_test)
    test_scores = classification_metrics_from_logits(test_logits, test_labels)
    calibrated_scores = classification_metrics_from_logits(calibrated_test, test_labels)
    method = str(row.get("method"))
    calibrated_method = CAL_METHOD_NAMES.get(method, f"{method}-calibrated")
    raw_row = dict(row)
    raw_row.update(
        {
            "method": calibrated_method,
            "test_macro_f1": float(calibrated_scores["macro_f1"]),
            "test_micro_f1": float(calibrated_scores["micro_f1"]),
            "test_accuracy": float(calibrated_scores["accuracy"]),
            "uncalibrated_macro_f1": float(test_scores["macro_f1"]),
            "uncalibrated_accuracy": float(test_scores["accuracy"]),
            "delta_macro_from_calibration": float(calibrated_scores["macro_f1"] - test_scores["macro_f1"]),
            "delta_accuracy_from_calibration": float(calibrated_scores["accuracy"] - test_scores["accuracy"]),
            "calibrated": True,
            "calibrated_test_logits_path": str(calibrated_path),
            "calibration_uses_test_labels": False,
        }
    )
    cal_row = {
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "model_name": row.get("model_name", ""),
        "method": calibrated_method,
        "support_ratio": row.get("support_ratio", ""),
        "temperature": cal["best_temperature"],
        "class_bias_vector": cal["class_bias_vector"],
        "delta_macro_from_calibration": raw_row["delta_macro_from_calibration"],
        "delta_accuracy_from_calibration": raw_row["delta_accuracy_from_calibration"],
        "ece_before": cal["ece_before"],
        "ece_after": cal["ece_after"],
        "nll_before": cal["nll_before"],
        "nll_after": cal["nll_after"],
        "brier_before": cal["brier_before"],
        "brier_after": cal["brier_after"],
        "constraint_satisfied_rate": cal["constraint_satisfied_rate"],
        "calibration_uses_test_labels": False,
    }
    pred_uncal = np.asarray(test_scores["pred"], dtype=np.int64)
    pred_cal = np.asarray(calibrated_scores["pred"], dtype=np.int64)
    uncal_lookup = {
        int(item["class_id"]): item
        for item in per_class_metric_rows(
            test_labels,
            pred_uncal,
            dataset=str(row.get("dataset")),
            model=str(row.get("model_name")),
            method=str(row.get("method")),
            seed=int(row.get("seed")),
            ratio=None if row.get("support_ratio") in {"", None} else float(row.get("support_ratio")),
            calibrated=False,
        )
    }
    per_class = per_class_metric_rows(
        test_labels,
        pred_cal,
        dataset=str(row.get("dataset")),
        model=str(row.get("model_name")),
        method=calibrated_method,
        seed=int(row.get("seed")),
        ratio=None if row.get("support_ratio") in {"", None} else float(row.get("support_ratio")),
        calibrated=True,
        uncalibrated_lookup=uncal_lookup,
    )
    confusion = confusion_rows(
        test_labels,
        pred_cal,
        dataset=str(row.get("dataset")),
        model=str(row.get("model_name")),
        method=calibrated_method,
        seed=int(row.get("seed")),
        ratio=None if row.get("support_ratio") in {"", None} else float(row.get("support_ratio")),
        calibrated=True,
    )
    return [raw_row], [cal_row], per_class, confusion


def _write_reports(output_dir: Path, dep: Mapping[str, Any], dataset_status: Mapping[str, Any]) -> None:
    code_audit = [
        "# Gate21 OpenSOTA Code Audit",
        "",
        f"- HeSF-Coarsen commit: `{dep.get('hesf_coarsen_commit', '')}`",
        f"- Python: `{dep.get('python', '')}`",
        f"- torch: `{dep.get('torch_version', dep.get('torch_error', 'unavailable'))}`",
        f"- DGL: `{dep.get('dgl_version', dep.get('dgl_error', 'unavailable'))}`",
        f"- CUDA available: `{dep.get('cuda_available', '')}`",
        f"- SeHGNN repo path exists: `{dep.get('sehgnn_repo_exists')}`",
        f"- OpenHGNN repo path exists: `{dep.get('openhgnn_repo_exists')}`",
        f"- HETTREE status: `excluded_code_unavailable`",
        f"- dataset load status: `{dict(dataset_status)}`",
        "",
        "HETTREE excluded because open-source code is unavailable.",
        "SeHGNN official / OpenHGNN models are the active SOTA evaluator targets.",
        "Lite hettree results are diagnostic only.",
        "HeSF-CAL is evaluated as compressed support graph + validation-only calibration.",
    ]
    (output_dir / "code_audit.md").write_text("\n".join(code_audit) + "\n", encoding="utf-8")
    final_report = [
        "# Gate21 OpenSOTA Final Report",
        "",
        "HETTREE excluded because open-source code is unavailable.",
        "SeHGNN official / OpenHGNN models are the active SOTA evaluator targets.",
        "Lite hettree results are diagnostic only.",
        "HeSF-CAL is evaluated as compressed support graph + validation-only calibration.",
        "",
        "This bridge records missing external repositories or unsupported adapters as explicit failed runs; it does not fabricate official SOTA metrics.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(final_report) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if args.clone_external:
        if "sehgnn_official" in args.models:
            clone_external_repo(SEHGNN_URL, Path(args.sehgnn_repo))
        if any(str(model).startswith("openhgnn_") for model in args.models):
            clone_external_repo(OPENHGNN_URL, Path(args.openhgnn_repo))
    root = Path(__file__).resolve().parents[2]
    dep = dependency_snapshot(sehgnn_repo=Path(args.sehgnn_repo), openhgnn_repo=Path(args.openhgnn_repo), hesf_commit=git_commit_hash(root))
    dep["sehgnn_remote"] = _remote_available(SEHGNN_URL)
    dep["openhgnn_remote"] = _remote_available(OPENHGNN_URL)
    write_json(output_dir / "dependency_report.json", dep)
    write_json(diagnostics_dir / "gate21_dependency_report.json", dep)

    pairs = _parse_dataset_seeds(args.dataset_seeds)
    if args.max_datasets:
        keep = []
        seen: set[str] = set()
        for dataset, seed in pairs:
            if dataset not in seen and len(seen) >= int(args.max_datasets):
                continue
            seen.add(dataset)
            keep.append((dataset, seed))
        pairs = keep
    methods = [str(method) for method in args.methods]
    models = [str(model) for model in args.models if str(model) not in set(args.skip_models or [])]
    ratios = [float(value) for value in args.support_ratios]

    raw_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    per_class_rows_out: list[dict[str, Any]] = []
    confusion_rows_out: list[dict[str, Any]] = []
    dataset_status: dict[str, Any] = {}
    model_runs = 0
    start = perf_counter()
    for dataset, seed in pairs:
        try:
            original = load_hgb_graph(Path(args.data_root), dataset)
            dataset_status[f"{dataset}:{seed}"] = "loaded"
        except Exception as exc:
            dataset_status[f"{dataset}:{seed}"] = f"failed_load:{exc}"
            if args.strict:
                raise
            continue
        labels = np.asarray(original.labels if original.labels is not None else np.full(original.num_nodes, -1), dtype=np.int64)
        target_type = infer_target_node_type(original)
        train_nodes, val_nodes, test_nodes, _split = select_task_protocol_split(
            original,
            labels,
            seed=int(seed),
            target_node_type=int(target_type),
        )
        for method in methods:
            method_ratios = [None] if method in {"full", "target-only"} else ratios
            for ratio in method_ratios:
                try:
                    graph, assignment, method_diag = _build_method_graph(
                        original,
                        method=method,
                        ratio=ratio,
                        seed=int(seed),
                        candidate_k=int(args.candidate_k),
                        target_type=int(target_type),
                    )
                    target_nodes = nodes_of_type(original, int(target_type))
                    coarse_target_ids = np.asarray(assignment[target_nodes], dtype=np.int64)
                    export = export_hgb_graph(
                        graph,
                        dataset_name=dataset,
                        method_name=method,
                        seed=int(seed),
                        support_ratio=ratio,
                        output_dir=output_dir,
                        target_type=f"type_{int(target_type)}",
                        train_idx=np.asarray(assignment[train_nodes], dtype=np.int64),
                        val_idx=np.asarray(assignment[val_nodes], dtype=np.int64),
                        test_idx=np.asarray(assignment[test_nodes], dtype=np.int64),
                        labels=graph.labels if graph.labels is not None else labels,
                        original_target_ids=coarse_target_ids,
                        metadata={
                            "primary_eval_mode": str(args.primary_eval_mode),
                            "gate21_method_source": method_diag.get("method_source", method_diag.get("method", "")),
                        },
                    )
                    export.update(_cost_fields(original, graph, target_type=int(target_type), requested_ratio=ratio))
                    export_rows.append(export)
                except Exception as exc:
                    export = {
                        "dataset": dataset,
                        "seed": int(seed),
                        "method": method,
                        "support_ratio": "" if ratio is None else float(ratio),
                        "export_dir": "",
                        "mapping_bijective": False,
                        "split_disjoint": False,
                        "no_test_label_export_leakage": False,
                        "export_status": "failed_export",
                        "error_message": str(exc),
                    }
                    export_rows.append(export)
                    if args.strict:
                        write_csv(diagnostics_dir / "gate21_hgb_export_audit.csv", export_rows)
                        raise
                    continue
                if args.export_only:
                    raw_rows.append(
                        dry_run_row(
                            dataset=dataset,
                            seed=int(seed),
                            method=method,
                            support_ratio=ratio,
                            model="export_only",
                            status="export_only",
                            error_message="model execution skipped by --export-only",
                        )
                    )
                    continue
                for model in models:
                    if args.max_runs is not None and model_runs >= int(args.max_runs):
                        continue
                    model_runs += 1
                    if args.dry_run:
                        row = dry_run_row(
                            dataset=dataset,
                            seed=int(seed),
                            method=method,
                            support_ratio=ratio,
                            model=model,
                            status="dry_run",
                            error_message="model execution skipped by --dry-run",
                        )
                    else:
                        row = _run_model(
                            model=model,
                            export_dir=Path(export["export_dir"]),
                            dataset=dataset,
                            seed=int(seed),
                            target_type=str(export["target_type"]),
                            args=args,
                            config={
                                "method": method,
                                "support_ratio": ratio,
                                "primary_eval_mode": str(args.primary_eval_mode),
                                "calibrate": bool(args.calibrate),
                            },
                        )
                    row.update(
                        {
                            "stage": "Gate21-OpenSOTA",
                            "primary_eval_mode": str(args.primary_eval_mode),
                            **_cost_fields(original, graph, target_type=int(target_type), requested_ratio=ratio),
                        }
                    )
                    raw_rows.append(row)
                    if args.calibrate:
                        extra_raw, cal_rows, per_rows, conf_rows = _maybe_calibrate(row, export_dir=Path(export["export_dir"]), output_dir=output_dir)
                        raw_rows.extend(extra_raw)
                        calibration_rows.extend(cal_rows)
                        per_class_rows_out.extend(per_rows)
                        confusion_rows_out.extend(conf_rows)
                    if args.strict and str(row.get("status")) not in {"success", "dry_run"}:
                        write_csv(output_dir / "gate21_raw_rows.csv", raw_rows)
                        write_csv(diagnostics_dir / "gate21_hgb_export_audit.csv", export_rows)
                        raise RuntimeError(f"requested model failed under --strict: {row.get('model_name')} {row.get('status')} {row.get('error_message')}")
        write_csv(output_dir / "gate21_raw_rows.csv", raw_rows)
        write_csv(diagnostics_dir / "gate21_hgb_export_audit.csv", export_rows)
        write_csv(diagnostics_dir / "gate21_calibration.csv", calibration_rows, fieldnames=CALIBRATION_FIELDS)
        write_csv(diagnostics_dir / "gate21_per_class_metrics.csv", per_class_rows_out, fieldnames=PER_CLASS_FIELDS)
        write_csv(diagnostics_dir / "gate21_confusion_matrix.csv", confusion_rows_out, fieldnames=CONFUSION_FIELDS)

    failures = [row for row in raw_rows if str(row.get("status")) != "success"]
    write_csv(output_dir / "gate21_raw_rows.csv", raw_rows)
    write_csv(diagnostics_dir / "gate21_hgb_export_audit.csv", export_rows)
    write_csv(diagnostics_dir / "gate21_calibration.csv", calibration_rows, fieldnames=CALIBRATION_FIELDS)
    write_csv(diagnostics_dir / "gate21_per_class_metrics.csv", per_class_rows_out, fieldnames=PER_CLASS_FIELDS)
    write_csv(diagnostics_dir / "gate21_confusion_matrix.csv", confusion_rows_out, fieldnames=CONFUSION_FIELDS)
    write_csv(diagnostics_dir / "gate21_failure_report.csv", failures)
    dep["dataset_load_status"] = dataset_status
    dep["total_wall_time_sec"] = float(perf_counter() - start)
    write_json(output_dir / "dependency_report.json", dep)
    write_json(diagnostics_dir / "gate21_dependency_report.json", dep)
    _write_reports(output_dir, dep, dataset_status)
    return summarize(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["DBLP", "ACM", "IMDB"])
    parser.add_argument("--dataset-seeds", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", default=["full", "target-only", "H6", "flatten", "typedhash"])
    parser.add_argument("--support-ratios", nargs="+", type=float, default=[0.30])
    parser.add_argument("--models", nargs="+", default=["sehgnn_official", "openhgnn_sehgnn", "openhgnn_hgt", "openhgnn_simplehgn"])
    parser.add_argument("--sehgnn-repo", type=Path, default=Path("external/SeHGNN"))
    parser.add_argument("--openhgnn-repo", type=Path, default=Path("external/OpenHGNN"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate21_open_sota"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--primary-eval-mode", default="compressed_projected")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-models", nargs="*", default=[])
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--strict", nargs="?", const=True, default=True, type=_bool_arg)
    parser.add_argument("--clone-external", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=16)
    args = parser.parse_args(argv)
    requested = {str(dataset).upper() for dataset in args.datasets}
    if args.dataset_seeds is None:
        args.dataset_seeds = [f"{dataset}:{seed}" for dataset, seeds in DEFAULT_DATASET_SEEDS.items() if dataset in requested for seed in seeds]
    else:
        pairs = _parse_dataset_seeds(args.dataset_seeds)
        args.dataset_seeds = [f"{dataset}:{seed}" for dataset, seed in pairs if dataset in requested]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
