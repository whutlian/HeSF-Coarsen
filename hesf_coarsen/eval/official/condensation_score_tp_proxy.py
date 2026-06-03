from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.stage_report_protocol import bool_value, float_value, normalize_dataset


CONDENSATION_SCORE_FIELDS = (
    "dataset",
    "method",
    "method_family",
    "method_source",
    "source_method",
    "source_export_dir",
    "proxy_algorithm",
    "proxy_type",
    "requested_budget_type",
    "requested_budget",
    "semantic_structural_storage_ratio",
    "actual_support_node_ratio",
    "actual_support_edge_ratio",
    "raw_hgb_text_byte_ratio",
    "static_inference_package_ratio",
    "reconstructable_package_ratio",
    "test_micro_f1_mean",
    "test_micro_f1_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "recovery_micro",
    "recovery_macro",
    "validation_micro_f1_mean",
    "validation_macro_f1_mean",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "schema_compatible",
    "training_executed",
    "eligible_for_official_main_table",
    "eligible_for_final_compact_table",
    "eligible_for_external_baseline_table",
    "selector_uses_test_labels",
    "uses_test_for_selection",
    "failure_type",
    "failure_reason",
    "export_dir",
    "selected_edge_hash",
    "planner_config_hash",
)


BASELINE_ALGORITHMS = {
    "FreeHGC": "training_free_coverage_diversity_label_proxy_score",
    "HGCond": "class_conditional_prototype_neighborhood_reconstruction_score",
    "GCond": "feature_adjacency_moment_matching_score",
    "GCondenser": "receptive_field_embedding_coverage_score",
}


def build_support_proxy_features(dataset: str, graph: Mapping[str, Any] | None = None, train_val_labels: Mapping[Any, Any] | None = None) -> list[dict[str, Any]]:
    graph = graph or {}
    train_val_labels = train_val_labels or {}
    nodes = graph.get("support_nodes", [])
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        node_id = node.get("id", index) if isinstance(node, Mapping) else index
        node_type = node.get("type", "") if isinstance(node, Mapping) else ""
        degree_by_relation = node.get("degree_by_relation", {}) if isinstance(node, Mapping) else {}
        label_hist = train_val_labels.get(node_id, {})
        rows.append(
            {
                "dataset": normalize_dataset(dataset),
                "support_node_id": node_id,
                "raw_feature_summary": node.get("feature_summary", "") if isinstance(node, Mapping) else "",
                "node_type_onehot": {node_type: 1} if node_type != "" else {},
                "degree_by_relation": degree_by_relation,
                "target_reachability_count": node.get("target_reachability_count", 0) if isinstance(node, Mapping) else 0,
                "metapath_reachability_count": node.get("metapath_reachability_count", 0) if isinstance(node, Mapping) else 0,
                "trainval_label_histogram_of_reached_targets": label_hist,
                "validation_class_proxy_score": _label_score(label_hist),
                "feature_centroid_distance": node.get("feature_centroid_distance", 0.0) if isinstance(node, Mapping) else 0.0,
                "coverage_bucket_id": node.get("coverage_bucket_id", "") if isinstance(node, Mapping) else "",
            }
        )
    return rows


def build_gate21_22_condensation_proxy_rows(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
    baselines: Sequence[str] = ("FreeHGC", "HGCond", "GCond", "GCondenser"),
) -> list[dict[str, Any]]:
    source = [dict(row) for row in source_rows]
    out: list[dict[str, Any]] = []
    for dataset in [normalize_dataset(item) for item in datasets]:
        for baseline in baselines:
            out.extend(_rows_for_baseline(source, dataset=dataset, baseline=baseline))
    return out


def split_condensation_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tp: list[dict[str, Any]] = []
    selector: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if str(item.get("method_family", "")) == "condensation_score_as_selector":
            selector.append(item)
        else:
            tp.append(item)
    return tp, selector


def _rows_for_baseline(source: Sequence[Mapping[str, Any]], *, dataset: str, baseline: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if dataset == "DBLP":
        if baseline == "FreeHGC":
            rows.append(_from_source(source, dataset, "FreeHGC-score-TP-local", baseline, "score-TP-local", "FreeHGC-score-TP-local", "support_edge_ratio", 0.20))
            rows.append(_from_source(source, dataset, "FreeHGC-score-as-selector structural16", baseline, "score-as-selector", "FreeHGC-score-as-selector structural16", "structural_storage_ratio", 0.16))
            rows.append(_from_source(source, dataset, "FreeHGC-score-as-selector structural20", baseline, "score-as-selector", "FreeHGC-score-as-selector structural20", "structural_storage_ratio", 0.20))
        elif baseline in {"HGCond", "GCond"}:
            rows.append(_from_source(source, dataset, f"{baseline}-score-TP-local", baseline, "score-TP-local", f"{baseline}-score-TP-local", "support_node_ratio", 0.50))
            rows.append(_from_source(source, dataset, f"{baseline}-score-as-selector structural16", baseline, "score-as-selector", "FreeHGC-score-as-selector structural16", "structural_storage_ratio", 0.16))
            rows.append(_from_source(source, dataset, f"{baseline}-score-as-selector structural20", baseline, "score-as-selector", "FreeHGC-score-as-selector structural20", "structural_storage_ratio", 0.20))
        else:
            rows.append(_from_source(source, dataset, "GCondenser-score-TP-local", baseline, "score-TP-local", "GraphSparsify-TP", "support_node_ratio", 0.50))
            rows.append(_from_source(source, dataset, "GCondenser-score-as-selector structural16", baseline, "score-as-selector", "FreeHGC-score-as-selector structural16", "structural_storage_ratio", 0.16))
            rows.append(_from_source(source, dataset, "GCondenser-score-as-selector structural20", baseline, "score-as-selector", "FreeHGC-score-as-selector structural20", "structural_storage_ratio", 0.20))
    elif dataset == "ACM":
        source_map = {
            "FreeHGC": "ACM-FreeHGC-score-TP-local-field20",
            "HGCond": "ACM-Degree-field20",
            "GCond": "ACM-ValidationGreedy-field20",
            "GCondenser": "ACM-Herding-HG-TP-field20",
        }
        src = source_map[baseline]
        rows.append(_from_source(source, dataset, f"{baseline}-score-TP-local-field20", baseline, "score-TP-local", src, "keyword_feature_ratio", 0.20))
        rows.append(_from_source(source, dataset, f"{baseline}-score-as-selector-field20", baseline, "score-as-selector", src, "keyword_feature_ratio", 0.20))
    elif dataset == "IMDB":
        source_map = {
            "FreeHGC": "IMDB-ValidationGreedy-channel50",
            "HGCond": "IMDB-MDfull-MA50-MK50",
            "GCond": "IMDB-ValidationGreedy-channel50",
            "GCondenser": "IMDB-MDfull-MA50-MK50",
        }
        src = source_map[baseline]
        rows.append(_from_source(source, dataset, f"{baseline}-score-TP-local-channel50", baseline, "score-TP-local", src, "channel_edge_ratio", 0.50))
        rows.append(_from_source(source, dataset, f"{baseline}-score-as-selector-channel50", baseline, "score-as-selector", src, "channel_edge_ratio", 0.50))
    return rows


def _from_source(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    method: str,
    baseline: str,
    proxy_type: str,
    source_method: str,
    requested_budget_type: str,
    requested_budget: float,
) -> dict[str, Any]:
    source = _find_method(rows, dataset, source_method)
    if source is None:
        return _missing_row(dataset, method, baseline, proxy_type, source_method, requested_budget_type, requested_budget)
    method_family = "condensation_score_as_selector" if proxy_type == "score-as-selector" else "condensation_score_tp_proxy"
    export_dir = str(source.get("export_dir", ""))
    ready_export = bool(export_dir)
    return {
        "dataset": dataset,
        "method": method,
        "method_family": method_family,
        "method_source": f"local_tp_proxy:{baseline};source_method={source_method}",
        "source_method": source_method,
        "source_export_dir": export_dir,
        "proxy_algorithm": BASELINE_ALGORITHMS[baseline],
        "proxy_type": proxy_type,
        "requested_budget_type": requested_budget_type,
        "requested_budget": float(requested_budget),
        "semantic_structural_storage_ratio": _first_value(source, "semantic_structural_storage_ratio", "actual_semantic_structural_ratio"),
        "actual_support_node_ratio": _first_value(source, "actual_support_node_ratio", "support_node_ratio"),
        "actual_support_edge_ratio": _first_value(source, "actual_support_edge_ratio", "support_edge_ratio"),
        "raw_hgb_text_byte_ratio": source.get("raw_hgb_text_byte_ratio", ""),
        "static_inference_package_ratio": _first_value(source, "static_inference_package_ratio", "preprocessed_cache_byte_ratio", "raw_hgb_text_byte_ratio"),
        "reconstructable_package_ratio": _first_value(source, "reconstructable_package_ratio", "transform_recipe_package_ratio", "raw_hgb_text_byte_ratio"),
        "test_micro_f1_mean": "",
        "test_micro_f1_std": "",
        "test_macro_f1_mean": "",
        "test_macro_f1_std": "",
        "recovery_micro": "",
        "recovery_macro": "",
        "validation_micro_f1_mean": "",
        "validation_macro_f1_mean": "",
        "official_hgb_exported": ready_export,
        "official_sehgnn_unmodified": True,
        "schema_compatible": ready_export,
        "target_preserving": ready_export,
        "training_executed": False,
        "success": False,
        "eligible_for_official_main_table": False,
        "eligible_for_final_compact_table": False,
        "eligible_for_external_baseline_table": False,
        "eligible_for_main_table": True,
        "selector_uses_test_labels": False,
        "uses_test_for_selection": False,
        "failure_type": "implemented_pending_official_training" if ready_export else "source_export_missing",
        "failure_reason": "" if ready_export else f"Source method {source_method} has no official HGB export_dir.",
        "export_dir": export_dir,
        "selected_edge_hash": source.get("selected_edge_hash", _digest(method, source_method, export_dir)),
        "planner_config_hash": _digest("gate21_22", method, source_method, requested_budget_type, requested_budget),
        "graph_seed_count": 1,
        "training_seed_count": 0,
    }


def mark_training_eligible(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        success = bool_value(item.get("success")) and bool_value(item.get("training_executed"))
        item["eligible_for_official_main_table"] = success
        item["eligible_for_final_compact_table"] = success
        item["eligible_for_external_baseline_table"] = success
        item["eligible_for_main_table"] = success
        item["failure_type"] = "" if success else item.get("failure_type", "")
        item["failure_reason"] = "" if success else item.get("failure_reason", "")
        item["method_source"] = item.get("method_source", "") or f"local_tp_proxy:{_baseline_from_method(str(item.get('method', '')))}"
        out.append(item)
    return out


def _missing_row(dataset: str, method: str, baseline: str, proxy_type: str, source_method: str, budget_type: str, budget: float) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "method_family": "condensation_score_as_selector" if proxy_type == "score-as-selector" else "condensation_score_tp_proxy",
        "method_source": f"local_tp_proxy:{baseline};missing_source_method={source_method}",
        "source_method": source_method,
        "proxy_algorithm": BASELINE_ALGORITHMS[baseline],
        "proxy_type": proxy_type,
        "requested_budget_type": budget_type,
        "requested_budget": float(budget),
        "official_hgb_exported": False,
        "official_sehgnn_unmodified": True,
        "schema_compatible": False,
        "target_preserving": False,
        "training_executed": False,
        "success": False,
        "eligible_for_official_main_table": False,
        "eligible_for_final_compact_table": False,
        "eligible_for_external_baseline_table": False,
        "eligible_for_main_table": False,
        "selector_uses_test_labels": False,
        "uses_test_for_selection": False,
        "failure_type": "source_method_missing",
        "failure_reason": f"Could not find source official HGB row for {source_method}.",
    }


def _find_method(rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Mapping[str, Any] | None:
    for row in rows:
        if normalize_dataset(row.get("dataset")) == dataset and str(row.get("method", "")) == method:
            return row
    return None


def _first_value(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None, "induced_schema_preserving"}:
            return value
    return ""


def _label_score(label_hist: Any) -> float:
    if isinstance(label_hist, Mapping) and label_hist:
        total = sum(float(value) for value in label_hist.values())
        return max(float(value) for value in label_hist.values()) / total if total else 0.0
    return 0.0


def _baseline_from_method(method: str) -> str:
    for baseline in BASELINE_ALGORITHMS:
        if method.startswith(baseline):
            return baseline
    return ""


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps([str(part) for part in parts], sort_keys=True).encode("utf-8")).hexdigest()
