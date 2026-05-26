from __future__ import annotations

from typing import Any, Sequence


LABEL_GRAPH_ABLATION_FIELDS = [
    "dataset",
    "seed",
    "method",
    "ablation_name",
    "label_feats_enabled",
    "num_label_hops",
    "num_feature_hops",
    "graph_edges_enabled",
    "feature_only_mode",
    "success",
    "test_micro_f1",
    "test_macro_f1",
    "validation_micro_f1",
    "validation_macro_f1",
    "recovery_vs_default_method_micro",
    "notes",
]


ABLATIONS = [
    ("default", True, 4, 2, True, False),
    ("no_label_feats", False, 0, 2, True, False),
    ("num_label_hops_0", True, 0, 2, True, False),
    ("num_hops_0_or_feature_only", False, 0, 0, False, True),
]


def planned_label_graph_ablation_rows(
    *,
    dataset: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    notes: str = "planned; not run in unmodified official main table",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in seeds]:
        for method in [str(value) for value in methods]:
            for name, label_feats, label_hops, feature_hops, graph_edges, feature_only in ABLATIONS:
                rows.append(
                    {
                        "dataset": str(dataset).upper(),
                        "seed": seed,
                        "method": method,
                        "ablation_name": name,
                        "label_feats_enabled": bool(label_feats),
                        "num_label_hops": int(label_hops),
                        "num_feature_hops": int(feature_hops),
                        "graph_edges_enabled": bool(graph_edges),
                        "feature_only_mode": bool(feature_only),
                        "success": "planned",
                        "test_micro_f1": "",
                        "test_macro_f1": "",
                        "validation_micro_f1": "",
                        "validation_macro_f1": "",
                        "recovery_vs_default_method_micro": "",
                        "notes": notes,
                    }
                )
    return rows
