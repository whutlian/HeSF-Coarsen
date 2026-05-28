from __future__ import annotations

from typing import Any

from hesf_coarsen.eval.official.gate21_9_decision import EMPTY_SHA256


def unsupported_tensor_dump_row(*, dataset: str, method: str, metapath_key: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "metapath_key": metapath_key,
        "feature_tensor_hash": EMPTY_SHA256,
        "feature_tensor_bytes": 0,
        "real_tensor_dumped": False,
        "introspection_supported": False,
        "failure_type": "official_sehgnn_intermediate_tensors_not_exposed",
        "failure_message": "No patched SeHGNN tensor dump was available for this run.",
    }
