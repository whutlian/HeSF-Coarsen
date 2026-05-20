from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hesf_coarsen.io.schema import HeteroGraph

COARSE_TRANSFER = "coarse_transfer"
APPROX_FULL_TARGET_ADAPTER = "approx_full_target_adapter"
REAL_FULL_TARGET_INFERENCE = "real_full_target_inference"


class FullTargetBackbone(Protocol):
    fidelity: str

    def fit(self, *args: Any, **kwargs: Any) -> Any: ...

    def predict(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class TaskEvalSummary:
    protocol: str
    backbone_fidelity: str
    metrics: dict


def evaluate_real_full_target_protocol(
    original_graph: HeteroGraph,
    compressed_support_graph: HeteroGraph,
    backbone: FullTargetBackbone,
    require_fidelity: str = "faithful",
) -> TaskEvalSummary:
    del original_graph, compressed_support_graph
    allowed = {"official", "faithful"}
    fidelity = str(getattr(backbone, "fidelity", ""))
    if fidelity not in allowed:
        raise ValueError(
            "real_full_target_inference requires an official or faithful backbone; "
            f"got fidelity={fidelity!r}"
        )
    if require_fidelity == "official" and fidelity != "official":
        raise ValueError("real_full_target_inference requires fidelity='official'")
    return TaskEvalSummary(
        protocol=REAL_FULL_TARGET_INFERENCE,
        backbone_fidelity=fidelity,
        metrics={"status": "not_run_interface_only"},
    )
