from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts._common import markdown_table, write_csv
from hesf_coarsen.accuracy.full_target_protocol import make_protocol_row, required_provenance_fields
from hesf_coarsen.accuracy.model_fidelity_registry import all_fidelity_records


AUDIT_ROWS = [
    {
        "file": "hesf_coarsen/accuracy/target_support_hybrid.py",
        "actual_behavior": "Thin wrapper: rewrites assignments to preserve target nodes, aggregates support nodes through existing coarsen_graph, returns diagnostics.",
        "readiness": "small production utility",
        "next18_scope": "keep for A1/A2",
    },
    {
        "file": "hesf_coarsen/accuracy/full_target_inference.py",
        "actual_behavior": "Dispatches local SeHGNN/HETTREE-style evaluators and now routes target-preserved predictions through explicit protocol metadata.",
        "readiness": "local adapter, not official reproduction",
        "next18_scope": "keep only with fidelity tags",
    },
    {
        "file": "hesf_coarsen/eval/sehgnn_task.py",
        "actual_behavior": "Local SeHGNN-inspired metapath projection + semantic fusion evaluator.",
        "readiness": "lite adapter",
        "next18_scope": "appendix diagnostic unless official repo integration is completed",
    },
    {
        "file": "hesf_coarsen/eval/hettree_task.py",
        "actual_behavior": "Local HETTREE-inspired semantic-tree feature builder and attention classifier.",
        "readiness": "lite adapter; helpers are useful",
        "next18_scope": "appendix diagnostic unless official repo integration is completed",
    },
    {
        "file": "hesf_coarsen/accuracy/target_selection.py",
        "actual_behavior": "Heuristic target-anchor selection using degree confidence, coverage, diversity, and pseudo-label balance.",
        "readiness": "deprecated heuristic",
        "next18_scope": "deprecated A3 loader/diagnostic only",
    },
    {
        "file": "hesf_coarsen/accuracy/meta_recon.py",
        "actual_behavior": "Approximate semantic-tree tensor reconstruction error.",
        "readiness": "proxy diagnostic",
        "next18_scope": "deprecated A4 diagnostic only",
    },
    {
        "file": "hesf_coarsen/accuracy/distillation.py",
        "actual_behavior": "Softmax/KL helpers plus deterministic random teacher logits.",
        "readiness": "placeholder/proxy",
        "next18_scope": "deprecated A5 diagnostic only",
    },
    {
        "file": "hesf_coarsen/accuracy/task_aligned_score.py",
        "actual_behavior": "Weighted sum of proxy deltas.",
        "readiness": "bookkeeping heuristic",
        "next18_scope": "deprecated diagnostic only",
    },
]


def _write_audit(output: Path) -> None:
    lines = [
        "# Next18 Accuracy Protocol Implementation Audit",
        "",
        markdown_table(AUDIT_ROWS, ["file", "actual_behavior", "readiness", "next18_scope"]),
        "",
        "## Scope Decision",
        "",
        "- `A1_target_preserve`: keep in scope.",
        "- `A2_hybridA_keepall`: keep in scope; currently equivalent to A1 in implementation.",
        "- `A3_hybridB_selecttarget`, `A4_hybridB_meta_recon`, `A5_hybridB_distill`: deprecated unless needed to load old outputs.",
    ]
    (output / "implementation_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _protocol_check_rows() -> list[dict[str, Any]]:
    metrics = {
        "projected_original_micro_f1": 0.20,
        "projected_original_macro_f1": 0.10,
        "projected_original_accuracy": 0.25,
        "hybrid_target_original_micro_f1": 0.40,
        "hybrid_target_original_macro_f1": 0.30,
        "hybrid_target_original_accuracy": 0.50,
    }
    rows = []
    for eval_mode in ("coarse_transfer", "approx_full_target_adapter", "real_full_target_inference"):
        rows.append(
            make_protocol_row(
                metrics,
                eval_mode=eval_mode,
                model_name="sehgnn_lite",
                model_fidelity="lite_adapter",
                official_repo="no",
                official_preprocess="no",
                adapter_mode="protocol_check",
                path_set="lite",
                split_policy="synthetic_stratified",
                max_hops=2,
            )
        )
    return rows


def _write_protocol_summary(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    provenance = ", ".join(required_provenance_fields())
    lines = [
        "# Next18 Full-Target Protocol Check",
        "",
        markdown_table(rows, ["eval_mode", "metric_source", "target_domain", "support_domain", "inference_domain", "macro_f1"]),
        "",
        f"Required provenance fields: {provenance}.",
        "",
        "`real_full_target_inference` consumes explicit `hybrid_target_original_*` metrics and is separate from projected/approximate adapters.",
    ]
    (output / "full_target_protocol_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_failure_report(output: Path) -> None:
    root = output.parent / "exp_next18_model_fidelity_failure_report"
    root.mkdir(parents=True, exist_ok=True)
    rows = all_fidelity_records()
    write_csv(root / "model_fidelity_registry.csv", rows)
    lines = [
        "# Next18 Model Fidelity Failure Report",
        "",
        "Official/high-fidelity integration was not completed in this local run.",
        "",
        markdown_table(rows, ["model_name", "model_fidelity", "repository", "official_repo", "official_preprocess", "adapter_mode"]),
        "",
        "SeHGNN official repository is reachable, but its preprocessing/model code is not vendored or adapted to the local HeSF graph format.",
        "The public HETTREE repository advertised in the paper could not be accessed from the tested GitHub URL.",
        "FreeHGC is documented as a ratio/protocol reference only and is not integrated locally.",
    ]
    (root / "model_fidelity_failure.md").write_text("\n".join(lines), encoding="utf-8")


def run_next18_accuracy_protocol_audit(output: str | Path) -> dict[str, int]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_audit(output)
    rows = _protocol_check_rows()
    write_csv(output / "full_target_protocol_check.csv", rows)
    _write_protocol_summary(output, rows)
    _write_failure_report(output)
    return {"audit_rows": len(AUDIT_ROWS), "protocol_rows": len(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/exp_next18_accuracy_protocol_audit"))
    args = parser.parse_args(argv)
    result = run_next18_accuracy_protocol_audit(args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
