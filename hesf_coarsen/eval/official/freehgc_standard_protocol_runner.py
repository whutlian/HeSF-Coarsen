from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FREEHGC_STANDARD_PROTOCOL_FIELDS = (
    "dataset",
    "ratio",
    "upstream_run_attempted",
    "upstream_run_success",
    "upstream_command",
    "upstream_commit_hash",
    "upstream_metric_micro",
    "upstream_metric_macro",
    "uses_standard_condensation_protocol",
    "uses_target_preserving_protocol",
    "official_hgb_exported",
    "official_sehgnn_unmodified",
    "eligible_for_official_main_table",
    "failure_type",
    "failure_reason",
)


def build_freehgc_standard_protocol_rows(
    repo_rows: Iterable[Mapping[str, Any]],
    *,
    datasets: Sequence[str] = ("DBLP", "ACM", "IMDB"),
    ratios: Sequence[float] = (0.50,),
) -> list[dict[str, Any]]:
    repo = next((dict(row) for row in repo_rows if str(row.get("repo_name", row.get("method", ""))) == "FreeHGC"), {})
    repo_path = Path(str(repo.get("local_path", "")))
    commit = str(repo.get("commit_hash", ""))
    train_hgb = repo_path / "HGB" / "train_hgb.py"
    model_hgb = repo_path / "HGB" / "model_hgb.py"
    can_attempt = bool(train_hgb.exists() and model_hgb.exists())
    reason = "" if can_attempt else "FreeHGC upstream HGB entrypoint is not runnable in this clone because HGB/model_hgb.py is missing; standard condensation is separated from official TP protocol."
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for ratio in ratios:
            rows.append(
                {
                    "dataset": dataset,
                    "ratio": float(ratio),
                    "upstream_run_attempted": False,
                    "upstream_run_success": False,
                    "upstream_command": f"python {train_hgb} --dataset {dataset} --reduction-rate {ratio}" if can_attempt else "",
                    "upstream_commit_hash": commit,
                    "upstream_metric_micro": "",
                    "upstream_metric_macro": "",
                    "uses_standard_condensation_protocol": True,
                    "uses_target_preserving_protocol": False,
                    "official_hgb_exported": False,
                    "official_sehgnn_unmodified": False,
                    "eligible_for_official_main_table": False,
                    "failure_type": "" if can_attempt else "standard_condensation_protocol_mismatch",
                    "failure_reason": reason,
                }
            )
    return rows
