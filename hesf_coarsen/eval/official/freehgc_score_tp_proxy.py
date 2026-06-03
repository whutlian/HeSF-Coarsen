from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from hesf_coarsen.eval.official.condensation_score_tp_proxy import build_gate21_22_condensation_proxy_rows


def build_freehgc_score_tp_proxy_rows(rows: Iterable[Mapping[str, Any]], *, datasets: Sequence[str] = ("DBLP", "ACM", "IMDB")) -> list[dict[str, Any]]:
    return build_gate21_22_condensation_proxy_rows(rows, datasets=datasets, baselines=("FreeHGC",))
