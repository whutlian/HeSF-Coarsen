from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Gate17 diagnostic figures.")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/gate17_tables"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("outputs/gate17_diagnostics"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate17_figures"))
    args = parser.parse_args(argv)
    import pandas as pd
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.tables_dir / "gate17_raw_rows.csv")
    gaps = pd.read_csv(args.tables_dir / "gate17_exact_only_paired_gaps.csv")
    proto = pd.read_csv(args.diagnostics_dir / "prototype_diagnostics.csv")
    occ = pd.read_csv(args.diagnostics_dir / "occlusion_block_scores.csv")

    for metric, name in [
        ("macro_f1", "macro_f1_vs_support_ratio.png"),
        ("accuracy", "accuracy_vs_support_ratio.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for method, group in raw[raw["status"].eq("success")].groupby("method"):
            agg = group.groupby("requested_support_ratio")[metric].mean().reset_index()
            plt.plot(agg["requested_support_ratio"], agg[metric], marker="o", label=str(method)[:28])
        plt.xlabel("requested support ratio")
        plt.ylabel(metric)
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(args.output_dir / name, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    if len(gaps):
        agg = gaps.groupby(["dataset", "method"])["delta_macro_f1"].mean().reset_index()
        labels = agg["dataset"] + "\n" + agg["method"].str.replace("HeSF-SS-", "", regex=False).str[:12]
        plt.bar(labels, agg["delta_macro_f1"])
    plt.xticks(rotation=80, fontsize=6)
    plt.ylabel("exact delta macro-F1")
    plt.tight_layout()
    plt.savefig(args.output_dir / "exact_budget_delta_macro_by_dataset.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    if len(gaps):
        gaps[gaps["dataset"].eq("DBLP")].groupby("method")["delta_macro_f1"].mean().sort_values().plot(kind="barh")
    plt.xlabel("DBLP delta macro-F1")
    plt.tight_layout()
    plt.savefig(args.output_dir / "dblp_gap_by_method.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    if len(proto):
        proto["prototype_member_count_max"].astype(float).hist(bins=30)
    plt.xlabel("prototype member count max")
    plt.ylabel("runs")
    plt.tight_layout()
    plt.savefig(args.output_dir / "prototype_member_count_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    if len(occ):
        occ["final_block_importance"].astype(float).hist(bins=30)
    plt.xlabel("occlusion final block importance")
    plt.ylabel("blocks")
    plt.tight_layout()
    plt.savefig(args.output_dir / "occlusion_importance_distribution.png", dpi=160)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
