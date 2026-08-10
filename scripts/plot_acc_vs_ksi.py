"""
Accuracy vs. KSI scatter plot -- the central conceptual result figure
(paper Section V-C / Fig. "acc_vs_ksi"). One point per condition, with
error bars from the mean +/- std already computed by run_ablations.py.

Reads results/ablation_summary.csv (produced by run_ablations.py) so it
does not require re-running any training.

Usage:
    python scripts/plot_acc_vs_ksi.py
"""

import ast
import csv
import os

import matplotlib.pyplot as plt

CSV_PATH = "results/ablation_summary.csv"
OUT_PATH = "results/accuracy_vs_ksi.png"

COLORS = {
    "full_method": "tab:blue", "no_concept_loss": "tab:red",
    "fixed_weight": "tab:orange", "ewc_baseline": "tab:gray",
    "ewc_plus_replay": "tab:purple",
}


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found -- run scripts/run_ablations.py first "
            f"to generate the summary this script reads from."
        )

    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    plt.figure(figsize=(7, 6))
    # Small per-condition label offsets to avoid overlapping text where
    # points sit close together (no_concept_loss and ewc_plus_replay in
    # particular land almost on top of each other -- that closeness is
    # itself part of the finding, but the labels still need to stay legible).
    label_offsets = {
        "full_method": (8, 8),
        "fixed_weight": (8, 8),
        "no_concept_loss": (10, -14),
        "ewc_plus_replay": (10, 10),
        "ewc_baseline": (8, 8),
    }

    for row in rows:
        cond = row["condition"]
        acc_mean, acc_std = float(row["acc_mean"]), float(row["acc_std"])
        ksi_mean, ksi_std = float(row["ksi_mean"]), float(row["ksi_std"])
        color = COLORS.get(cond, None)

        plt.errorbar(
            acc_mean, ksi_mean, xerr=acc_std, yerr=ksi_std,
            fmt="o", markersize=10, capsize=4, color=color, label=cond,
        )
        offset = label_offsets.get(cond, (8, 8))
        plt.annotate(cond, (acc_mean, ksi_mean),
                     textcoords="offset points", xytext=offset, fontsize=9, color=color)

    plt.xlabel("Final accuracy")
    plt.ylabel("Final Knowledge Stability Index (KSI)")
    plt.title("Accuracy vs. KSI across conditions\n(mean ± std over 3 seeds)")
    plt.grid(alpha=0.2)
    plt.xlim(0.1, 1.05)  # headroom so right-side labels aren't clipped at the plot edge
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
