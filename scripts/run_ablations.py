"""
Run the full ablation comparison across multiple seeds and aggregate
mean +/- std for final accuracy and final KSI per condition.

Usage:
    python scripts/run_ablations.py --seeds 0 1 2 --epochs 1 --device cpu
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_cl.data import get_split_mnist_tasks
from scripts.train import run_experiment

CONDITIONS = {
    "full_method":        dict(method="concept", use_adaptive=True,  use_replay=True, use_concept=True),
    "no_concept_loss":    dict(method="concept", use_adaptive=True,  use_replay=True, use_concept=False),
    "fixed_weight":       dict(method="concept", use_adaptive=False, use_replay=True, use_concept=True),
    "ewc_baseline":       dict(method="ewc",     use_adaptive=True,  use_replay=False, use_concept=False, ewc_lambda=100.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    print(f"Loading Split-MNIST once, reused across {len(CONDITIONS)} conditions x {len(args.seeds)} seeds...")
    train_loaders, test_loaders = get_split_mnist_tasks(batch_size=args.batch_size)

    raw_results = {name: {"acc": [], "ksi": []} for name in CONDITIONS}

    for cond_name, cond_kwargs in CONDITIONS.items():
        for seed in args.seeds:
            print(f"\n>>> Condition: {cond_name} | seed {seed}")
            result = run_experiment(
                epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                seed=seed, train_loaders=train_loaders, test_loaders=test_loaders,
                save_plot_path=None, verbose=False, **cond_kwargs,
            )
            raw_results[cond_name]["acc"].append(result["final_acc"])
            raw_results[cond_name]["ksi"].append(result["final_ksi"])
            print(f"    final_acc={result['final_acc']:.4f}  final_ksi={result['final_ksi']:.4f}")

    # --- Aggregate and print summary table ---
    os.makedirs("results", exist_ok=True)
    summary_rows = []
    print("\n" + "=" * 70)
    print(f"{'Condition':<20}{'Acc (mean±std)':<22}{'KSI (mean±std)':<22}")
    print("=" * 70)
    for cond_name, vals in raw_results.items():
        acc_mean, acc_std = statistics.mean(vals["acc"]), (statistics.stdev(vals["acc"]) if len(vals["acc"]) > 1 else 0.0)
        ksi_mean, ksi_std = statistics.mean(vals["ksi"]), (statistics.stdev(vals["ksi"]) if len(vals["ksi"]) > 1 else 0.0)
        print(f"{cond_name:<20}{acc_mean:.4f} ± {acc_std:.4f}    {ksi_mean:.4f} ± {ksi_std:.4f}")
        summary_rows.append({
            "condition": cond_name,
            "acc_mean": acc_mean, "acc_std": acc_std,
            "ksi_mean": ksi_mean, "ksi_std": ksi_std,
            "n_seeds": len(vals["acc"]),
            "raw_acc": vals["acc"], "raw_ksi": vals["ksi"],
        })
    print("=" * 70)

    csv_path = "results/ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "acc_mean", "acc_std", "ksi_mean", "ksi_std", "n_seeds", "raw_acc", "raw_ksi"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
