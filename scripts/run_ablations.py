
import argparse
import csv
import os
import statistics
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_cl.data import get_split_mnist_tasks, SPLIT_MNIST_TASKS
from scripts.train import run_experiment

CONDITIONS = {
    "full_method":        dict(method="concept", use_adaptive=True,  use_replay=True, use_concept=True),
    "no_concept_loss":    dict(method="concept", use_adaptive=True,  use_replay=True, use_concept=False),
    "fixed_weight":       dict(method="concept", use_adaptive=False, use_replay=True, use_concept=True),
    "ewc_baseline":       dict(method="ewc",     use_adaptive=True,  use_replay=False, use_concept=False, ewc_lambda=1000.0),
    "ewc_plus_replay":    dict(method="ewc",     use_adaptive=True,  use_replay=True,  use_concept=False, ewc_lambda=1000.0),
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

    raw_results = {name: {"acc": [], "ksi": [], "ksi_trajectories": [], "acc_trajectories": []} for name in CONDITIONS}

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
            raw_results[cond_name]["ksi_trajectories"].append(result["ksi_history"])
            raw_results[cond_name]["acc_trajectories"].append(result["avg_acc_history"])
            print(f"    final_acc={result['final_acc']:.4f}  final_ksi={result['final_ksi']:.4f}")

    # --- Aggregate and print summary table ---
    os.makedirs("results", exist_ok=True)
    summary_rows = []
    print("\n" + "=" * 70)
    print(f"{'Condition':<20}{'Acc (mean±std)':<22}{'KSI (mean±std)':<22}")
    print("=" * 70)
    for cond_name, vals in raw_results.items():
        # Filter out any NaN (e.g. KSI is undefined until a class has >= 2
        # history points) so one bad value doesn't crash the whole summary.
        clean_acc = [v for v in vals["acc"] if v == v]  # v == v is False only for NaN
        clean_ksi = [v for v in vals["ksi"] if v == v]

        acc_mean = statistics.mean(clean_acc) if clean_acc else float("nan")
        acc_std = statistics.stdev(clean_acc) if len(clean_acc) > 1 else 0.0
        ksi_mean = statistics.mean(clean_ksi) if clean_ksi else float("nan")
        ksi_std = statistics.stdev(clean_ksi) if len(clean_ksi) > 1 else 0.0

        n_note = "" if len(clean_ksi) == len(vals["ksi"]) else f"  (KSI valid for {len(clean_ksi)}/{len(vals['ksi'])} seeds)"
        print(f"{cond_name:<20}{acc_mean:.4f} ± {acc_std:.4f}    {ksi_mean:.4f} ± {ksi_std:.4f}{n_note}")
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

    # --- Trajectory plots: mean +/- std band per condition, per task ---
    colors = {
        "full_method": "tab:blue", "no_concept_loss": "tab:red",
        "fixed_weight": "tab:orange", "ewc_baseline": "tab:gray",
        "ewc_plus_replay": "tab:purple",
    }
    n_tasks = len(SPLIT_MNIST_TASKS)
    task_x = list(range(1, n_tasks + 1))  # 1-indexed to match "Task 1..5" in the paper text

    def plot_trajectory(metric_key, ylabel, title, ylim, save_path, legend_loc="lower left"):
        """Shared plotting logic for any per-task trajectory metric (KSI, accuracy, ...)."""
        plt.figure(figsize=(8, 6))
        for cond_name, vals in raw_results.items():
            trajectories = vals[metric_key]  # list of per-seed lists, each length n_tasks
            # Some early entries can be NaN (e.g. KSI has no prior point at
            # task 0) -- handle per-task-index NaN filtering rather than
            # dropping whole runs.
            means, stds = [], []
            for t in range(n_tasks):
                vals_at_t = [traj[t] for traj in trajectories if t < len(traj) and traj[t] == traj[t]]
                if vals_at_t:
                    means.append(statistics.mean(vals_at_t))
                    stds.append(statistics.stdev(vals_at_t) if len(vals_at_t) > 1 else 0.0)
                else:
                    means.append(float("nan"))
                    stds.append(0.0)

            color = colors.get(cond_name, None)
            plt.plot(task_x, means, "o-", label=cond_name, color=color)
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            plt.fill_between(task_x, lower, upper, alpha=0.15, color=color)

        plt.xlabel("Task index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

    plot_trajectory(
        metric_key="ksi_trajectories", ylabel="Knowledge Stability Index (KSI)",
        title=f"KSI trajectory across tasks (mean ± std over {len(args.seeds)} seeds)",
        ylim=(-0.3, 1.05), save_path="results/ksi_trajectories.png", legend_loc="lower left",
    )
    plot_trajectory(
        metric_key="acc_trajectories", ylabel="Cumulative test accuracy",
        title=f"Accuracy trajectory across tasks (mean ± std over {len(args.seeds)} seeds)",
        ylim=(0.0, 1.05), save_path="results/accuracy_trajectories.png", legend_loc="lower left",
    )


if __name__ == "__main__":
    main()
