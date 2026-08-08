"""
Train Bayesian Concept Consolidation on Split-MNIST.

Produces the core comparison figure: parameter drift vs. Knowledge
Stability Index (KSI) across tasks -- the "money plot" for the paper.

Usage:
    python scripts/train.py --epochs 3 --device cpu
    python scripts/train.py --epochs 1 --device cpu --no_concept
    python scripts/train.py --epochs 1 --device cpu --seed 42
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_cl.model import ConceptCLModel, ConceptPrototypeStore
from concept_cl.losses import total_loss
from concept_cl.data import get_split_mnist_tasks, SPLIT_MNIST_TASKS
from concept_cl.replay_buffer import UncertaintyReplayBuffer
from concept_cl.ewc import EWCHelper


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_distance(model, initial_state):
    dist = 0.0
    for name, param in model.named_parameters():
        dist += torch.norm(param.detach().cpu() - initial_state[name]).item() ** 2
    return dist ** 0.5


def evaluate(model, test_loaders, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for loader in test_loaders:
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x, sample=False)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            accs.append(correct / total if total > 0 else float("nan"))
    model.train()
    return accs


def run_experiment(epochs=1, batch_size=128, device="cpu",
                    use_adaptive=True, use_replay=True, use_concept=True,
                    method="concept", ewc_lambda=100.0,
                    seed=0, train_loaders=None, test_loaders=None,
                    save_plot_path=None, verbose=True):
    """
    Runs one full Split-MNIST continual-learning pass under the given
    settings and returns a dict of results.

    method: "concept" (this paper's approach: concept-stability loss,
            optionally adaptive/replay -- controlled by use_* flags) or
            "ewc" (classic Elastic Weight Consolidation baseline; use_concept
            is ignored in this mode, EWC penalty replaces it).

    Pass in pre-built train_loaders/test_loaders (from get_split_mnist_tasks)
    to avoid reloading MNIST from disk on every call when running many
    conditions.
    """
    set_seed(seed)
    device = torch.device(device)

    if train_loaders is None or test_loaders is None:
        train_loaders, test_loaders = get_split_mnist_tasks(batch_size=batch_size)

    model = ConceptCLModel(in_channels=1, feat_dim=128, concept_dim=32, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    proto_store = ConceptPrototypeStore()
    replay_buf = UncertaintyReplayBuffer(per_class_capacity=50)
    ewc_helper = EWCHelper(model, lambda_ewc=ewc_lambda) if method == "ewc" else None

    initial_state = {name: p.detach().clone().cpu() for name, p in model.named_parameters()}
    param_drift_history, ksi_history, avg_acc_history = [], [], []

    for task_idx, (classes, train_loader) in enumerate(zip(SPLIT_MNIST_TASKS, train_loaders)):
        if verbose:
            print(f"\n=== Task {task_idx}: classes {classes} ===")
        model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                replay_batch = replay_buf.sample(batch_size // 2, device) if use_replay else None

                optimizer.zero_grad()
                if method == "ewc":
                    logits, _ = model(x, sample=False)
                    task_ce = torch.nn.functional.cross_entropy(logits, y)
                    ewc_pen = ewc_helper.penalty()
                    loss = task_ce + ewc_pen
                    components = {"task": task_ce.item(),
                                  "ewc_penalty": ewc_pen.item() if torch.is_tensor(ewc_pen) else ewc_pen,
                                  "kl": 0.0, "concept": 0.0, "replay": 0.0}
                else:
                    loss, components, _ = total_loss(
                        model, x, y, proto_store,
                        replay_batch=replay_batch, use_adaptive=use_adaptive, use_concept=use_concept,
                    )
                loss.backward()
                # Safety net: EWC's penalty can still spike after a task
                # boundary; clip to prevent gradient-explosion-driven
                # accuracy collapse like the first EWC run showed.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            if verbose:
                print(f"  epoch {epoch}: {components}")

        model.eval()
        with torch.no_grad():
            class_concepts = {c: [] for c in classes}
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                _, concepts = model(x, sample=False)
                for c in classes:
                    mask = (y == c)
                    if mask.sum() > 0:
                        class_concepts[c].append(concepts[mask])
            for c, chunks in class_concepts.items():
                proto_store.register_or_update(c, torch.cat(chunks, dim=0), task_idx)

            for old_class, stored_x in replay_buf.buffer_x.items():
                if old_class in classes:
                    continue
                stored_x = stored_x.to(device)
                _, old_concepts = model(stored_x, sample=False)
                proto_store.register_or_update(old_class, old_concepts, task_idx)
        model.train()

        if method == "ewc":
            ewc_helper.register_task(train_loader, device)

        # ALWAYS store samples for later KSI re-evaluation, regardless of
        # whether this method uses replay in its training loss -- KSI
        # tracking needs old-class samples to re-embed at later task
        # boundaries even for methods (like plain EWC) that don't replay.
        # Using replay in the loss (use_replay) is a separate concern from
        # storing samples for measurement purposes (always on).
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            replay_buf.add_task_data(model, x, y)
            break

        accs = evaluate(model, test_loaders[:task_idx + 1], device)
        avg_acc = sum(accs) / len(accs)
        drift = parameter_distance(model, initial_state)
        ksi = proto_store.knowledge_stability_index()

        avg_acc_history.append(avg_acc)
        param_drift_history.append(drift)
        ksi_history.append(ksi)
        if verbose:
            print(f"  avg_acc_so_far={avg_acc:.4f}  param_drift={drift:.3f}  KSI={ksi}")

    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(7, 5))
        tasks = list(range(len(SPLIT_MNIST_TASKS)))
        ax1.plot(tasks, param_drift_history, "o-", color="tab:red", label="Parameter drift (L2)")
        ax1.set_xlabel("Task index")
        ax1.set_ylabel("Parameter drift", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(tasks, ksi_history, "s-", color="tab:blue", label="Knowledge Stability Index")
        ax2.set_ylabel("KSI (concept similarity)", color="tab:blue")
        ax2.set_ylim(0, 1.05)
        plt.title("Weight drift vs. concept drift across tasks")
        fig.tight_layout()
        plt.savefig(save_plot_path, dpi=150)
        plt.close(fig)

    return {
        "avg_acc_history": avg_acc_history,
        "param_drift_history": param_drift_history,
        "ksi_history": ksi_history,
        "final_acc": avg_acc_history[-1],
        "final_ksi": ksi_history[-1],
        "final_drift": param_drift_history[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_adaptive", action="store_true",
                         help="disable adaptive concept weighting (ablation)")
    parser.add_argument("--no_replay", action="store_true",
                         help="disable replay (ablation)")
    parser.add_argument("--no_concept", action="store_true",
                         help="disable concept-stability loss entirely (ablation / baseline)")
    args = parser.parse_args()

    results = run_experiment(
        epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        use_adaptive=not args.no_adaptive, use_replay=not args.no_replay,
        use_concept=not args.no_concept, seed=args.seed,
        save_plot_path="results/drift_vs_ksi.png", verbose=True,
    )
    print("\nSaved results/drift_vs_ksi.png")
    print("avg_acc_history:", results["avg_acc_history"])
    print("final_acc:", results["final_acc"], "final_ksi:", results["final_ksi"])


if __name__ == "__main__":
    main()
