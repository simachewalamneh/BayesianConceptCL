"""
Train Bayesian Concept Consolidation on Split-MNIST.

Produces the core comparison figure: parameter drift vs. Knowledge
Stability Index (KSI) across tasks -- the "money plot" for the paper.

Usage:
    python scripts/train.py --epochs 3 --device cpu
"""

import argparse
import copy
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_cl.model import ConceptCLModel, ConceptPrototypeStore
from concept_cl.losses import total_loss
from concept_cl.data import get_split_mnist_tasks, SPLIT_MNIST_TASKS
from concept_cl.replay_buffer import UncertaintyReplayBuffer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_adaptive", action="store_true", default=True)
    parser.add_argument("--use_replay", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loaders, test_loaders = get_split_mnist_tasks(batch_size=args.batch_size)

    model = ConceptCLModel(in_channels=1, feat_dim=128, concept_dim=32, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    proto_store = ConceptPrototypeStore()
    replay_buf = UncertaintyReplayBuffer(per_class_capacity=50)

    initial_state = {name: p.detach().clone().cpu() for name, p in model.named_parameters()}

    param_drift_history, ksi_history, avg_acc_history = [], [], []

    for task_idx, (classes, train_loader) in enumerate(zip(SPLIT_MNIST_TASKS, train_loaders)):
        print(f"\n=== Task {task_idx}: classes {classes} ===")
        model.train()
        for epoch in range(args.epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                replay_batch = replay_buf.sample(args.batch_size // 2, device) if args.use_replay else None

                optimizer.zero_grad()
                loss, components, _ = total_loss(
                    model, x, y, proto_store,
                    replay_batch=replay_batch, use_adaptive=args.use_adaptive,
                )
                loss.backward()
                optimizer.step()
            print(f"  epoch {epoch}: {components}")

        # Register/update concept prototypes for this task's classes
        model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                _, concepts = model(x, sample=False)
                for c in classes:
                    mask = (y == c)
                    if mask.sum() > 0:
                        proto_store.register_or_update(c, concepts[mask], task_idx)
        model.train()

        # Update replay buffer with this task's most-uncertain examples
        if args.use_replay:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                replay_buf.add_task_data(model, x, y)
                break  # one batch is enough for a demo-scale buffer

        # Metrics
        accs = evaluate(model, test_loaders[:task_idx + 1], device)
        avg_acc = sum(accs) / len(accs)
        drift = parameter_distance(model, initial_state)
        ksi = proto_store.knowledge_stability_index()

        avg_acc_history.append(avg_acc)
        param_drift_history.append(drift)
        ksi_history.append(ksi)
        print(f"  avg_acc_so_far={avg_acc:.4f}  param_drift={drift:.3f}  KSI={ksi}")

    # --- The money plot ---
    os.makedirs("results", exist_ok=True)
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
    plt.savefig("results/drift_vs_ksi.png", dpi=150)
    print("\nSaved results/drift_vs_ksi.png")
    print("avg_acc_history:", avg_acc_history)


if __name__ == "__main__":
    main()
