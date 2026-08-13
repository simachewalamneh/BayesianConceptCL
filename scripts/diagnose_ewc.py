import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_cl.model import ConceptCLModel
from concept_cl.ewc import EWCHelper
from concept_cl.data import get_split_mnist_tasks
from scripts.train import evaluate, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_ewc", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")
    set_seed(args.seed)
    train_loaders, test_loaders = get_split_mnist_tasks(batch_size=128)

    model = ConceptCLModel(in_channels=1, feat_dim=128, concept_dim=32, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ewc_helper = EWCHelper(model, lambda_ewc=args.lambda_ewc)

    print(f"lambda_ewc = {args.lambda_ewc}\n")
    print("=== Task 0 (training normally, no EWC penalty active yet) ===")
    last_loss = None
    for x, y in train_loaders[0]:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, sample=False)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        last_loss = loss.item()
    print(f"  final Task 0 batch loss: {last_loss:.4f}")

    accs0 = evaluate(model, test_loaders[:1], device)
    print(f"  Task 0 test accuracy right after training: {accs0}")

    ewc_helper.register_task(train_loaders[0], device)
    fisher_vals = torch.cat([f.flatten() for f in ewc_helper.fisher.values()])
    print(f"  Fisher stats over all params: mean={fisher_vals.mean().item():.6e}  "
          f"max={fisher_vals.max().item():.6e}  min={fisher_vals.min().item():.6e}")

    print("\n=== Task 1 (EWC penalty now active) ===")
    for i, (x, y) in enumerate(train_loaders[1]):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, sample=False)
        task_loss = F.cross_entropy(logits, y)
        penalty = ewc_helper.penalty()
        total = task_loss + penalty
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if i < 10 or i % 20 == 0:
            ratio = penalty.item() / (task_loss.item() + 1e-8)
            print(f"  batch {i:3d}: task_loss={task_loss.item():.4f}  "
                  f"ewc_penalty={penalty.item():.4f}  ratio={ratio:.1f}x  "
                  f"grad_norm_pre_clip={grad_norm.item():.4f}")

    accs = evaluate(model, test_loaders[:2], device)
    print(f"\nAfter Task 1: per-task test accuracy on [task0, task1] = {accs}")
    print("(if task1 accuracy is near 0.5 / chance-level for 2 classes, the model")
    print(" essentially isn't learning task 1 at all -- penalty is too strong.")
    print(" if task0 accuracy collapsed toward 0, the penalty isn't protecting")
    print(" old knowledge at all -- something else is wrong or it's too weak.)")


if __name__ == "__main__":
    main()
