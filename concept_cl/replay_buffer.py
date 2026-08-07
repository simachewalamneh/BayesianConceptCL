"""
Replay buffer that stores samples flagged by predictive uncertainty rather
than random/reservoir sampling -- keeps the examples the model is least
confident about, per class.
"""

import torch


class UncertaintyReplayBuffer:
    def __init__(self, per_class_capacity: int = 50):
        self.per_class_capacity = per_class_capacity
        self.buffer_x = {}   # class_id -> tensor of stored inputs
        self.buffer_y = {}   # class_id -> tensor of stored labels

    @torch.no_grad()
    def add_task_data(self, model, x: torch.Tensor, y: torch.Tensor, n_mc_samples: int = 5):
        uncertainty = model.predictive_uncertainty(x, n_samples=n_mc_samples)
        for class_id in y.unique().tolist():
            mask = (y == class_id)
            class_x, class_u = x[mask], uncertainty[mask]
            k = min(self.per_class_capacity, class_x.shape[0])
            top_idx = torch.topk(class_u, k).indices
            self.buffer_x[class_id] = class_x[top_idx].cpu()
            self.buffer_y[class_id] = torch.full((k,), class_id, dtype=torch.long)

    def sample(self, batch_size: int, device):
        if not self.buffer_x:
            return None
        all_x = torch.cat(list(self.buffer_x.values()), dim=0)
        all_y = torch.cat(list(self.buffer_y.values()), dim=0)
        if all_x.shape[0] == 0:
            return None
        idx = torch.randperm(all_x.shape[0])[:batch_size]
        return all_x[idx].to(device), all_y[idx].to(device)
