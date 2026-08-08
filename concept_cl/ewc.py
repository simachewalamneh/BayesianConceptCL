"""
Classic Elastic Weight Consolidation (Kirkpatrick et al. 2017), used here
purely as a LITERATURE BASELINE to compare against concept-stability
consolidation on the same accuracy/KSI metrics.

After each task, EWCHelper computes a diagonal Fisher Information
approximation (mean squared gradient of the log-likelihood over a batch of
that task's data) and stores the parameter values at that point ("star"
values). The penalty term is:

    L_EWC = sum_i  lambda * F_i * (theta_i - theta_star_i)^2

summed (not just latest task -- Fisher accumulates additively across tasks,
the standard online-EWC-lite approach) over all parameters, for every
previously completed task.
"""

import torch
import torch.nn.functional as F


class EWCHelper:
    def __init__(self, model, lambda_ewc: float = 100.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self.star_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._has_task = False

    @torch.no_grad()
    def _update_star(self):
        for n, p in self.model.named_parameters():
            self.star_params[n] = p.detach().clone()

    def register_task(self, dataloader, device, n_batches: int = 5):
        """
        Call once after finishing training on a task. Accumulates Fisher
        information (summed across tasks -- standard "EWC" as opposed to
        the more complex per-task-penalty variant) and updates star params
        to the current (just-finished-training) values.
        """
        self.model.eval()
        new_fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        seen = 0
        for i, (x, y) in enumerate(dataloader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            logits, _ = self.model(x, sample=False)
            log_probs = F.log_softmax(logits, dim=-1)
            # sample-based Fisher: use the model's own predicted labels
            sampled_y = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
            loss = F.nll_loss(log_probs, sampled_y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += p.grad.detach() ** 2
            seen += 1
        if seen > 0:
            for n in new_fisher:
                new_fisher[n] /= seen

        # Accumulate additively across tasks (simple online-EWC-lite)
        for n in self.fisher:
            self.fisher[n] += new_fisher[n]

        self._update_star()
        self._has_task = True
        self.model.train()

    def penalty(self) -> torch.Tensor:
        """
        Mean squared penalty across ALL parameter elements combined
        (not summed) -- summing over thousands of individual weights times
        a large lambda caused gradient explosion and near-chance accuracy
        in the first run. Averaging keeps the scale sane regardless of
        how many parameters the model has.
        """
        if not self._has_task:
            return torch.tensor(0.0)
        total_sq = 0.0
        total_count = 0
        for n, p in self.model.named_parameters():
            total_sq = total_sq + (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()
            total_count += p.numel()
        return self.lambda_ewc * total_sq / max(total_count, 1)
