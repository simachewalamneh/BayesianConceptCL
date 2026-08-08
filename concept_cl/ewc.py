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
        Sum over all parameters (the standard EWC formulation), relying on
        gradient clipping (applied in the training loop) for numerical
        stability instead of pre-shrinking the loss via averaging.

        NOTE on why this changed twice: the original sum-based version
        exploded (lambda=100 was far too large relative to an unclipped
        sum over hundreds of thousands of parameters). The mean-based fix
        that followed overcorrected in the OPPOSITE direction: dividing by
        total parameter count -- dominated by large, near-zero-Fisher
        layers like the encoder's projection layer (~400K params) -- diluted
        the signal from the small number of genuinely important
        (high-Fisher) parameters down to ~0, giving effectively zero
        protection and full catastrophic forgetting. Diagnostic run
        confirmed: ewc_penalty printed as 0.0000 every batch, and Task 0
        accuracy collapsed to 0.0 after Task 1 -- textbook unregularized
        forgetting. Reverting to sum + relying on the already-added
        grad-norm clipping (max_norm=5.0) fixes both failure modes at once.
        """
        if not self._has_task:
            return torch.tensor(0.0)
        loss = 0.0
        for n, p in self.model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()
        return self.lambda_ewc * loss
