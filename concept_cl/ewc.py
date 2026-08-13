
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
      
        if not self._has_task:
            return torch.tensor(0.0)
        loss = 0.0
        for n, p in self.model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()
        return self.lambda_ewc * loss
