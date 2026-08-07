"""
Bayesian variational linear layer (Bayes-by-Backprop style).

Weights are distributions, not point estimates:
    w ~ N(mu, softplus(rho)^2)

This is the stochastic backbone that produces the epistemic uncertainty
signal used throughout the framework (KL loss, adaptive plasticity,
uncertainty-weighted replay).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))

        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))

    def _sigma(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample or self.training:
            w_sigma = self._sigma(self.weight_rho)
            b_sigma = self._sigma(self.bias_rho)
            w = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
            b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(w) || p(w)) with a zero-mean Gaussian prior N(0, prior_sigma^2)."""
        w_sigma = self._sigma(self.weight_rho)
        b_sigma = self._sigma(self.bias_rho)

        kl_w = self._kl_gaussian(self.weight_mu, w_sigma, self.prior_sigma)
        kl_b = self._kl_gaussian(self.bias_mu, b_sigma, self.prior_sigma)
        return kl_w + kl_b

    @staticmethod
    def _kl_gaussian(mu: torch.Tensor, sigma: torch.Tensor, prior_sigma: float) -> torch.Tensor:
        prior_var = prior_sigma ** 2
        kl = torch.log(prior_sigma / sigma) + (sigma ** 2 + mu ** 2) / (2 * prior_var) - 0.5
        return kl.sum()

    def posterior_variance(self) -> torch.Tensor:
        """Per-output-unit variance, used for adaptive plasticity (lambda_i = 1 / (sigma_i^2 + eps))."""
        w_sigma = self._sigma(self.weight_rho)
        return (w_sigma ** 2).mean(dim=1)  # shape: (out_features,)
