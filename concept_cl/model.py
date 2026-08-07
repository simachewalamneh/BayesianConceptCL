"""
Bayesian Concept Consolidation (BCC) model.

Pipeline:
    Input -> Feature Extractor -> Bayesian Latent Layer -> Concept Layer -> Task Head

The Concept Layer projects the Bayesian latent into K "concept slots"
(one prototype per class seen so far). A ConceptPrototypeStore tracks
each class's concept vector at the moment it was first learned, frozen
thereafter, and used as the anchor for the concept-stability loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayesian_layers import BayesianLinear


class FeatureExtractor(nn.Module):
    """Small CNN, swap for something bigger if you move past MNIST/CIFAR."""

    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self._feat_dim = feat_dim
        self.proj = None  # lazy init once we know the flattened size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        if self.proj is None:
            self.proj = nn.Linear(h.shape[1], self._feat_dim).to(h.device)
        return F.relu(self.proj(h))


class ConceptCLModel(nn.Module):
    def __init__(self, in_channels: int = 1, feat_dim: int = 128,
                 concept_dim: int = 32, num_classes: int = 10, prior_sigma: float = 1.0):
        super().__init__()
        self.encoder = FeatureExtractor(in_channels, feat_dim)

        # Bayesian latent layer: produces distribution over latent z
        self.bayes_latent = BayesianLinear(feat_dim, feat_dim, prior_sigma=prior_sigma)

        # Concept layer: latent -> concept space (this is what gets protected,
        # not the raw weights)
        self.concept_layer = BayesianLinear(feat_dim, concept_dim, prior_sigma=prior_sigma)

        self.task_head = nn.Linear(concept_dim, num_classes)

        self.concept_dim = concept_dim

    def forward(self, x: torch.Tensor, sample: bool = True):
        h = self.encoder(x)
        z = F.relu(self.bayes_latent(h, sample=sample))
        c = self.concept_layer(z, sample=sample)  # concept vector
        logits = self.task_head(c)
        return logits, c

    def kl_divergence(self) -> torch.Tensor:
        return self.bayes_latent.kl_divergence() + self.concept_layer.kl_divergence()

    def concept_posterior_variance(self) -> torch.Tensor:
        """Per-concept-dim variance -> feeds adaptive plasticity weight lambda_i."""
        return self.concept_layer.posterior_variance()  # shape: (concept_dim,)

    def predictive_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """MC sampling -> per-example predictive entropy (epistemic uncertainty)."""
        probs = []
        for _ in range(n_samples):
            logits, _ = self.forward(x, sample=True)
            probs.append(F.softmax(logits, dim=-1))
        mean_probs = torch.stack(probs).mean(0)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        return entropy


class ConceptPrototypeStore:
    """
    Tracks one frozen concept prototype per class: the class's mean concept
    vector at the moment it was first learned. This is the anchor for
    L_concept and the basis for the Knowledge Stability Index (KSI).
    """

    def __init__(self):
        self.prototypes = {}       # class_id -> frozen prototype (torch.Tensor)
        self.history = {}          # class_id -> list of (task_idx, prototype) for KSI tracking

    @torch.no_grad()
    def register_or_update(self, class_id: int, concept_vectors: torch.Tensor, task_idx: int):
        """
        Call once per task on the current concept vectors for that class.
        First call for a class_id freezes the prototype. Later calls only
        log to history (for KSI) — they do NOT move the frozen prototype.
        """
        mean_vec = concept_vectors.mean(dim=0).detach().cpu()
        if class_id not in self.prototypes:
            self.prototypes[class_id] = mean_vec
            self.history[class_id] = [(task_idx, mean_vec)]
        else:
            self.history[class_id].append((task_idx, mean_vec))

    def get_prototype(self, class_id: int):
        return self.prototypes.get(class_id, None)

    def knowledge_stability_index(self) -> float:
        """
        KSI = mean cosine similarity between each class's original prototype
        and its most recent observed concept vector, averaged over all
        classes with at least 2 recorded points.
        KSI -> 1: concepts unchanged. KSI -> 0: concepts drifted.
        """
        sims = []
        for class_id, hist in self.history.items():
            if len(hist) < 2:
                continue
            original = hist[0][1]
            latest = hist[-1][1]
            sim = F.cosine_similarity(original.unsqueeze(0), latest.unsqueeze(0)).item()
            sims.append(sim)
        return sum(sims) / len(sims) if sims else float("nan")
