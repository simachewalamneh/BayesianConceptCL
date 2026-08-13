import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayesian_layers import BayesianLinear


class FeatureExtractor(nn.Module):
    
    def __init__(self, in_channels: int = 1, feat_dim: int = 128, input_size: int = 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # two stride-2 maxpools -> input_size // 4 spatial dims, 64 channels
        conv_out_dim = 64 * (input_size // 4) * (input_size // 4)
        self.proj = nn.Linear(conv_out_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return F.relu(self.proj(h))


class ConceptCLModel(nn.Module):
    def __init__(self, in_channels: int = 1, feat_dim: int = 128,
                 concept_dim: int = 32, num_classes: int = 10, prior_sigma: float = 1.0,
                 input_size: int = 28):
        super().__init__()
        self.encoder = FeatureExtractor(in_channels, feat_dim, input_size=input_size)

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
   

    def __init__(self):
        self.prototypes = {}       # class_id -> frozen prototype (torch.Tensor)
        self.history = {}          # class_id -> list of (task_idx, prototype) for KSI tracking

    @torch.no_grad()
    def register_or_update(self, class_id: int, concept_vectors: torch.Tensor, task_idx: int):
     
        mean_vec = concept_vectors.mean(dim=0).detach().cpu()
        if class_id not in self.prototypes:
            self.prototypes[class_id] = mean_vec
            self.history[class_id] = [(task_idx, mean_vec)]
        else:
            self.history[class_id].append((task_idx, mean_vec))

    def get_prototype(self, class_id: int):
        return self.prototypes.get(class_id, None)

    def _per_class_similarities(self) -> dict:
        
        sims = {}
        for class_id, hist in self.history.items():
            if len(hist) < 2:
                continue
            original = hist[0][1]
            latest = hist[-1][1]
            sims[class_id] = F.cosine_similarity(original.unsqueeze(0), latest.unsqueeze(0)).item()
        return sims

    def knowledge_stability_index(self) -> float:
     
        sims = list(self._per_class_similarities().values())
        return sum(sims) / len(sims) if sims else float("nan")

    def knowledge_stability_stats(self) -> dict:
       
        sim_dict = self._per_class_similarities()
        if not sim_dict:
            return {}
        sims = list(sim_dict.values())
        n = len(sims)
        mean = sum(sims) / n
        sorted_sims = sorted(sims)
        median = (sorted_sims[n // 2] if n % 2 == 1
                  else (sorted_sims[n // 2 - 1] + sorted_sims[n // 2]) / 2)
        variance = sum((s - mean) ** 2 for s in sims) / n
        std = variance ** 0.5
        min_class_id = min(sim_dict, key=sim_dict.get)
        return {
            "mean": mean, "median": median, "min": min(sims), "std": std,
            "n_classes": n, "min_class_id": min_class_id,
        }
