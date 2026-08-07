"""
Loss terms for Bayesian Concept Consolidation.

    L = L_task + lambda1 * L_KL + lambda2 * L_concept + lambda3 * L_replay

L_concept and the adaptive lambda2_i (per-concept-dim) are the core
novel pieces; L_task and L_KL are standard.
"""

import torch
import torch.nn.functional as F


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def concept_stability_loss(current_concepts: torch.Tensor, targets: torch.Tensor,
                            prototype_store, mode: str = "cosine") -> torch.Tensor:
    """
    Penalizes drift between each sample's current concept vector and its
    class's frozen prototype (skips classes with no prototype yet, i.e.
    brand-new classes being learned for the first time).
    """
    losses = []
    for i in range(current_concepts.shape[0]):
        class_id = int(targets[i].item())
        proto = prototype_store.get_prototype(class_id)
        if proto is None:
            continue
        proto = proto.to(current_concepts.device)
        if mode == "cosine":
            losses.append(1 - F.cosine_similarity(current_concepts[i:i + 1], proto.unsqueeze(0)))
        else:
            losses.append(F.mse_loss(current_concepts[i], proto))
    if not losses:
        return torch.tensor(0.0, device=current_concepts.device)
    return torch.stack([l.squeeze() for l in losses]).mean()


def adaptive_concept_weight(posterior_variance: torch.Tensor, eps: float = 1e-3,
                             max_weight: float = 10.0) -> torch.Tensor:
    """
    lambda_i = 1 / (sigma_i^2 + eps), clamped to [0, max_weight].
    High certainty (low variance) -> large penalty -> protect concept.
    High uncertainty (high variance) -> small penalty -> allow adaptation.
    Without clamping, this exploded to ~1000x early in training (posterior
    variance starts near-zero from the rho init), swamping the task loss
    entirely and causing task accuracy to collapse across tasks instead of
    the model actually learning new classes.
    """
    raw = 1.0 / (posterior_variance + eps)
    return torch.clamp(raw, max=max_weight)


def uncertainty_weighted_replay_loss(model, replay_x: torch.Tensor, replay_y: torch.Tensor,
                                      n_mc_samples: int = 5) -> torch.Tensor:
    """
    Weight each replay sample's loss by its current predictive uncertainty
    (epistemic entropy from MC sampling) -- prioritizes samples the model
    is at risk of forgetting over ones it still handles confidently.
    """
    with torch.no_grad():
        uncertainty = model.predictive_uncertainty(replay_x, n_samples=n_mc_samples)
        weights = uncertainty / (uncertainty.sum() + 1e-8) * replay_x.shape[0]

    logits, _ = model(replay_x, sample=True)
    per_sample_loss = F.cross_entropy(logits, replay_y, reduction="none")
    return (weights * per_sample_loss).mean()


def total_loss(model, x, y, prototype_store,
                lambda1_kl: float = 1e-4, lambda2_concept: float = 1.0,
                lambda3_replay: float = 1.0,
                replay_batch=None, use_adaptive: bool = True):
    logits, concepts = model(x, sample=True)

    l_task = task_loss(logits, y)
    l_kl = model.kl_divergence() * lambda1_kl

    # Concept-stability loss on the CURRENT batch will almost always be 0 in
    # class-incremental settings like Split-MNIST, since a task's own batch
    # only contains classes with no prototype yet. The real protective
    # signal has to come from the REPLAY batch, since that's the only place
    # old-task data (with existing prototypes) flows through the model
    # during later tasks -- so we compute it on both and sum them.
    l_concept_current = concept_stability_loss(concepts, y, prototype_store)

    l_concept_replay = torch.tensor(0.0, device=x.device)
    replay_concepts = None
    if replay_batch is not None:
        rx, ry = replay_batch
        replay_logits, replay_concepts = model(rx, sample=True)
        l_concept_replay = concept_stability_loss(replay_concepts, ry, prototype_store)

    l_concept_raw = l_concept_current + l_concept_replay
    if use_adaptive:
        post_var = model.concept_posterior_variance()
        adaptive_w = adaptive_concept_weight(post_var).mean()
        l_concept = adaptive_w * l_concept_raw
    else:
        l_concept = lambda2_concept * l_concept_raw

    l_replay = torch.tensor(0.0, device=x.device)
    if replay_batch is not None:
        rx, ry = replay_batch
        l_replay = lambda3_replay * uncertainty_weighted_replay_loss(model, rx, ry)

    loss = l_task + l_kl + l_concept + l_replay
    components = {
        "task": l_task.item(), "kl": l_kl.item(),
        "concept": l_concept.item(), "replay": l_replay.item() if torch.is_tensor(l_replay) else l_replay,
    }
    return loss, components, concepts