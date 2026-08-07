# Bayesian Concept Consolidation (BCC)

**Central hypothesis:** catastrophic forgetting is better explained by *drift
in semantic concepts* than by drift in raw parameters. Standard methods
(EWC, SI, Replay) protect weights or samples; this protects what a class
*means* to the network.

## Pipeline
```
Input -> Feature Extractor -> Bayesian Latent Layer -> Concept Layer -> Task Head
```
Each class gets a frozen concept prototype the first time it's learned.
Later tasks are penalized for letting that class's concept vector drift
away from its prototype (`L_concept`), with the penalty strength set
adaptively by the concept layer's own posterior variance
(`lambda_i = 1 / (sigma_i^2 + eps)`).

## Loss
```
L = L_task + lambda1 * L_KL + lambda2 * L_concept + lambda3 * L_replay
```

## The key evaluation: Knowledge Stability Index (KSI)
`KSI = mean cosine similarity between each class's original and current
concept prototype.` KSI -> 1 means concepts are preserved; KSI -> 0 means
semantic drift despite whatever the parameter-level numbers say.

**The figure that matters:** plot parameter drift (L2 distance from
initial weights) alongside KSI across tasks. The hypothesis is confirmed
if parameter drift keeps growing while KSI stays high — i.e., the network
keeps changing at the weight level but preserves what it "knows."
`scripts/train.py` produces this plot automatically (`results/drift_vs_ksi.png`).

## Repo layout
```
concept_cl/
  bayesian_layers.py   # Bayes-by-Backprop variational linear layer
  model.py              # encoder -> Bayesian latent -> concept layer -> head
                         # + ConceptPrototypeStore (prototypes + KSI)
  losses.py              # L_task, L_KL, L_concept, adaptive weighting, replay
  replay_buffer.py       # uncertainty-prioritized replay buffer
  data.py                 # Split-MNIST (5 tasks, 2 classes each)
scripts/
  train.py                # full training loop, produces the money plot
results/                  # saved plots / metrics land here
```

## Run
```bash
pip install -r requirements.txt
python scripts/train.py --epochs 3 --device cpu
```

## Roadmap
- [x] Phase 1 — Bayesian backbone + predictive uncertainty
- [x] Phase 2 — Concept bottleneck + prototype tracking
- [x] Phase 3 — Concept stability loss
- [x] Phase 4 — Adaptive plasticity (posterior-variance-weighted)
- [x] Phase 5 — Uncertainty-prioritized replay
- [ ] Phase 6 — Full evaluation: ACC / BWT / FWT / Forgetting / ECE, baselines
      (naive, EWC, VCL — reuse numbers from `bnn-study`), ablations
      (remove L_concept / adaptive weighting / uncertainty replay one at a time)
- [ ] Move from Split-MNIST to Split-CIFAR10 once the pipeline is validated
- [ ] t-SNE visualization of concept space at task 1 vs. task 5

## Status
Scaffolded and syntax-checked; not yet run end-to-end (no GPU/torch in the
environment this was written in) — validate a first pass locally before
trusting the numbers, and check for shape/device bugs on the first run.
