---
name: TAC-SM Neural Survival Field differentiable module
description: Gradient/design notes for tacm/tacm/neural_survival_field.py — the PyTorch NSF module connecting PSM-004 fitness to TACSM training.
---

# Neural Survival Field — Design Notes

## Rule
`FitnessEncoder` parameters only receive gradients if a fitness discrimination auxiliary loss (BCE on binary high/low mask) is included. Without it, the encoder is not on the gradient graph of the geometric losses.

**Why:** The contrastive, decay, and robustness losses use the fitness mask *as a boolean*, derived from `fitness_vecs @ prior_weights`. The `fitness_encoder` network produces `fitness_logits` which go into `survival_probs`, but those probs are not used by the geometric losses unless explicitly wired in. A 0.10-weighted BCE loss (logits vs binary label) closes the loop.

**How to apply:** The `forward()` method includes `loss_fitness_disc` in the returned dict and adds `0.10 * l_fitness_disc` to `l_total`. Verify with `all(p.grad is not None for p in nsf.parameters())` after `.backward()`.

## 4-component loss structure
1. `loss_contrastive` (w=0.40) — InfoNCE: hi-fit pairs similar, hi vs lo-fit different
2. `loss_decay` (w=0.35) — hi-fit pulled to survival centroid; lo-fit pushed away (margin)
3. `loss_robustness` (w=0.25) — hi-fit embeddings consistent under Gaussian noise
4. `loss_fitness_disc` (w=0.10) — BCE trains FitnessEncoder; ensures full gradient flow

## PSM-004 prior weights
`FitnessEncoder` first linear layer initialised with PSM-004 weights: [0.25, 0.25, 0.20, 0.15, 0.15] (reuse, transfer, robustness, recovery, verify).

## Integration in TACSM training loop
```python
from tacm.neural_survival_field import SurvivalFieldLoss
nsf_loss = SurvivalFieldLoss(cfg.d_model)
l_nsf = nsf_loss(memory.get_embeddings(), memory.get_fitness_vecs())
loss_total += cfg.w_survival_field * l_nsf
```

## Module entry points (both validated)
- `cd tacm && python -m scripts.run_psm_progression --seeds 0 1 2 3 4`
- `cd tacm && python -m tacm.scripts.run_psm_progression --seeds 0 1 2 3 4`
  (forwarding module at `tacm/tacm/scripts/run_psm_progression.py` + `__init__.py`)
