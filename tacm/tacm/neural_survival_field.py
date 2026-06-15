"""
TAC-SM: Neural Survival Field (NSF) — PSM-Grounded Differentiable Module

Connects the PSM-004 survival field research to the TACSM PyTorch model.

The NSF computes a differentiable survival loss from procedure fitness profiles.
High-fitness procedure embeddings are rewarded; low-fitness embeddings are penalised.
This shapes which procedures persist in the Structure Memory during training.

Loss components:
  L_survival_field  = L_fitness_contrastive + L_decay_signal + L_robustness

Integration with existing tacm/survival.py:
  - tacm/survival.py handles lifecycle *state* transitions (symbolic)
  - tacm/neural_survival_field.py handles the *gradient* signal (differentiable)
  - The two are complementary, not competing
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fitness Encoder ───────────────────────────────────────────────────────────

class FitnessEncoder(nn.Module):
    """
    Maps a procedure fitness vector (5 components) to a scalar survival logit.

    Input:  [reuse_score, transfer_score, robustness, recovery, verify_score]
    Output: scalar fitness logit ∈ ℝ  (sigmoid → probability of survival)

    The weights correspond to PSM-004 component weights:
      w_reuse=0.25, w_transfer=0.25, w_robustness=0.20, w_recovery=0.15, w_verify=0.15
    """

    COMPONENT_DIM = 5
    COMPONENT_NAMES = ["reuse", "transfer", "robustness", "recovery", "verify"]

    # PSM-004 prior weights (used to initialise the linear layer)
    PRIOR_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.COMPONENT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise first layer with PSM-004 prior weights
        with torch.no_grad():
            w = torch.tensor(self.PRIOR_WEIGHTS, dtype=torch.float32).unsqueeze(0)
            self.net[0].weight.data[:1] = w
            self.net[0].bias.data.zero_()

    def forward(self, fitness_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fitness_vecs: (batch, 5) — normalised fitness components
        Returns:
            logits: (batch,) — survival logit
        """
        return self.net(fitness_vecs).squeeze(-1)


# ── Neural Survival Field ─────────────────────────────────────────────────────

class NeuralSurvivalField(nn.Module):
    """
    Differentiable counterpart to the PSM-004 SurvivalField.

    Computes three loss components:

    1. L_fitness_contrastive
       Contrastive loss: high-fitness embeddings should be closer to each other
       than to low-fitness embeddings. Uses InfoNCE-style (cosine + temperature).

    2. L_decay_signal
       High-fitness procedures are rewarded (pulled toward a survival centroid).
       Low-fitness procedures are penalised (pushed away from the centroid).
       Implemented as a signed margin loss.

    3. L_robustness
       Under Gaussian perturbation to embeddings, high-fitness procedures
       should maintain cosine similarity to their clean embedding.
       Low-fitness procedures need not. Implemented as a consistency loss.

    Total: L_nsf = w1 * L_fitness_contrastive
                 + w2 * L_decay_signal
                 + w3 * L_robustness
    """

    def __init__(
        self,
        embedding_dim:     int   = 512,
        fitness_hidden:    int   = 32,
        temperature:       float = 0.07,
        margin:            float = 0.20,
        noise_std:         float = 0.05,
        w_contrastive:     float = 0.40,
        w_decay:           float = 0.35,
        w_robustness:      float = 0.25,
        fitness_threshold: float = 0.45,   # cutoff from PSM-004
    ):
        super().__init__()
        self.embedding_dim     = embedding_dim
        self.temperature       = temperature
        self.margin            = margin
        self.noise_std         = noise_std
        self.w_contrastive     = w_contrastive
        self.w_decay           = w_decay
        self.w_robustness      = w_robustness
        self.fitness_threshold = fitness_threshold

        # Learnable fitness encoder (maps 5-dim fitness → scalar logit)
        self.fitness_encoder = FitnessEncoder(hidden_dim=fitness_hidden)

        # Learnable survival centroid (high-fitness procedures are pulled toward it)
        self.survival_centroid = nn.Parameter(
            torch.randn(embedding_dim) / math.sqrt(embedding_dim)
        )

        # Projection head (maps procedure embeddings to a survival-relevant space)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        embeddings:   torch.Tensor,    # (N, embedding_dim) — procedure embeddings
        fitness_vecs: torch.Tensor,    # (N, 5)             — fitness component vectors
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all NSF loss components.

        Args:
            embeddings:   Procedure embedding vectors, one per procedure. These
                          should be the outputs of the Structure Memory encoder.
            fitness_vecs: Five-component fitness vectors (reuse, transfer,
                          robustness, recovery, verify), normalised to [0, 1].

        Returns:
            dict with keys:
                loss_total          — weighted sum of all components
                loss_contrastive    — fitness contrastive loss
                loss_decay          — decay signal loss
                loss_robustness     — perturbation robustness loss
                fitness_logits      — scalar fitness logit per procedure
                survival_probs      — sigmoid of fitness logit
        """
        N = embeddings.size(0)
        if N == 0:
            zero = embeddings.new_zeros(())
            return {k: zero for k in ["loss_total", "loss_contrastive",
                                      "loss_decay", "loss_robustness"]}

        # Fitness logits and probabilities
        fitness_logits = self.fitness_encoder(fitness_vecs)       # (N,)
        survival_probs = torch.sigmoid(fitness_logits)            # (N,)

        # Binary high/low fitness mask (from PSM-004 cutoff)
        fitness_scalar = fitness_vecs @ fitness_vecs.new_tensor(
            FitnessEncoder.PRIOR_WEIGHTS
        )                                                          # (N,) weighted mean
        high_mask = (fitness_scalar >= self.fitness_threshold)    # (N,) bool

        # Project embeddings
        z = self.proj(F.normalize(embeddings, dim=-1))            # (N, emb_dim//4)
        z = F.normalize(z, dim=-1)

        # Compute each loss
        l_contrastive = self._fitness_contrastive(z, high_mask)
        l_decay       = self._decay_signal(embeddings, high_mask)
        l_robustness  = self._robustness_consistency(embeddings, high_mask)

        # Fitness discrimination: train encoder to predict high/low label.
        # This ensures fitness_encoder parameters always receive gradients.
        fitness_labels = high_mask.float()                        # (N,)
        l_fitness_disc = F.binary_cross_entropy_with_logits(
            fitness_logits, fitness_labels
        )

        l_total = (
            self.w_contrastive * l_contrastive
            + self.w_decay       * l_decay
            + self.w_robustness  * l_robustness
            + 0.10               * l_fitness_disc
        )

        return {
            "loss_total":         l_total,
            "loss_contrastive":   l_contrastive,
            "loss_decay":         l_decay,
            "loss_robustness":    l_robustness,
            "loss_fitness_disc":  l_fitness_disc,
            "fitness_logits":     fitness_logits,
            "survival_probs":     survival_probs,
        }

    def survival_loss(
        self,
        embeddings:   torch.Tensor,
        fitness_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper — returns just the total loss scalar."""
        return self.forward(embeddings, fitness_vecs)["loss_total"]

    def predict_survival(
        self,
        embeddings:   torch.Tensor,
        fitness_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-procedure survival probability ∈ [0, 1]."""
        return self.forward(embeddings, fitness_vecs)["survival_probs"]

    # ── Loss components ───────────────────────────────────────────────────────

    def _fitness_contrastive(
        self,
        z:         torch.Tensor,    # (N, d) — projected, normalised embeddings
        high_mask: torch.Tensor,    # (N,) bool
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.

        Positives: pairs of high-fitness procedures (should be similar).
        Negatives: high vs low-fitness pairs (should be different).

        If there are no positives or no negatives, returns 0.
        """
        N = z.size(0)
        if high_mask.sum() < 2 or (~high_mask).sum() < 1:
            return z.new_zeros(())

        # Compute pairwise cosine similarity matrix
        sim = (z @ z.T) / self.temperature                        # (N, N)

        # Mask diagonal (self-similarity)
        mask_eye = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_eye, float("-inf"))

        # For each high-fitness anchor, positives = other high-fitness,
        # negatives = all low-fitness
        high_idx = high_mask.nonzero(as_tuple=False).squeeze(-1)  # (H,)
        low_idx  = (~high_mask).nonzero(as_tuple=False).squeeze(-1) # (L,)

        losses = []
        for i in high_idx:
            pos_sims = sim[i, high_idx[high_idx != i]]            # (H-1,)
            neg_sims = sim[i, low_idx]                            # (L,)
            if pos_sims.numel() == 0 or neg_sims.numel() == 0:
                continue
            # InfoNCE: log(sum(exp(pos))) - log(sum(exp(pos)) + sum(exp(neg)))
            pos_exp = pos_sims.exp().sum()
            neg_exp = neg_sims.exp().sum()
            loss    = -(pos_exp / (pos_exp + neg_exp + 1e-9)).log()
            losses.append(loss)

        if not losses:
            return z.new_zeros(())
        return torch.stack(losses).mean()

    def _decay_signal(
        self,
        embeddings: torch.Tensor,   # (N, D) — raw embeddings
        high_mask:  torch.Tensor,   # (N,) bool
    ) -> torch.Tensor:
        """
        High-fitness: pull toward survival centroid.
        Low-fitness:  push away from survival centroid (margin loss).
        """
        centroid = F.normalize(self.survival_centroid, dim=0)     # (D,)
        emb_norm = F.normalize(embeddings, dim=-1)                # (N, D)
        cos_sim  = (emb_norm * centroid).sum(dim=-1)              # (N,)

        # High-fitness: maximise cosine similarity → loss = 1 - sim
        hi_loss = (1.0 - cos_sim[high_mask]).mean() if high_mask.any() else embeddings.new_zeros(())
        # Low-fitness: cosine similarity should be below margin → hinge
        lo_mask = ~high_mask
        lo_loss = (F.relu(cos_sim[lo_mask] - self.margin)).mean() if lo_mask.any() else embeddings.new_zeros(())

        return hi_loss + lo_loss

    def _robustness_consistency(
        self,
        embeddings: torch.Tensor,   # (N, D)
        high_mask:  torch.Tensor,   # (N,) bool
    ) -> torch.Tensor:
        """
        Under Gaussian noise, high-fitness embeddings should remain self-consistent.
        Low-fitness embeddings are not penalised for inconsistency.

        Loss = mean cosine distance between clean and noisy embedding,
               averaged over high-fitness procedures only.
        """
        if not high_mask.any():
            return embeddings.new_zeros(())

        hi_emb   = embeddings[high_mask]                          # (H, D)
        noise    = torch.randn_like(hi_emb) * self.noise_std
        noisy    = hi_emb + noise

        clean_n  = F.normalize(hi_emb, dim=-1)
        noisy_n  = F.normalize(noisy, dim=-1)
        cos_dist = 1.0 - (clean_n * noisy_n).sum(dim=-1)         # (H,)

        return cos_dist.mean()


# ── Integration Helper ────────────────────────────────────────────────────────

def make_fitness_vecs(
    reuse_counts:   torch.Tensor,    # (N,) int
    transfer_scores: torch.Tensor,   # (N,) float ∈ [0,1]
    robustness:     torch.Tensor,    # (N,) float ∈ [0,1]
    recovery_rates: torch.Tensor,    # (N,) float ∈ [0,1]
    verify_scores:  torch.Tensor,    # (N,) float ∈ [0,1]
    max_reuse:      int = 20,
) -> torch.Tensor:
    """
    Build a (N, 5) fitness vector tensor from raw procedure attributes.

    Normalises reuse_count to [0, 1] and stacks all components.
    Matches the PSM-004 FitnessProfile component definition exactly.
    """
    reuse_norm = (reuse_counts.float() / max(max_reuse, 1)).clamp(0, 1)
    return torch.stack([
        reuse_norm,
        transfer_scores,
        robustness,
        recovery_rates,
        verify_scores,
    ], dim=-1)


# ── Survival Loss for TACSM training loop ────────────────────────────────────

class SurvivalFieldLoss(nn.Module):
    """
    Drop-in loss module for the TACSM training loop.

    Usage in losses.py:
        from tacm.neural_survival_field import SurvivalFieldLoss
        nsf_loss = SurvivalFieldLoss(cfg.d_model)

        # In training loop:
        l_nsf = nsf_loss(
            procedure_embeddings = memory.get_embeddings(),    # (N, D)
            fitness_vecs         = memory.get_fitness_vecs(),  # (N, 5)
        )
        loss_total += cfg.w_survival_field * l_nsf

    The loss is zero when there are no procedures in memory (first steps).
    """

    def __init__(
        self,
        embedding_dim:  int   = 512,
        fitness_hidden: int   = 32,
        temperature:    float = 0.07,
        margin:         float = 0.20,
        noise_std:      float = 0.05,
    ):
        super().__init__()
        self.nsf = NeuralSurvivalField(
            embedding_dim  = embedding_dim,
            fitness_hidden = fitness_hidden,
            temperature    = temperature,
            margin         = margin,
            noise_std      = noise_std,
        )

    def forward(
        self,
        procedure_embeddings: torch.Tensor,   # (N, D)
        fitness_vecs:         torch.Tensor,   # (N, 5)
    ) -> torch.Tensor:
        if procedure_embeddings.size(0) == 0:
            return procedure_embeddings.new_zeros(())
        return self.nsf.survival_loss(procedure_embeddings, fitness_vecs)
