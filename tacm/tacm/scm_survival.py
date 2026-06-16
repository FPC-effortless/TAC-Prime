"""
TAC-SCM-REAL001: NSF Survival Scorer

Scores how worthy each discovered structure is of being kept, written to memory,
and refined.  Outputs per-structure gates that downstream components use.

Survival formula (configurable weighted sum):
    survival = reuse_w * reuse
             + transfer_w * transfer
             + robustness_w * robustness
             + compression_w * compression
             - cost_w * cost
             - interference_w * interference

Reuses the NeuralSurvivalField from neural_survival_field.py for the
differentiable survival loss, wrapping it with the additional signals.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_types import SurvivalOutput
from .scm_config import TACSCMConfig
from .neural_survival_field import NeuralSurvivalField, make_fitness_vecs


class NSFSurvivalScorer(nn.Module):
    """
    Scores structure embeddings for survival using configurable weighted signals.

    Inputs (all optional; defaults to 0.5 when not provided)
    ---------------------------------------------------------
    structure_embeddings : (N, d_structure)
    reuse_signal         : (N,) float ∈ [0,1]
    transfer_signal      : (N,) float ∈ [0,1]
    robustness_signal    : (N,) float ∈ [0,1]
    compression_signal   : (N,) float ∈ [0,1]
    cost_signal          : (N,) float ∈ [0,1]
    interference_signal  : (N,) float ∈ [0,1]

    Outputs
    -------
    SurvivalOutput with survival_score, gates, keep_mask, losses
    """

    # Default survival formula weights
    DEFAULT_WEIGHTS = dict(
        reuse=0.30, transfer=0.25, robustness=0.20,
        compression=0.15, cost=0.05, interference=0.05,
    )

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        self.cfg = cfg
        D        = cfg.d_structure

        # Learnable signal encoder: maps raw 6-dim signal → d_structure
        self.signal_encoder = nn.Sequential(
            nn.Linear(6, D // 2, bias=True),
            nn.GELU(),
            nn.Linear(D // 2, D, bias=False),
            nn.LayerNorm(D),
        )

        # Survival head: fused (structure + signal) → scalar score
        self.survival_head = nn.Sequential(
            nn.Linear(D * 2, D, bias=False),
            nn.GELU(),
            nn.Linear(D, 1, bias=True),
        )

        # Dedicated gate heads (separate from main score so they can specialise)
        self.decay_gate_head  = nn.Linear(D, 1, bias=True)
        self.write_gate_head  = nn.Linear(D, 1, bias=True)
        self.refine_gate_head = nn.Linear(D, 1, bias=True)

        # Reuse the NSF for the differentiable loss component
        self.nsf = NeuralSurvivalField(
            embedding_dim     = D,
            fitness_hidden    = max(16, D // 4),
            temperature       = 0.07,
            margin            = 0.20,
            noise_std         = 0.05,
            w_contrastive     = 0.40,
            w_decay           = 0.35,
            w_robustness      = 0.25,
        )

        # Survival threshold for keep_mask
        self.survival_threshold = 0.4

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        structure_embeddings: torch.Tensor,                  # (N, D)
        reuse_signal:         Optional[torch.Tensor] = None, # (N,)
        transfer_signal:      Optional[torch.Tensor] = None,
        robustness_signal:    Optional[torch.Tensor] = None,
        compression_signal:   Optional[torch.Tensor] = None,
        cost_signal:          Optional[torch.Tensor] = None,
        interference_signal:  Optional[torch.Tensor] = None,
    ) -> SurvivalOutput:

        N, D   = structure_embeddings.shape
        device = structure_embeddings.device

        def _default(t: Optional[torch.Tensor], val: float = 0.5) -> torch.Tensor:
            return t if t is not None else structure_embeddings.new_full((N,), val)

        reuse_s       = _default(reuse_signal,       0.5)
        transfer_s    = _default(transfer_signal,    0.5)
        robustness_s  = _default(robustness_signal,  0.5)
        compression_s = _default(compression_signal, 0.5)
        cost_s        = _default(cost_signal,        0.1)
        interference_s= _default(interference_signal,0.1)

        # ── 1. Weighted survival score (heuristic baseline) ───────────────────
        w  = self.DEFAULT_WEIGHTS
        survival_heuristic = (
            w["reuse"]        * reuse_s
            + w["transfer"]   * transfer_s
            + w["robustness"] * robustness_s
            + w["compression"]* compression_s
            - w["cost"]       * cost_s
            - w["interference"]* interference_s
        ).clamp(0, 1)                                        # (N,)

        # ── 2. Learnable survival score ───────────────────────────────────────
        signal_vec = torch.stack([
            reuse_s, transfer_s, robustness_s,
            compression_s, cost_s, interference_s,
        ], dim=-1)                                           # (N, 6)
        signal_enc = self.signal_encoder(signal_vec)         # (N, D)

        fused = torch.cat([
            F.normalize(structure_embeddings, dim=-1),
            signal_enc,
        ], dim=-1)                                           # (N, 2D)
        survival_logit  = self.survival_head(fused).squeeze(-1)  # (N,)
        survival_learned = torch.sigmoid(survival_logit)         # (N,)

        # Blend heuristic and learned
        survival_score = 0.5 * survival_heuristic + 0.5 * survival_learned  # (N,)

        # ── 3. Gate heads ─────────────────────────────────────────────────────
        struct_n    = F.normalize(structure_embeddings, dim=-1)
        decay_gate  = torch.sigmoid(self.decay_gate_head(struct_n).squeeze(-1))   # low = strong
        write_gate  = torch.sigmoid(self.write_gate_head(struct_n).squeeze(-1))
        refine_gate = torch.sigmoid(self.refine_gate_head(struct_n).squeeze(-1))

        # Scale gates by survival score
        decay_gate  = decay_gate  * (1.0 - survival_score)
        write_gate  = write_gate  * survival_score
        refine_gate = refine_gate * survival_score

        # ── 4. Keep mask ──────────────────────────────────────────────────────
        keep_mask = survival_score >= self.survival_threshold

        # ── 5. NSF differentiable loss ────────────────────────────────────────
        fitness_vecs = make_fitness_vecs(
            reuse_counts    = (reuse_s * 20).long(),
            transfer_scores = transfer_s,
            robustness      = robustness_s,
            recovery_rates  = robustness_s,   # proxy
            verify_scores   = compression_s,
            max_reuse       = 20,
        )                                                    # (N, 5)

        if N >= 2:
            nsf_out = self.nsf(structure_embeddings, fitness_vecs)
            loss_nsf = nsf_out["loss_total"]
        else:
            loss_nsf = structure_embeddings.new_zeros(())

        # Auxiliary: survival score should predict heuristic
        loss_aux = F.mse_loss(survival_learned, survival_heuristic.detach())
        loss_total = self.cfg.survival_loss_weight * (loss_nsf + 0.1 * loss_aux)

        return SurvivalOutput(
            survival_score = survival_score,
            decay_gate     = decay_gate,
            write_gate     = write_gate,
            refine_gate    = refine_gate,
            keep_mask      = keep_mask,
            loss_survival  = loss_nsf,
            loss_total     = loss_total,
        )
