"""
TAC-SCM-REAL001: DPSL Refinement Layer

Dynamic Procedural Structure Learning (DPSL) refines structure embeddings
based on feedback, survival scores, and inter-structure similarity.

Operations
----------
1. Gated update     : blend incoming feedback into structure embedding
2. Merge similar    : structures with cosine sim > merge_threshold collapse
3. Specialize       : structures with low usage get pushed apart (placeholder)
4. Survival-conditioned: refinement strength ∝ survival score
5. Feedback-conditioned: optional external feedback signal
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_types import DPSLRefinementOutput
from .scm_config import TACSCMConfig


class DPSLRefinementLayer(nn.Module):
    """
    Refines a batch of structure embeddings in-place.

    Input
    -----
    structure_embeddings : (N, d_structure)
    survival_scores      : (N,) ∈ [0, 1]   — from NSFSurvivalScorer
    feedback             : (N, d_structure) or None — external feedback signal

    Output
    ------
    DPSLRefinementOutput with refined_embeddings and diagnostics
    """

    MERGE_THRESHOLD     = 0.90   # cosine sim above which structures merge
    SPECIALIZE_MARGIN   = 0.20   # target cosine sim for push-apart (specialization)

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        D = cfg.d_structure

        # Gated update network: [emb || feedback] → gate + delta
        self.gate_net = nn.Sequential(
            nn.Linear(D * 2, D, bias=False),
            nn.GELU(),
            nn.LayerNorm(D),
        )
        self.gate_scalar = nn.Linear(D, 1, bias=True)
        self.delta_net   = nn.Linear(D, D, bias=False)

        # Survival-conditioned scale: how much to refine based on survival
        self.surv_scale  = nn.Linear(1, 1, bias=True)

        # Merge projection: produce merged embedding from pair
        self.merge_proj = nn.Linear(D * 2, D, bias=False)

        # Loss weight
        self.w_refine = cfg.refinement_loss_weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        structure_embeddings: torch.Tensor,                    # (N, D)
        survival_scores:      torch.Tensor,                    # (N,)
        feedback:             Optional[torch.Tensor] = None,   # (N, D) or None
    ) -> DPSLRefinementOutput:

        N, D   = structure_embeddings.shape
        device = structure_embeddings.device

        if N == 0:
            z = structure_embeddings.new_zeros(())
            return DPSLRefinementOutput(
                refined_embeddings = structure_embeddings,
                gate_values        = structure_embeddings.new_zeros(N),
                merge_mask         = torch.zeros(N, dtype=torch.bool, device=device),
                loss_refinement    = z,
                loss_total         = z,
            )

        refined = structure_embeddings.clone()

        # ── 1. Gated feedback update ──────────────────────────────────────────
        if feedback is None:
            # Default feedback: mean of the batch as a proxy global signal
            feedback = structure_embeddings.mean(dim=0, keepdim=True).expand(N, D)

        fused      = torch.cat([refined, feedback], dim=-1)  # (N, 2D)
        gate_feat  = self.gate_net(fused)                     # (N, D)
        gate_vals  = torch.sigmoid(self.gate_scalar(gate_feat).squeeze(-1))  # (N,)
        delta      = torch.tanh(self.delta_net(gate_feat))   # (N, D)

        # Survival-conditioned refinement scale
        surv_scale = torch.sigmoid(
            self.surv_scale(survival_scores.unsqueeze(-1))   # (N, 1)
        ).squeeze(-1)                                         # (N,)
        effective_gate = gate_vals * surv_scale               # (N,)

        refined = refined + effective_gate.unsqueeze(-1) * delta  # (N, D)

        # ── 2. Merge similar structures ───────────────────────────────────────
        merge_mask = torch.zeros(N, dtype=torch.bool, device=device)

        if N >= 2:
            refined_n  = F.normalize(refined, dim=-1)         # (N, D)
            sim_matrix = refined_n @ refined_n.T              # (N, N)
            # Only consider upper triangle (unique pairs)
            for i in range(N - 1):
                if merge_mask[i]:
                    continue
                for j in range(i + 1, N):
                    if merge_mask[j]:
                        continue
                    if sim_matrix[i, j].item() >= self.MERGE_THRESHOLD:
                        # Merge j into i (weighted by survival)
                        si = survival_scores[i].item()
                        sj = survival_scores[j].item()
                        wi = si / (si + sj + 1e-8)
                        wj = sj / (si + sj + 1e-8)
                        # Use learned merge projection
                        merged = self.merge_proj(
                            torch.cat([refined[i], refined[j]], dim=-1).unsqueeze(0)
                        ).squeeze(0)
                        refined[i] = wi * merged + wj * refined[i]
                        merge_mask[j] = True

        # ── 3. Specialization: push low-sim pairs apart (placeholder) ─────────
        # Currently implemented as a light diversity loss — the actual
        # specialisation routing is done by the identity field router.
        if N >= 2:
            refined_n2 = F.normalize(refined[~merge_mask], dim=-1)
            if refined_n2.shape[0] >= 2:
                sim_sub = refined_n2 @ refined_n2.T           # (M, M)
                eye     = torch.eye(sim_sub.shape[0], device=device)
                off_sim = sim_sub.masked_fill(eye.bool(), 0)
                loss_div = F.relu(off_sim - self.SPECIALIZE_MARGIN).mean()
            else:
                loss_div = refined.new_zeros(())
        else:
            loss_div = refined.new_zeros(())

        # ── 4. Stability loss: refined should not drift too far ───────────────
        loss_drift = F.mse_loss(
            F.normalize(refined, dim=-1),
            F.normalize(structure_embeddings.detach(), dim=-1),
        )

        loss_refinement = loss_div + 0.1 * loss_drift
        loss_total      = self.w_refine * loss_refinement

        return DPSLRefinementOutput(
            refined_embeddings = refined,
            gate_values        = effective_gate,
            merge_mask         = merge_mask,
            loss_refinement    = loss_refinement,
            loss_total         = loss_total,
        )
