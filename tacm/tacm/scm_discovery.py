"""
TAC-SCM-REAL001: Structure Discovery Layer

JEPA-inspired structure discovery. An online encoder projects hidden states to a
latent structure space; a stop-gradient target encoder provides training signal;
a predictor MLP learns to predict target latents from online latents.

VICReg-style spread and covariance losses prevent representational collapse.
Structure candidates are extracted by clustering latents along the sequence.

References:
  - JEPA (LeCun 2022): predict latent representations, not pixels
  - VICReg (Bardes et al. 2022): variance-invariance-covariance regularization
  - I-JEPA (Assran et al. 2023): image JEPA with masked patch prediction
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_types import StructureDiscoveryOutput
from .scm_config import TACSCMConfig


# ── Small building blocks ──────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Structure Discovery Layer ──────────────────────────────────────────────────

class StructureDiscoveryLayer(nn.Module):
    """
    JEPA-based structure discovery engine.

    Architecture
    ------------
    online_encoder : d_model → d_structure   (participates in gradient)
    target_encoder : d_model → d_structure   (EMA of online_encoder, stop-gradient)
    predictor      : d_structure → d_structure

    Training objectives
    -------------------
    L_prediction  : MSE between predictor(online(x_t)) and stop_grad(target(x_{t+offset}))
    L_temporal    : MSE between online(x_t) and online(x_{t-1}) (slow variation)
    L_variance    : std of latents across batch should be > threshold (VICReg)
    L_covariance  : off-diagonal covariance should be near zero (VICReg)

    Structure candidates
    --------------------
    After encoding, cluster T latents into n_candidates using soft k-means (1 iter).
    The candidates are the cluster centers — compact representations of the
    distinct structures present in this token window.
    """

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model     = cfg.d_model
        self.d_struct    = cfg.d_structure
        self.n_cand      = cfg.n_structure_candidates
        self.future_off  = cfg.future_offset
        self.ema_decay   = cfg.target_ema_decay
        self.stop_grad   = cfg.stop_gradient_target

        # Online encoder: projects hidden → latent structure space
        self.online_encoder = _MLP(
            cfg.d_model, cfg.d_model, cfg.d_structure, cfg.structure_dropout
        )

        # Target encoder: EMA-updated copy (no grad)
        self.target_encoder = _MLP(
            cfg.d_model, cfg.d_model, cfg.d_structure, 0.0
        )
        # Initialise target = online
        self._copy_online_to_target()
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Predictor: latent → predicted latent (learns temporal structure)
        self.predictor = _MLP(
            cfg.d_structure, cfg.d_structure * 2, cfg.d_structure, cfg.structure_dropout
        )

        # Candidate extraction: learned cluster initialisation query
        self.cluster_queries = nn.Parameter(
            torch.randn(cfg.n_structure_candidates, cfg.d_structure) * 0.02
        )

        # Loss weights from config
        self.w_pred   = cfg.jepa_prediction_weight
        self.w_temp   = cfg.temporal_consistency_weight
        self.w_var    = cfg.spread_loss_weight
        self.w_cov    = cfg.covariance_loss_weight

        self._step = 0

    # ── EMA update ────────────────────────────────────────────────────────────

    def _copy_online_to_target(self):
        for p_on, p_tgt in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.copy_(p_on.data)

    @torch.no_grad()
    def update_target_ema(self):
        """Call once per training step (after optimizer.step())."""
        decay = self.ema_decay
        for p_on, p_tgt in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(decay).add_(p_on.data, alpha=1.0 - decay)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, T, d_model)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) bool/float
    ) -> StructureDiscoveryOutput:

        B, T, D = hidden_states.shape
        device  = hidden_states.device

        # ── 1. Online encoding ────────────────────────────────────────────────
        latent = self.online_encoder(hidden_states)           # (B, T, d_struct)

        # ── 2. Target encoding (stop-gradient) ───────────────────────────────
        if self.stop_grad:
            with torch.no_grad():
                target = self.target_encoder(hidden_states)  # (B, T, d_struct)
        else:
            target = self.target_encoder(hidden_states)

        # ── 3. Future prediction ──────────────────────────────────────────────
        # Predict target[t+offset] from predictor(latent[t])
        offset       = min(self.future_off, T - 1)
        pred_input   = latent[:, :T - offset, :]              # (B, T-off, d_s)
        pred_out     = self.predictor(pred_input)              # (B, T-off, d_s)
        target_fut   = target[:, offset:, :].detach()         # (B, T-off, d_s) stop-grad

        loss_pred = F.mse_loss(
            F.normalize(pred_out, dim=-1),
            F.normalize(target_fut, dim=-1),
        )

        # Pad predicted / target to full T for output consistency
        pad_pred   = F.pad(pred_out, (0, 0, 0, offset))       # (B, T, d_s)
        pad_target = F.pad(target_fut, (0, 0, 0, offset))     # (B, T, d_s) (approx)

        # ── 4. Temporal consistency ───────────────────────────────────────────
        if T > 1:
            loss_temporal = F.mse_loss(latent[:, 1:, :], latent[:, :-1, :].detach())
        else:
            loss_temporal = latent.new_zeros(())

        # ── 5. VICReg spread loss ─────────────────────────────────────────────
        # Flatten batch × time for computing statistics
        z_flat = latent.reshape(B * T, self.d_struct)          # (BT, d_struct)
        z_std  = z_flat.std(dim=0)                             # (d_struct,)
        loss_var = F.relu(1.0 - z_std).mean()                  # push std > 1

        # ── 6. VICReg covariance loss ─────────────────────────────────────────
        z_c   = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov   = (z_c.T @ z_c) / (B * T - 1)                   # (d_s, d_s)
        off   = cov.pow(2)
        eye   = torch.eye(self.d_struct, device=device, dtype=cov.dtype)
        loss_cov = (off * (1 - eye)).sum() / self.d_struct

        # ── 7. Collapse metric (diagnostic) ───────────────────────────────────
        collapse_metric = z_std.mean().detach()

        # ── 8. Extract structure candidates via soft k-means (1 iteration) ───
        candidates = self._extract_candidates(latent)          # (B, n_cand, d_s)

        # ── 9. Total loss ─────────────────────────────────────────────────────
        loss_total = (
            self.w_pred * loss_pred
            + self.w_temp * loss_temporal
            + self.w_var  * loss_var
            + self.w_cov  * loss_cov
        )

        self._step += 1

        return StructureDiscoveryOutput(
            latent_state           = latent,
            predicted_latent_state = pad_pred,
            target_latent_state    = pad_target,
            structure_candidates   = candidates,
            loss_prediction        = loss_pred,
            loss_variance          = loss_var,
            loss_covariance        = loss_cov,
            loss_total             = loss_total,
            collapse_metric        = collapse_metric,
        )

    # ── Candidate extraction ───────────────────────────────────────────────────

    def _extract_candidates(
        self,
        latent: torch.Tensor,   # (B, T, d_struct)
    ) -> torch.Tensor:
        """
        Soft k-means (1 iteration) over token positions.
        Cluster queries are learnable initialisation.
        Returns (B, n_cand, d_struct) cluster centers.
        """
        B, T, D = latent.shape
        queries = self.cluster_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n, D)

        # Attention scores: (B, n_cand, T)
        q_n = F.normalize(queries, dim=-1)                   # (B, n, D)
        l_n = F.normalize(latent,  dim=-1)                   # (B, T, D)
        scores = torch.bmm(q_n, l_n.transpose(1, 2))        # (B, n, T)
        weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # (B, n, T)

        # Weighted sum over token positions → candidate centers
        candidates = torch.bmm(weights, latent)              # (B, n, D)
        return candidates
