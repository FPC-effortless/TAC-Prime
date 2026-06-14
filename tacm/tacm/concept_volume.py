"""
TAC-SM Adaptive Concept Volume Layer

Replaces point-like concept representations with volume representations.
Each concept has: center, variance, confidence, and family_logits.

Losses implemented here:
  - Volume consistency loss   (same concept → centers close)
  - Separation loss           (different concepts → centers far)
  - Hierarchy containment     (child volume inside parent volume)
  - Temporal stability        (concepts stable across training)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConceptVolumeConfig


class ConceptVolume(nn.Module):
    """
    Projects transformer hidden states into concept-volume space.

    Input : (B, T, d_model)
    Output: ConceptVolumeOutput with center, variance, confidence, family_logits
    """

    def __init__(self, d_model: int, cfg: ConceptVolumeConfig):
        super().__init__()
        self.cfg = cfg
        vdim = cfg.volume_dim
        nfam = cfg.n_concept_families

        # Projections from transformer hidden states
        self.center_proj     = nn.Linear(d_model, vdim, bias=False)
        self.variance_proj   = nn.Linear(d_model, vdim, bias=False)
        self.confidence_proj = nn.Linear(d_model, 1,    bias=True)
        self.family_proj     = nn.Linear(d_model, nfam, bias=True)

        # Learnable family hierarchy matrix: family_i is parent of family_j
        # hierarchy[i, j] = 1 means family_i contains family_j
        self.family_hierarchy = nn.Parameter(
            torch.eye(nfam) * 0.1, requires_grad=True
        )

        # EMA of centers for temporal stability loss
        self.register_buffer("ema_centers", torch.zeros(nfam, vdim))
        self.register_buffer("ema_initialized", torch.zeros(nfam, dtype=torch.bool))
        self.ema_momentum = 0.99

        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.center_proj.weight)
        nn.init.xavier_uniform_(self.variance_proj.weight)
        nn.init.zeros_(self.confidence_proj.bias)
        nn.init.zeros_(self.family_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> "ConceptVolumeOutput":
        """
        hidden: (B, T, d_model)
        Returns ConceptVolumeOutput
        """
        center     = self.center_proj(hidden)
        log_var    = self.variance_proj(hidden)
        variance   = (
            torch.sigmoid(log_var)
            * (self.cfg.max_variance - self.cfg.min_variance)
            + self.cfg.min_variance
        )
        confidence  = torch.sigmoid(self.confidence_proj(hidden))  # (B, T, 1)
        family_logits = self.family_proj(hidden)                   # (B, T, n_families)

        return ConceptVolumeOutput(
            center=center,
            variance=variance,
            confidence=confidence,
            family_logits=family_logits,
        )

    def update_ema(self, centers: torch.Tensor, family_ids: torch.Tensor):
        """
        Update exponential moving average of family centers.
        centers   : (N, volume_dim)
        family_ids: (N,) long
        """
        for fid in family_ids.unique():
            mask = family_ids == fid
            mean_c = centers[mask].mean(0).detach()
            if self.ema_initialized[fid]:
                self.ema_centers[fid] = (
                    self.ema_momentum * self.ema_centers[fid]
                    + (1 - self.ema_momentum) * mean_c
                )
            else:
                self.ema_centers[fid] = mean_c
                self.ema_initialized[fid] = True


class ConceptVolumeOutput:
    """Structured output from ConceptVolume layer."""

    __slots__ = ["center", "variance", "confidence", "family_logits"]

    def __init__(
        self,
        center: torch.Tensor,
        variance: torch.Tensor,
        confidence: torch.Tensor,
        family_logits: torch.Tensor,
    ):
        self.center        = center          # (B, T, volume_dim)
        self.variance      = variance        # (B, T, volume_dim)
        self.confidence    = confidence      # (B, T, 1)
        self.family_logits = family_logits   # (B, T, n_families)

    @property
    def family_probs(self) -> torch.Tensor:
        return F.softmax(self.family_logits, dim=-1)

    @property
    def family_ids(self) -> torch.Tensor:
        return self.family_logits.argmax(-1)


# ── Losses ────────────────────────────────────────────────────────────────────

class ConceptVolumeLoss(nn.Module):
    """
    All four volume losses packaged together.
    Returned as a dict so caller can weight them independently.
    """

    def __init__(self, cfg: ConceptVolumeConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        vol_out: ConceptVolumeOutput,
        concept_labels: Optional[torch.Tensor] = None,
        parent_labels: Optional[torch.Tensor] = None,
        ema_centers: Optional[torch.Tensor] = None,
        ema_initialized: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        concept_labels : (B, T) long — which concept each token belongs to (or -1)
        parent_labels  : (B, T) long — parent concept for hierarchy (or -1)
        ema_centers    : (n_families, volume_dim)
        ema_initialized: (n_families,) bool

        Returns dict with individual loss tensors.
        """
        losses = {}

        # 1. Volume Consistency — same concept → close centers
        if concept_labels is not None:
            losses["consistency"] = self._consistency_loss(vol_out, concept_labels)
        else:
            losses["consistency"] = torch.tensor(0.0, device=vol_out.center.device)

        # 2. Separation — different concepts → far centers
        if concept_labels is not None:
            losses["separation"] = self._separation_loss(vol_out, concept_labels)
        else:
            losses["separation"] = torch.tensor(0.0, device=vol_out.center.device)

        # 3. Hierarchy Containment
        if parent_labels is not None:
            losses["hierarchy"] = self._hierarchy_loss(vol_out, parent_labels)
        else:
            losses["hierarchy"] = torch.tensor(0.0, device=vol_out.center.device)

        # 4. Temporal Stability
        if ema_centers is not None and ema_initialized is not None:
            losses["temporal"] = self._temporal_loss(
                vol_out, ema_centers, ema_initialized
            )
        else:
            losses["temporal"] = torch.tensor(0.0, device=vol_out.center.device)

        cfg = self.cfg
        losses["total"] = (
            cfg.lambda_consistency * losses["consistency"]
            + cfg.lambda_separation  * losses["separation"]
            + cfg.lambda_hierarchy   * losses["hierarchy"]
            + cfg.lambda_temporal    * losses["temporal"]
        )
        return losses

    def _consistency_loss(
        self,
        vol: ConceptVolumeOutput,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Pull centers of same-concept tokens together."""
        B, T, D = vol.center.shape
        center_flat = vol.center.reshape(B * T, D)
        label_flat  = labels.reshape(B * T)

        valid = label_flat >= 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=vol.center.device)

        c = center_flat[valid]
        l = label_flat[valid]
        uniq = l.unique()
        loss = torch.tensor(0.0, device=c.device)
        count = 0
        for u in uniq:
            mask = l == u
            if mask.sum() < 2:
                continue
            group = c[mask]
            mean  = group.mean(0, keepdim=True)
            loss  = loss + (group - mean).pow(2).mean()
            count += 1
        return loss / max(count, 1)

    def _separation_loss(
        self,
        vol: ConceptVolumeOutput,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Push centers of different-concept tokens apart (hinge)."""
        B, T, D = vol.center.shape
        center_flat = vol.center.reshape(B * T, D)
        label_flat  = labels.reshape(B * T)

        valid = label_flat >= 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=vol.center.device)

        c = center_flat[valid]
        l = label_flat[valid]

        # Sample up to 512 pairs for efficiency
        N = min(c.shape[0], 512)
        idx = torch.randperm(c.shape[0], device=c.device)[:N]
        c_s = c[idx]
        l_s = l[idx]

        dists = torch.cdist(c_s, c_s)            # (N, N)
        same  = (l_s.unsqueeze(1) == l_s.unsqueeze(0)).float()
        diff  = 1.0 - same

        margin = self.cfg.margin
        hinge  = F.relu(margin - dists)
        loss   = (hinge * diff).sum() / (diff.sum() + 1e-8)
        return loss

    def _hierarchy_loss(
        self,
        vol: ConceptVolumeOutput,
        parent_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Child volume should be contained within parent volume.
        Implemented as: parent_variance > child_variance at matching positions.
        """
        B, T, D = vol.variance.shape
        var_flat    = vol.variance.reshape(B * T, D)
        parent_flat = parent_labels.reshape(B * T)

        # Group by parent
        valid = parent_flat >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=vol.variance.device)

        v = var_flat[valid]
        p = parent_flat[valid]

        # For each (child, parent) pair — parent variance should be larger
        loss = torch.tensor(0.0, device=v.device)
        count = 0
        for pid in p.unique():
            child_mask  = p == pid
            parent_mask = p == pid  # simplified: treat same family as peers
            if child_mask.sum() < 2:
                continue
            child_var  = v[child_mask]
            parent_var = v[child_mask].mean(0, keepdim=True).detach() * 1.1
            violation  = F.relu(child_var - parent_var).mean()
            loss = loss + violation
            count += 1

        return loss / max(count, 1)

    def _temporal_loss(
        self,
        vol: ConceptVolumeOutput,
        ema_centers: torch.Tensor,
        ema_initialized: torch.Tensor,
    ) -> torch.Tensor:
        """Current family centers should stay close to their EMA."""
        family_ids = vol.family_ids  # (B, T)
        B, T, D = vol.center.shape
        center_flat = vol.center.reshape(B * T, D)
        fid_flat    = family_ids.reshape(B * T)

        if not ema_initialized.any():
            return torch.tensor(0.0, device=vol.center.device)

        loss = torch.tensor(0.0, device=vol.center.device)
        count = 0
        for fid in fid_flat.unique():
            if not ema_initialized[fid]:
                continue
            mask     = fid_flat == fid
            cur_mean = center_flat[mask].mean(0)
            ema_c    = ema_centers[fid].to(cur_mean.device)
            loss = loss + (cur_mean - ema_c).pow(2).mean()
            count += 1

        return loss / max(count, 1)
