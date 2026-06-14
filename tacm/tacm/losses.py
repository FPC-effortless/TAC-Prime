"""
TAC-SM Total Loss

L_total =
    w_next_token       * L_next_token
  + w_multi_token      * L_multi_token
  + w_volume           * L_volume
  + w_family_route     * L_family_route
  + w_expert_route     * L_expert_route
  + w_structure_memory * L_structure_memory
  + w_transfer         * L_transfer
  + w_survival         * L_survival
  + w_verifier         * L_verifier

All weights are configurable via TrainingConfig.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainingConfig


class StructureMemoryLoss(nn.Module):
    """
    Encourages the model to produce embeddings close to successfully retrieved
    memory structures and far from failed ones.

    Uses a triplet-style loss:
      anchor   = current task embedding
      positive = retrieved successful structure embedding
      negative = retrieved failed / low-score structure embedding
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor:   torch.Tensor,   # (B, D)
        positive: torch.Tensor,   # (B, D) — successful structure
        negative: torch.Tensor,   # (B, D) — failed / low-score structure
    ) -> torch.Tensor:
        a = F.normalize(anchor,   dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negative, dim=-1)

        d_pos = (a - p).pow(2).sum(-1)   # (B,)
        d_neg = (a - n).pow(2).sum(-1)   # (B,)

        return F.relu(d_pos - d_neg + self.margin).mean()


class TransferLoss(nn.Module):
    """
    Encourages embeddings of similar tasks (same family, different repositories)
    to be close in embedding space — enabling structure transfer.

    source_emb : embedding of source task (B, D)
    target_emb : embedding of target task (B, D)
    transfer_labels : 1 if same family / transferable, 0 if not (B,)
    """

    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.margin      = margin
        self.temperature = temperature

    def forward(
        self,
        source_emb:      torch.Tensor,
        target_emb:      torch.Tensor,
        transfer_labels: torch.Tensor,   # (B,) float {0, 1}
    ) -> torch.Tensor:
        s = F.normalize(source_emb, dim=-1)
        t = F.normalize(target_emb, dim=-1)
        sims = (s * t).sum(-1) / self.temperature   # (B,)

        # Pull together when transferable, push apart when not
        pull = transfer_labels       * (1.0 - sims)
        push = (1 - transfer_labels) * F.relu(sims - (1.0 - self.margin))
        return (pull + push).mean()


class SurvivalLoss(nn.Module):
    """
    Encourages the model to produce embeddings that remain stable
    under small perturbations (proxy for survival).

    Consistency under noise: perturbed embedding should produce
    same verifier success prediction.
    """

    def __init__(self, noise_std: float = 0.05):
        super().__init__()
        self.noise_std = noise_std

    def forward(
        self,
        embedding:    torch.Tensor,     # (B, D)
        verifier_proj: nn.Module,       # callable: (B, D) → (B,) success logit
    ) -> torch.Tensor:
        with torch.no_grad():
            base_pred = verifier_proj(embedding)

        noise     = torch.randn_like(embedding) * self.noise_std
        noisy_pred = verifier_proj(embedding + noise)

        return (base_pred.detach() - noisy_pred).pow(2).mean()


class TotalLoss(nn.Module):
    """
    Aggregates all TAC-SM losses with configurable weights.
    Individual sub-losses are computed externally and passed as a dict.
    This module applies weights and sums.
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        weights = {
            "next_token":       cfg.w_next_token,
            "multi_token":      cfg.w_multi_token,
            "volume":           cfg.w_volume,
            "family_route":     cfg.w_family_route,
            "expert_route":     cfg.w_expert_route,
            "structure_memory": cfg.w_structure_memory,
            "transfer":         cfg.w_transfer,
            "survival":         cfg.w_survival,
            "verifier":         cfg.w_verifier,
        }
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for name, w in weights.items():
            if name in loss_dict and w > 0:
                total = total + w * loss_dict[name]
        return total

    def breakdown(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Returns human-readable loss breakdown for logging."""
        return {k: v.item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
