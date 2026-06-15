"""
TAC-SM Two-Level Structure Router

Routes tokens through:
  ConceptVolume → Structure Family → Specialist Expert

Never routes directly to experts — always goes through the family level first.

Routing outputs:
  { family_id, expert_id, routing_confidence }

Losses:
  - family routing accuracy (cross-entropy)
  - expert routing accuracy
  - entropy regularisation
  - load balancing
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RouterConfig
from .concept_volume import ConceptVolumeOutput


# Structure family names — used for logging and specialisation tracking
FAMILY_NAMES = [
    "CodeRepair",
    "MathProcedure",
    "Verification",
    "Planning",
    "Retrieval",
    "MemoryUpdate",
    "Abstraction",
    "Transfer",
]


class FamilyRouter(nn.Module):
    """
    Level 1: ConceptVolume → Structure Family.
    Soft routing during training (straight-through), hard during inference.
    """

    def __init__(self, volume_dim: int, n_families: int, temperature: float = 1.0):
        super().__init__()
        self.n_families  = n_families
        self.temperature = temperature
        self.router = nn.Sequential(
            nn.Linear(volume_dim, volume_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(volume_dim * 2, n_families, bias=True),
        )
        nn.init.normal_(self.router[0].weight, std=0.02)
        nn.init.normal_(self.router[2].weight, std=0.02)
        nn.init.zeros_(self.router[2].bias)

    def forward(self, concept_center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        concept_center : (B, T, volume_dim)
        Returns:
          logits  : (B, T, n_families)
          family_ids: (B, T) long
        """
        logits     = self.router(concept_center) / self.temperature
        family_ids = logits.argmax(-1)
        return logits, family_ids


class ExpertRouter(nn.Module):
    """
    Level 2: Family embedding → Specialist Expert.
    Top-k routing with load-balancing auxiliary loss.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_experts  = n_experts
        self.top_k      = top_k
        self.temperature = temperature

        self.router = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, n_experts, bias=True),
        )
        # Per-expert bias (following Switch/DeepSeek load balancing)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts))
        nn.init.normal_(self.router[0].weight, std=0.02)
        nn.init.normal_(self.router[2].weight, std=0.02)
        nn.init.zeros_(self.router[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> "ExpertRoutingOutput":
        """
        x          : (B, T, d_model)
        family_ids : (B, T)

        Returns ExpertRoutingOutput
        """
        B, T, D = x.shape
        logits  = self.router(x) / self.temperature   # (B, T, n_experts)
        logits  = logits + self.expert_bias            # bias per expert

        # Top-k selection
        topk_logits, topk_ids = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        # Routing confidence: max probability
        all_probs  = F.softmax(logits, dim=-1)
        confidence = all_probs.max(dim=-1).values

        return ExpertRoutingOutput(
            logits=logits,
            all_probs=all_probs,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            confidence=confidence,
            family_ids=family_ids,
        )


class ExpertRoutingOutput:
    __slots__ = [
        "logits", "all_probs", "topk_ids", "topk_weights", "confidence", "family_ids"
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StructureRouter(nn.Module):
    """
    Full two-level router.
    FamilyRouter → ExpertRouter, with combined routing output.

    TAC-Prime-ID001: optionally biases routing with a projected identity_context
    vector (identity_router_bias_scale * projected_identity_context added to the
    router hidden input before expert-logit computation).
    """

    def __init__(
        self,
        d_model: int,
        volume_dim: int,
        cfg: RouterConfig,
    ):
        super().__init__()
        self.cfg           = cfg
        self.family_router = FamilyRouter(volume_dim, cfg.n_families, cfg.temperature)
        self.expert_router = ExpertRouter(d_model, cfg.n_experts, cfg.top_k, cfg.temperature)

        # Family embedding lookup — conditions expert routing on family
        self.family_embed = nn.Embedding(cfg.n_families, d_model)
        nn.init.normal_(self.family_embed.weight, std=0.02)

        # Identity bias projection (TAC-Prime-ID001) — projects identity_context
        # into the same space as the router hidden input
        self.identity_bias_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.identity_bias_proj.weight)  # zero-init → no effect initially

    def forward(
        self,
        hidden: torch.Tensor,
        concept_center: torch.Tensor,
        identity_context: Optional[torch.Tensor] = None,
        identity_router_bias_scale: float = 0.25,
    ) -> "StructureRoutingOutput":
        """
        hidden        : (B, T, d_model)
        concept_center: (B, T, volume_dim)
        identity_context : (B, T, d_model) or None — from IdentityFieldLayer.
            When provided, a small bias is added to the router hidden input.
            Does not hard-code family labels.

        Returns StructureRoutingOutput
        """
        # Level 1: family
        family_logits, family_ids = self.family_router(concept_center)

        # Condition hidden on family for expert routing
        fam_emb   = self.family_embed(family_ids)          # (B, T, d_model)
        x_cond    = hidden + fam_emb

        # TAC-Prime-ID001: add identity bias to router input (zero by default)
        if identity_context is not None and identity_router_bias_scale > 0.0:
            id_bias = self.identity_bias_proj(identity_context)  # (B, T, d_model)
            x_cond  = x_cond + identity_router_bias_scale * id_bias

        # Level 2: expert
        expert_out = self.expert_router(x_cond, family_ids)

        return StructureRoutingOutput(
            family_logits  = family_logits,
            family_ids     = family_ids,
            expert_logits  = expert_out.logits,
            expert_probs   = expert_out.all_probs,
            topk_ids       = expert_out.topk_ids,
            topk_weights   = expert_out.topk_weights,
            routing_confidence = expert_out.confidence,
        )


class StructureRoutingOutput:
    __slots__ = [
        "family_logits", "family_ids",
        "expert_logits", "expert_probs",
        "topk_ids", "topk_weights",
        "routing_confidence",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Routing Losses ────────────────────────────────────────────────────────────

class RouterLoss(nn.Module):
    """
    Combined routing losses:
      1. Family routing accuracy  (supervised if labels available)
      2. Expert routing accuracy  (supervised if labels available)
      3. Entropy regularisation   (maximise routing entropy → exploration)
      4. Load balancing           (uniform expert usage)
    """

    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        routing: StructureRoutingOutput,
        family_labels: Optional[torch.Tensor] = None,
        expert_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        losses = {}
        device = routing.family_logits.device

        # 1. Family accuracy
        if family_labels is not None:
            B, T = family_labels.shape
            fl = routing.family_logits.reshape(B * T, -1)
            ll = family_labels.reshape(B * T)
            valid = ll >= 0
            if valid.any():
                losses["family"] = F.cross_entropy(fl[valid], ll[valid])
            else:
                losses["family"] = torch.tensor(0.0, device=device)
        else:
            losses["family"] = torch.tensor(0.0, device=device)

        # 2. Expert accuracy
        if expert_labels is not None:
            B, T = expert_labels.shape
            el = routing.expert_logits.reshape(B * T, -1)
            ll = expert_labels.reshape(B * T)
            valid = ll >= 0
            if valid.any():
                losses["expert"] = F.cross_entropy(el[valid], ll[valid])
            else:
                losses["expert"] = torch.tensor(0.0, device=device)
        else:
            losses["expert"] = torch.tensor(0.0, device=device)

        # 3. Entropy regularisation — encourage uniform family routing
        fam_probs = F.softmax(routing.family_logits, dim=-1)         # (B, T, nF)
        fam_entropy = -(fam_probs * (fam_probs + 1e-9).log()).sum(-1) # (B, T)
        losses["entropy"] = -fam_entropy.mean()   # negative → maximise entropy

        # 4. Load balancing — expert usage should be uniform
        # Following Switch Transformer: f_i * p_i
        B, T, nE = routing.expert_probs.shape
        # Fraction of tokens routed to expert i
        expert_counts = torch.zeros(nE, device=device)
        topk_flat = routing.topk_ids.reshape(-1, self.cfg.top_k)
        for k in range(self.cfg.top_k):
            expert_counts.scatter_add_(
                0,
                topk_flat[:, k],
                torch.ones(topk_flat.shape[0], device=device),
            )
        f_i = expert_counts / (B * T * self.cfg.top_k)     # fraction
        p_i = routing.expert_probs.reshape(B * T, nE).mean(0)
        losses["load_balance"] = nE * (f_i * p_i).sum()

        losses["total"] = (
            losses["family"]
            + losses["expert"]
            + self.cfg.lambda_entropy      * losses["entropy"]
            + self.cfg.lambda_load_balance * losses["load_balance"]
        )
        return losses
