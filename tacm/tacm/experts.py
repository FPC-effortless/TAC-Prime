"""
TAC-SM Shared Expert + Specialist Experts (DeepSeek-style MoE)

Architecture:
  Shared Expert     — always active; handles syntax, common language, generic reasoning
  Specialist Experts — top-k activated per token; handle domain-specific tasks

Specialists (by index):
  0  CodeRepair
  1  RepositoryNavigation
  2  AlgorithmTransfer
  3  Planning
  4  MemoryUpdate
  5  Verification
  6  Abstraction
  7  Retrieval
  8+ GeneralPurpose (remainder)

Tracking:
  - Expert entropy
  - Routing frequency
  - Expert reuse rate
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExpertConfig
from .router import StructureRoutingOutput

SPECIALIST_NAMES = [
    "CodeRepair",
    "RepositoryNavigation",
    "AlgorithmTransfer",
    "Planning",
    "MemoryUpdate",
    "Verification",
    "Abstraction",
    "Retrieval",
]


class FFNExpert(nn.Module):
    """Single SwiGLU expert FFN."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self._init()

    def _init(self):
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight,   std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class SharedExpert(nn.Module):
    """Always-active shared expert — handles syntax, common language, generic reasoning."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ffn  = FFNExpert(d_model, hidden_dim, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(x))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with:
      - One always-active shared expert
      - N specialist experts (top-k activated)
      - Expert utilisation tracking (no-grad)
    """

    def __init__(self, d_model: int, cfg: ExpertConfig):
        super().__init__()
        self.cfg       = cfg
        self.n_experts = cfg.n_experts
        self.top_k     = cfg.top_k

        self.shared = SharedExpert(d_model, cfg.shared_expert_dim, cfg.dropout)
        self.experts = nn.ModuleList([
            FFNExpert(d_model, cfg.expert_hidden_dim, cfg.dropout)
            for _ in range(cfg.n_experts)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        # Utilisation counters (no gradient, purely for monitoring)
        self.register_buffer("_routing_counts",     torch.zeros(cfg.n_experts))
        self.register_buffer("_routing_total",      torch.tensor(0.0))
        self.register_buffer("_expert_output_norms", torch.zeros(cfg.n_experts))

    def forward(
        self,
        x: torch.Tensor,
        routing: StructureRoutingOutput,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x       : (B, T, d_model)
        routing : StructureRoutingOutput  (topk_ids, topk_weights already computed)

        Returns:
          output : (B, T, d_model)
          stats  : dict of monitoring tensors
        """
        B, T, D = x.shape
        x_flat       = x.reshape(B * T, D)
        topk_ids     = routing.topk_ids.reshape(B * T, self.top_k)
        topk_weights = routing.topk_weights.reshape(B * T, self.top_k)

        # Shared expert — always runs
        shared_out = self.shared(x)   # (B, T, D)

        # Specialist experts — sparse
        expert_out = torch.zeros_like(x_flat)

        # Group tokens by expert for efficient batching
        for eid in range(self.n_experts):
            # Find tokens that route to this expert (any top-k slot)
            tok_mask    = (topk_ids == eid).any(dim=-1)   # (B*T,)
            if not tok_mask.any():
                continue

            tok_in      = x_flat[tok_mask]                # (n_tok, D)
            out         = self.experts[eid](tok_in)        # (n_tok, D)

            # Weight by routing score for this expert
            weights_for_eid = torch.zeros(B * T, device=x.device)
            for k in range(self.top_k):
                slot_match = topk_ids[:, k] == eid
                weights_for_eid[slot_match] = topk_weights[slot_match, k]

            expert_out[tok_mask] += out * weights_for_eid[tok_mask].unsqueeze(-1)

            # Track utilisation
            with torch.no_grad():
                self._routing_counts[eid] += tok_mask.float().sum()
                self._expert_output_norms[eid] = out.norm(dim=-1).mean()

        with torch.no_grad():
            self._routing_total += B * T

        expert_out_2d = expert_out.reshape(B, T, D)
        combined      = self.out_norm(shared_out + expert_out_2d)

        stats = {
            "expert_counts": self._routing_counts.clone(),
            "total_tokens":  self._routing_total.clone(),
            "output_norms":  self._expert_output_norms.clone(),
        }
        return combined, stats

    # ── Monitoring helpers ──────────────────────────────────────────────────

    def expert_utilisation(self) -> torch.Tensor:
        """Returns fraction of tokens routed to each expert."""
        if self._routing_total < 1:
            return torch.zeros(self.n_experts)
        return self._routing_counts / self._routing_total

    def expert_entropy(self) -> float:
        """Entropy of routing distribution — higher is more balanced."""
        p = self.expert_utilisation()
        p = p / (p.sum() + 1e-9)
        return -(p * (p + 1e-9).log()).sum().item()

    def reset_stats(self):
        self._routing_counts.zero_()
        self._routing_total.zero_()
        self._expert_output_norms.zero_()

    def utilisation_report(self) -> str:
        util  = self.expert_utilisation()
        lines = [f"  Expert entropy: {self.expert_entropy():.3f}"]
        for i, u in enumerate(util.tolist()):
            name = SPECIALIST_NAMES[i] if i < len(SPECIALIST_NAMES) else f"General-{i}"
            lines.append(f"  [{i:02d}] {name:<25} {u * 100:.1f}%")
        return "\n".join(lines)
