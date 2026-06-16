"""
TAC-SCM-REAL001: Structure Memory

Trainable, optionally persistent external memory for discovered structures.

Design principles
-----------------
Parameters learn how to use memory (read keys, write gates).
Memory slots store discovered reusable structures (not parameters).

The memory is a fixed-size bank:
  keys   : (n_slots, d_structure)  — what to match against
  values : (n_slots, d_structure)  — what to retrieve
  usage  : (n_slots,)             — how often each slot has been read
  age    : (n_slots,)             — steps since last write
  survival: (n_slots,)            — survival score (decayed over time)

Read: cosine similarity → top-k → weighted sum
Write: find least-used/lowest-survival slot; gated update
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_types import StructureMemoryOutput
from .scm_config import TACSCMConfig


class StructureMemory(nn.Module):
    """
    External structure memory bank.

    This is an nn.Module so its read/write projections are trainable,
    but the bank itself (keys, values) is a non-parameter buffer that
    stores discovered structures and can be saved/loaded separately.
    """

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        self.cfg        = cfg
        self.n_slots    = cfg.n_structure_slots
        self.d          = cfg.d_structure
        self.top_k      = min(8, cfg.n_structure_slots)
        self.write_rate = cfg.memory_write_rate
        self.decay      = cfg.survival_decay

        # ── Trainable projections ──────────────────────────────────────────────
        self.query_proj = nn.Linear(cfg.d_structure, cfg.d_structure, bias=False)
        self.value_proj = nn.Linear(cfg.d_structure, cfg.d_structure, bias=False)
        self.gate_proj  = nn.Linear(cfg.d_structure * 2, 1, bias=True)
        self.out_proj   = nn.Linear(cfg.d_structure, cfg.d_structure, bias=False)

        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.normal_(self.gate_proj.weight,  std=0.02)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.out_proj.weight,   std=0.02)

        # ── Memory bank (non-parameter buffers) ───────────────────────────────
        self.register_buffer("keys",     torch.zeros(self.n_slots, self.d))
        self.register_buffer("values",   torch.zeros(self.n_slots, self.d))
        self.register_buffer("usage",    torch.zeros(self.n_slots))
        self.register_buffer("age",      torch.zeros(self.n_slots))
        self.register_buffer("survival", torch.zeros(self.n_slots))
        self.register_buffer("filled",   torch.zeros(self.n_slots, dtype=torch.bool))

        # ── Step counter ──────────────────────────────────────────────────────
        self._step: int = 0

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, query: torch.Tensor) -> StructureMemoryOutput:
        """
        Retrieve top-k structures for a batch of queries.

        Args
        ----
        query : (B, d_structure)

        Returns
        -------
        StructureMemoryOutput
        """
        B, D = query.shape
        device = query.device

        q = F.normalize(self.query_proj(query), dim=-1)     # (B, D)

        n_filled = self.filled.sum().item()
        if n_filled == 0:
            # Memory empty: return zeros
            k = self.top_k
            return StructureMemoryOutput(
                retrieved_keys     = query.new_zeros(B, k, D),
                retrieved_values   = query.new_zeros(B, k, D),
                retrieval_scores   = query.new_zeros(B, k),
                retrieved_ids      = torch.zeros(B, k, dtype=torch.long, device=device),
                retrieved_survival = query.new_zeros(B, k),
                context_vector     = query.new_zeros(B, D),
            )

        # Cosine similarity against all filled slots
        filled_idx = self.filled.nonzero(as_tuple=False).squeeze(-1)  # (F,)
        mem_keys   = F.normalize(self.keys[filled_idx], dim=-1)       # (F, D)
        scores     = q @ mem_keys.T                                    # (B, F)

        k = min(self.top_k, filled_idx.shape[0])
        topk_scores, topk_local_idx = scores.topk(k, dim=-1)          # (B, k)
        topk_idx = filled_idx[topk_local_idx]                         # (B, k) global idx

        # Update usage counts
        with torch.no_grad():
            flat_idx = topk_idx.reshape(-1)
            ones     = torch.ones(flat_idx.shape[0], device=device)
            self.usage.scatter_add_(0, flat_idx, ones)

        retrieved_keys     = self.keys[topk_idx]                      # (B, k, D)
        retrieved_values   = self.values[topk_idx]                    # (B, k, D)
        retrieved_survival = self.survival[topk_idx]                  # (B, k)

        # Weighted combination using softmax over scores
        attn   = torch.softmax(topk_scores, dim=-1).unsqueeze(-1)     # (B, k, 1)
        ctx_raw = (attn * retrieved_values).sum(dim=1)                # (B, D)
        context = self.out_proj(ctx_raw)                              # (B, D)

        # Pad to full top_k if fewer slots are filled
        if k < self.top_k:
            pad = self.top_k - k
            retrieved_keys     = F.pad(retrieved_keys,     (0, 0, 0, pad))
            retrieved_values   = F.pad(retrieved_values,   (0, 0, 0, pad))
            topk_scores        = F.pad(topk_scores,        (0, pad))
            retrieved_survival = F.pad(retrieved_survival, (0, pad))
            zero_idx = torch.zeros(B, pad, dtype=torch.long, device=device)
            topk_idx = torch.cat([topk_idx, zero_idx], dim=-1)

        return StructureMemoryOutput(
            retrieved_keys     = retrieved_keys,
            retrieved_values   = retrieved_values,
            retrieval_scores   = topk_scores,
            retrieved_ids      = topk_idx,
            retrieved_survival = retrieved_survival,
            context_vector     = context,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def write(
        self,
        structure_candidates: torch.Tensor,  # (N, d_structure) — batch of structures
        survival_scores:      torch.Tensor,  # (N,) ∈ [0, 1]
        write_mask:           Optional[torch.Tensor] = None,  # (N,) bool
    ) -> torch.Tensor:
        """
        Write structure candidates to memory slots.

        Uses survival-weighted replacement: new structure replaces the slot
        with the lowest survival score (if new_survival > threshold).

        Returns written_ids (N,) — slot index written (-1 if skipped).
        """
        N     = structure_candidates.shape[0]
        device = structure_candidates.device
        written = torch.full((N,), -1, dtype=torch.long, device=device)

        for i in range(N):
            if write_mask is not None and not write_mask[i]:
                continue
            if torch.rand(1).item() > self.write_rate:
                continue

            s_new = survival_scores[i].item()
            emb   = structure_candidates[i]                    # (D,)

            # Check for near-duplicate (cosine sim > 0.95) → update in place
            if self.filled.any():
                filled_idx = self.filled.nonzero(as_tuple=False).squeeze(-1)
                sims = F.cosine_similarity(
                    emb.unsqueeze(0), self.keys[filled_idx], dim=-1
                )
                best_sim, best_local = sims.max(dim=0)
                if best_sim.item() > 0.95:
                    slot = filled_idx[best_local].item()
                    alpha = 0.1
                    self.keys[slot]     = (1 - alpha) * self.keys[slot]     + alpha * emb
                    self.values[slot]   = (1 - alpha) * self.values[slot]   + alpha * emb
                    self.survival[slot] = max(self.survival[slot].item(), s_new)
                    self.age[slot]      = 0
                    written[i]          = slot
                    continue

            if not self.filled.all():
                # Empty slot available
                empty_slots = (~self.filled).nonzero(as_tuple=False).squeeze(-1)
                slot        = empty_slots[0].item()
            else:
                # Replace lowest-survival slot
                slot = self.survival.argmin().item()
                if self.survival[slot].item() > s_new:
                    continue  # existing structure is stronger, skip

            val_proj = self.value_proj(emb.unsqueeze(0)).squeeze(0)
            self.keys[slot]     = emb
            self.values[slot]   = val_proj
            self.usage[slot]    = 0
            self.age[slot]      = 0
            self.survival[slot] = s_new
            self.filled[slot]   = True
            written[i]          = slot

        return written

    # ── Maintenance ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def step_decay(self):
        """Apply per-step survival decay. Call once per training step."""
        self.survival.mul_(self.decay)
        self.age.add_(1)
        self._step += 1

    @torch.no_grad()
    def prune(self, threshold: float = 0.01):
        """Remove slots with survival below threshold."""
        weak = self.filled & (self.survival < threshold)
        self.filled[weak]   = False
        self.keys[weak]     = 0
        self.values[weak]   = 0
        self.survival[weak] = 0
        self.usage[weak]    = 0
        self.age[weak]      = 0

    def reset(self):
        """Clear all memory slots."""
        self.keys.zero_()
        self.values.zero_()
        self.usage.zero_()
        self.age.zero_()
        self.survival.zero_()
        self.filled.fill_(False)
        self._step = 0

    def detach(self):
        """Detach all buffer tensors (for stateful inference)."""
        self.keys     = self.keys.detach()
        self.values   = self.values.detach()
        self.usage    = self.usage.detach()
        self.age      = self.age.detach()
        self.survival = self.survival.detach()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_memory_state(self) -> Dict[str, torch.Tensor]:
        return {
            "keys":     self.keys.cpu().clone(),
            "values":   self.values.cpu().clone(),
            "usage":    self.usage.cpu().clone(),
            "age":      self.age.cpu().clone(),
            "survival": self.survival.cpu().clone(),
            "filled":   self.filled.cpu().clone(),
        }

    def load_memory_state(self, state: Dict[str, torch.Tensor]):
        device = self.keys.device
        self.keys.copy_(state["keys"].to(device))
        self.values.copy_(state["values"].to(device))
        self.usage.copy_(state["usage"].to(device))
        self.age.copy_(state["age"].to(device))
        self.survival.copy_(state["survival"].to(device))
        self.filled.copy_(state["filled"].to(device))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, float]:
        n_filled = self.filled.sum().item()
        return {
            "n_filled":       n_filled,
            "fill_rate":      n_filled / self.n_slots,
            "mean_survival":  self.survival[self.filled].mean().item() if n_filled > 0 else 0.0,
            "mean_usage":     self.usage[self.filled].mean().item()    if n_filled > 0 else 0.0,
            "mean_age":       self.age[self.filled].mean().item()      if n_filled > 0 else 0.0,
        }
