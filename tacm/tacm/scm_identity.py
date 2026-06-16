"""
TAC-SCM-REAL001: Structure Identity Field

StructureIdentityState: stateful carrier of active computational structures.
StructureIdentityFieldLayer: routes tokens to structure slots; reads from
and writes to StructureIdentityState.

Conceptual distinction from identity.py (TAC-Prime-ID001):
  identity.py IdentityState  — carries identity IDs + stability (symbolic)
  scm_identity.py            — carries actual structure embeddings (computational)

The identity field maintains n_identity_slots structure slots per batch.
Each forward pass:
  1. Compute token-to-slot routing (which token activates which slot)
  2. Read from active slots → structure context
  3. Update slots using incoming structure candidates (gated EMA)
  4. Return updated hidden + updated state + routing info
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_types import StructureIdentityState
from .scm_config import TACSCMConfig


class StructureIdentityFieldLayer(nn.Module):
    """
    Maintains and updates a bank of active structure slots.

    Forward
    -------
    hidden_states        : (B, T, d_model)
    structure_candidates : (B, n_cand, d_structure) or None
    state                : StructureIdentityState or None
    attention_mask       : (B, T) bool or None

    Returns
    -------
    updated_hidden_states : (B, T, d_model)
    updated_state         : StructureIdentityState
    route_logits          : (B, T, n_slots)
    route_weights         : (B, T, n_slots)
    structure_readout     : (B, T, d_model)
    aux_losses            : dict of str → tensor
    """

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        self.cfg     = cfg
        self.d_model = cfg.d_model
        self.d_str   = cfg.d_structure
        self.n_slots = cfg.n_identity_slots
        self.decay   = cfg.identity_state_decay
        self.scale   = cfg.identity_residual_scale
        self.dropout = nn.Dropout(cfg.identity_dropout)

        # ── Projections ────────────────────────────────────────────────────────

        # Token query: hidden → slot routing query
        self.token_query = nn.Linear(cfg.d_model, cfg.d_structure, bias=False)

        # Slot key: d_structure → d_structure (learned slot embeddings)
        self.slot_key = nn.Parameter(
            torch.randn(cfg.n_identity_slots, cfg.d_structure) * 0.02
        )

        # Read: structure → model space
        self.read_proj = nn.Linear(cfg.d_structure, cfg.d_model, bias=False)

        # Write gate: [slot_emb || candidate] → scalar gate
        self.write_gate = nn.Sequential(
            nn.Linear(cfg.d_structure * 2, cfg.d_structure, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_structure, 1, bias=True),
        )

        # Candidate aggregator: project n_cand candidates → n_slot updates
        # This is a small attention: slots attend over candidates
        self.cand_to_slot_attn = nn.MultiheadAttention(
            cfg.d_structure,
            num_heads=max(1, cfg.d_structure // 64),
            dropout=cfg.identity_dropout,
            batch_first=True,
        )

        # Output fusion: [hidden || structure_readout] → hidden
        self.fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False),
            nn.LayerNorm(cfg.d_model),
        )

        # Route entropy loss weight
        self.w_entropy = cfg.route_entropy_weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── State lifecycle ────────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device) -> StructureIdentityState:
        return StructureIdentityState.zeros(
            batch_size, self.n_slots, self.d_str, device=device
        )

    def reset_state(self, state: StructureIdentityState) -> StructureIdentityState:
        return state.reset()

    def detach_state(self, state: StructureIdentityState) -> StructureIdentityState:
        return state.detach()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states:        torch.Tensor,
        structure_candidates: Optional[torch.Tensor] = None,
        state:                Optional[StructureIdentityState] = None,
        attention_mask:       Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, StructureIdentityState, torch.Tensor,
               torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        B, T, Dm = hidden_states.shape
        device   = hidden_states.device

        # Initialise state if not provided
        if state is None:
            state = self.init_state(B, device)

        # ── 1. Token → slot routing ───────────────────────────────────────────
        token_q  = self.token_query(hidden_states)             # (B, T, d_str)
        slot_k   = self.slot_key.unsqueeze(0).expand(B, -1, -1)  # (B, n_slots, d_str)

        # Bias routing by existing slot weights (EMA of activation)
        slot_bias = state.slot_weights.unsqueeze(1)            # (B, 1, n_slots)
        route_logits = torch.bmm(
            F.normalize(token_q, dim=-1),
            F.normalize(slot_k,  dim=-1).transpose(1, 2),
        ) + slot_bias                                          # (B, T, n_slots)

        route_weights = torch.softmax(
            route_logits / math.sqrt(self.d_str), dim=-1
        )                                                      # (B, T, n_slots)

        # ── 2. Read from active slots ─────────────────────────────────────────
        # structure_readout = sum_s route_w[t,s] * slot_emb[s]
        # slot_embeddings: (B, n_slots, d_str)
        slot_emb    = state.slot_embeddings                    # (B, n_slots, d_str)
        # (B, T, n_slots) × (B, n_slots, d_str) → (B, T, d_str)
        struct_read = torch.bmm(route_weights, slot_emb)      # (B, T, d_str)
        struct_read_model = self.read_proj(struct_read)        # (B, T, d_model)
        struct_read_model = self.dropout(struct_read_model)

        # ── 3. Fusion into hidden states ──────────────────────────────────────
        fused = self.fusion(
            torch.cat([hidden_states, struct_read_model], dim=-1)
        )                                                      # (B, T, d_model)
        updated_hidden = hidden_states + self.scale * fused

        # ── 4. Update slots from incoming structure candidates ─────────────────
        if structure_candidates is not None:
            # slots attend over candidates → candidate context per slot
            slot_emb_norm = F.layer_norm(slot_emb, [self.d_str])
            cand_norm     = F.layer_norm(structure_candidates, [self.d_str])
            slot_ctx, _   = self.cand_to_slot_attn(
                slot_emb_norm, cand_norm, cand_norm
            )                                                  # (B, n_slots, d_str)

            # Write gate for each slot
            gate_in  = torch.cat([slot_emb, slot_ctx], dim=-1)  # (B, n_slots, 2d_str)
            gate     = torch.sigmoid(
                self.write_gate(gate_in).squeeze(-1)           # (B, n_slots)
            )

            # EMA update: slot = decay * slot + (1-decay) * gate * slot_ctx
            new_slot_emb = (
                self.decay * slot_emb
                + (1.0 - self.decay) * gate.unsqueeze(-1) * slot_ctx
            )

            # Update decision memory (EMA of structure context)
            new_decision_mem = (
                self.decay * state.decision_memory
                + (1.0 - self.decay) * slot_ctx
            )
        else:
            # Decay only
            new_slot_emb     = self.decay * slot_emb
            new_decision_mem = state.decision_memory

        # ── 5. Update routing history and stability ───────────────────────────
        route_mean      = route_weights.mean(dim=1)            # (B, n_slots)
        new_route_hist  = (
            self.decay * state.route_history + (1.0 - self.decay) * route_mean
        )

        # Stability: cosine similarity between old and new slot embeddings
        old_n = F.normalize(slot_emb,     dim=-1)
        new_n = F.normalize(new_slot_emb, dim=-1)
        stability = (old_n * new_n).sum(dim=-1).clamp(-1, 1)  # (B, n_slots)
        new_stab  = self.decay * state.stability_scores + (1.0 - self.decay) * stability

        # Active slot weights: how much each slot contributed to routing
        new_slot_weights = (
            self.decay * state.slot_weights + (1.0 - self.decay) * route_mean
        )

        # ── 6. Build updated state ────────────────────────────────────────────
        updated_state = StructureIdentityState(
            slot_embeddings  = new_slot_emb,
            slot_weights     = new_slot_weights,
            route_history    = new_route_hist,
            stability_scores = new_stab,
            decision_memory  = new_decision_mem,
            step_count       = state.step_count + 1,
        )

        # ── 7. Auxiliary losses ───────────────────────────────────────────────
        # Route entropy loss: encourage distributed routing
        ent = -(route_weights * (route_weights + 1e-9).log()).sum(dim=-1).mean()
        loss_entropy = -self.w_entropy * ent  # negative: maximise entropy

        # Slot stability loss: slots should not change too fast
        loss_stab = (1.0 - stability.clamp(-1, 1)).mean()

        aux_losses = {
            "identity_route_entropy": loss_entropy,
            "identity_slot_stability": 0.01 * loss_stab,
        }

        return (
            updated_hidden,
            updated_state,
            route_logits,
            route_weights,
            struct_read_model,
            aux_losses,
        )
