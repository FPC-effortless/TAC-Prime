"""
TAC-Prime IdentityField Module (TAC-Prime-ID001)

Carries persistent computational identities alongside ConceptVolume and
StructureMemory — without replacing either.

Components:
  IdentityState        — persistent cross-call state (stability, memory, history)
  IdentityFieldOutput  — single-call output (context, aux losses, coherence)
  IdentityFieldLayer   — learnable nn.Module; forward updates IdentityState

Architecture role:
  backbone → IdentityField → ConceptVolume → identity-biased StructureRouter
           → MoE → identity-biased StructureMemory
           → identity-biased ProceduralMemory → SurvivalField → Verifier
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class IdentityState:
    """
    Persistent state carried across calls for a batch.

    Attributes
    ----------
    stability       : (B, n_identities)   how stable each identity is
    identity_memory : (B, n_identities, d_model)  accumulated identity context
    route_history   : (B, n_identities)   optional exponential-moving-average
                      of how often each identity was active
    active_identity : (B,)  int64 — index of the currently dominant identity
    """
    stability:       "torch.Tensor"            # (B, n_identities)
    identity_memory: "torch.Tensor"            # (B, n_identities, d_model)
    route_history:   Optional["torch.Tensor"]  # (B, n_identities) or None
    active_identity: Optional["torch.Tensor"]  # (B,) int64 or None

    def detach(self) -> "IdentityState":
        """Return a copy with all tensors detached (for stateful inference)."""
        return IdentityState(
            stability       = self.stability.detach(),
            identity_memory = self.identity_memory.detach(),
            route_history   = self.route_history.detach() if self.route_history is not None else None,
            active_identity = self.active_identity.detach() if self.active_identity is not None else None,
        )

    @staticmethod
    def zeros(batch_size: int, n_identities: int, d_model: int,
              device=None) -> "IdentityState":
        """Create a blank initial state."""
        kw = {} if device is None else {"device": device}
        return IdentityState(
            stability       = torch.ones(batch_size, n_identities, **kw),
            identity_memory = torch.zeros(batch_size, n_identities, d_model, **kw),
            route_history   = torch.zeros(batch_size, n_identities, **kw),
            active_identity = torch.zeros(batch_size, dtype=torch.long, **kw),
        )


@dataclass
class IdentityFieldOutput:
    """
    Single-forward-pass output of IdentityFieldLayer.

    Attributes
    ----------
    identity_context    : (B, T, d_model)  residual bias to add to hidden states
    identity_state      : updated IdentityState
    active_identity     : (B,) int64 — dominant identity index
    identity_coherence  : (B, n_identities, n_identities)  pairwise coherence
    aux_losses          : dict of scalar tensors:
                            identity_reuse, identity_energy,
                            identity_coherence_loss, identity_separation
    """
    identity_context:   "torch.Tensor"
    identity_state:     IdentityState
    active_identity:    "torch.Tensor"
    identity_coherence: "torch.Tensor"
    aux_losses:         Dict[str, "torch.Tensor"]


# ── IdentityFieldLayer ────────────────────────────────────────────────────────

class IdentityFieldLayer(nn.Module):
    """
    Carries n_identities persistent computational identities.

    Forward pass:
      1. token-to-identity affinity via learned embeddings
      2. energy-budgeted identity routing (soft top-k)
      3. identity_context = weighted sum of identity memories
      4. update identity state (decay + accumulate)
      5. return auxiliary losses for joint training
    """

    def __init__(
        self,
        d_model:              int,
        n_identities:         int   = 16,
        identity_energy_budget: float = 4.0,
        identity_state_decay: float  = 0.8,
        identity_loss_weight: float  = 0.05,
    ):
        super().__init__()
        self.d_model               = d_model
        self.n_identities          = n_identities
        self.identity_energy_budget = identity_energy_budget
        self.identity_state_decay  = identity_state_decay
        self.identity_loss_weight  = identity_loss_weight

        # Learnable identity embeddings  (n_identities, d_model)
        self.identity_embeddings = nn.Embedding(n_identities, d_model)
        nn.init.normal_(self.identity_embeddings.weight, std=0.02)

        # Project hidden → identity query space
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.query_proj.weight, std=0.02)

        # Project identity context → residual space (optional gating)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.context_proj.weight, std=0.02)

        # Project identity embedding for coherence matrix
        self.coherence_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.coherence_proj.weight, std=0.02)

        # Layer norm for identity memory update
        self.identity_norm = nn.LayerNorm(d_model)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden:   "torch.Tensor",                  # (B, T, d_model)
        state:    Optional[IdentityState] = None,  # previous state or None
    ) -> IdentityFieldOutput:
        """
        hidden : (B, T, d_model)
        state  : previous IdentityState (or None → fresh zeros)

        Returns IdentityFieldOutput.
        """
        B, T, D = hidden.shape
        device  = hidden.device

        # Initialise state if needed
        if state is None:
            state = IdentityState.zeros(B, self.n_identities, D, device=device)

        # ── 1. Token → identity affinity ──────────────────────────────────
        q = self.query_proj(hidden)                               # (B, T, D)
        id_embs = self.identity_embeddings.weight                 # (n_id, D)
        id_embs_n = F.normalize(id_embs, dim=-1)
        q_n = F.normalize(q, dim=-1)
        # logits: (B, T, n_identities)
        logits = q_n @ id_embs_n.T

        # ── 2. Energy-budgeted soft routing ───────────────────────────────
        # Soft-max weights over identities per token
        weights = F.softmax(logits, dim=-1)                       # (B, T, n_id)

        # Energy = sum of weights per identity across all tokens
        energy = weights.sum(dim=1)                               # (B, n_id)

        # Mask identities that exceed energy budget (per batch item)
        budget_mask = (energy < self.identity_energy_budget).float()  # (B, n_id)
        masked_weights = weights * budget_mask.unsqueeze(1)           # (B, T, n_id)
        # Re-normalise (add eps to avoid /0)
        masked_weights = masked_weights / (masked_weights.sum(-1, keepdim=True) + 1e-8)

        # ── 3. Compute identity_context from identity memories ─────────────
        # identity_memory: (B, n_id, D)  → mix using token weights
        # identity_context[b, t, :] = sum_i masked_weights[b,t,i] * id_mem[b,i,:]
        identity_context = torch.einsum(
            "bti,bid->btd", masked_weights, state.identity_memory
        )                                                          # (B, T, D)
        identity_context = self.context_proj(identity_context)

        # ── 4. Dominant identity per batch item ───────────────────────────
        # Use mean weight over sequence as dominance signal
        mean_weights = masked_weights.mean(1)                     # (B, n_id)
        active_identity = mean_weights.argmax(-1)                 # (B,)

        # ── 5. Coherence matrix ───────────────────────────────────────────
        id_proj = self.coherence_proj(id_embs)                    # (n_id, D)
        id_proj_n = F.normalize(id_proj, dim=-1)
        coherence_mat = id_proj_n @ id_proj_n.T                   # (n_id, n_id)
        # Expand to batch
        coherence = coherence_mat.unsqueeze(0).expand(B, -1, -1)  # (B, n_id, n_id)

        # ── 6. Update IdentityState ────────────────────────────────────────
        decay = self.identity_state_decay

        # Stability: EMA of energy usage
        new_stability = (
            decay * state.stability
            + (1 - decay) * (energy / (self.identity_energy_budget + 1e-8))
        )

        # Identity memory: EMA accumulation
        # New signal = mean-pooled hidden, weighted by identity
        hidden_mean = hidden.mean(1)                              # (B, D)
        # delta_mem[b, i] = mean_weights[b,i] * hidden_mean[b,:]
        delta_mem = mean_weights.unsqueeze(-1) * hidden_mean.unsqueeze(1)  # (B, n_id, D)
        new_id_mem = (
            decay * state.identity_memory
            + (1 - decay) * delta_mem
        )
        new_id_mem = self.identity_norm(new_id_mem)

        # Route history: EMA of mean_weights
        prev_hist = state.route_history
        if prev_hist is None:
            new_route_history = mean_weights.detach()
        else:
            new_route_history = (
                decay * prev_hist
                + (1 - decay) * mean_weights.detach()
            )

        new_state = IdentityState(
            stability       = new_stability,
            identity_memory = new_id_mem,
            route_history   = new_route_history,
            active_identity = active_identity.detach(),
        )

        # ── 7. Auxiliary losses ────────────────────────────────────────────
        aux = self._compute_aux_losses(
            weights, energy, coherence_mat, id_embs_n, device
        )

        return IdentityFieldOutput(
            identity_context   = identity_context,
            identity_state     = new_state,
            active_identity    = active_identity,
            identity_coherence = coherence,
            aux_losses         = aux,
        )

    # ── aux losses ────────────────────────────────────────────────────────────

    def _compute_aux_losses(
        self,
        weights:       "torch.Tensor",   # (B, T, n_id)
        energy:        "torch.Tensor",   # (B, n_id)
        coherence_mat: "torch.Tensor",   # (n_id, n_id)
        id_embs_n:     "torch.Tensor",   # (n_id, D)
        device,
    ) -> Dict[str, "torch.Tensor"]:
        """
        identity_reuse       — encourages high-weight identities to be reused
        identity_energy      — penalises exceeding per-identity energy budget
        identity_coherence   — encourages distinct (low off-diag coherence)
        identity_separation  — explicit pairwise embedding separation
        """
        # 1. Reuse: maximise entropy of identity usage distribution
        mean_usage = energy.mean(0)                               # (n_id,)
        mean_usage = mean_usage / (mean_usage.sum() + 1e-8)
        reuse_loss = -(mean_usage * (mean_usage + 1e-9).log()).sum()
        # Negative entropy → we want high entropy (diverse use) → negate for min
        reuse_loss = -reuse_loss

        # 2. Energy: hinge on over-budget usage
        over_budget = F.relu(energy - self.identity_energy_budget)
        energy_loss = over_budget.mean()

        # 3. Coherence: off-diagonal elements should be small
        n = self.n_identities
        mask = 1.0 - torch.eye(n, device=device)
        coherence_loss = (coherence_mat.abs() * mask).mean()

        # 4. Separation: pairwise cosine similarity should be low (negative)
        sim_matrix = id_embs_n @ id_embs_n.T                     # (n_id, n_id)
        off_diag = sim_matrix * mask
        separation_loss = F.relu(off_diag).mean()

        w = self.identity_loss_weight
        return {
            "identity_reuse":        w * reuse_loss,
            "identity_energy":       w * energy_loss,
            "identity_coherence":    w * coherence_loss,
            "identity_separation":   w * separation_loss,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset_state(
        self,
        batch_size: int,
        device,
    ) -> IdentityState:
        """Create a fresh zero identity state for a new batch."""
        return IdentityState.zeros(batch_size, self.n_identities, self.d_model, device)

    def project_for_router(
        self,
        identity_context: "torch.Tensor",  # (B, T, d_model)
    ) -> "torch.Tensor":
        """
        Returns a router-ready projection of identity_context.
        No additional parameters needed — uses context_proj weight.
        Shape: (B, T, d_model)
        """
        return identity_context
