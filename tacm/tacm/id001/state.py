"""
TAC-Prime-ID001: IdentityState (NumPy simulation)

Pure-Python/NumPy equivalent of the PyTorch IdentityState dataclass.
Used by the benchmark and unit tests without requiring torch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class IdentityStateNP:
    """
    Persistent identity state across calls (numpy version).

    Attributes
    ----------
    stability       : (n_identities,)           EMA of per-identity energy usage
    identity_memory : (n_identities, d_model)   EMA of accumulated hidden context
    route_history   : (n_identities,)           EMA of routing weights
    active_identity : int                       dominant identity index
    """
    stability:       np.ndarray   # (n_identities,)
    identity_memory: np.ndarray   # (n_identities, d_model)
    route_history:   np.ndarray   # (n_identities,)
    active_identity: int = 0

    def copy(self) -> "IdentityStateNP":
        return IdentityStateNP(
            stability       = self.stability.copy(),
            identity_memory = self.identity_memory.copy(),
            route_history   = self.route_history.copy(),
            active_identity = self.active_identity,
        )


def identity_state_zeros(n_identities: int, d_model: int) -> IdentityStateNP:
    """Create a blank initial identity state."""
    return IdentityStateNP(
        stability       = np.ones(n_identities, dtype=np.float32),
        identity_memory = np.zeros((n_identities, d_model), dtype=np.float32),
        route_history   = np.zeros(n_identities, dtype=np.float32),
        active_identity = 0,
    )


def decay_identity_state(
    state:          IdentityStateNP,
    hidden_mean:    np.ndarray,      # (d_model,)
    weights:        np.ndarray,      # (n_identities,)  mean routing weights
    energy:         np.ndarray,      # (n_identities,)  total token energy
    energy_budget:  float,
    decay:          float,
) -> IdentityStateNP:
    """
    Apply one EMA update to the identity state.

    Parameters
    ----------
    state         : previous IdentityStateNP
    hidden_mean   : mean-pooled hidden vector for this call (d_model,)
    weights       : mean routing weights across tokens (n_identities,)
    energy        : per-identity total energy = Σ_token weight[i]
    energy_budget : max allowed energy per identity
    decay         : EMA decay factor (0 → forget, 1 → keep all)
    """
    # Normalise weights
    w = weights / (weights.sum() + 1e-8)

    # Stability: EMA of energy fraction
    new_stability = (
        decay * state.stability
        + (1 - decay) * (energy / (energy_budget + 1e-8))
    )

    # Identity memory: EMA accumulation weighted by routing
    delta_mem = w[:, None] * hidden_mean[None, :]        # (n_id, d_model)
    new_id_mem = (
        decay * state.identity_memory
        + (1 - decay) * delta_mem
    )

    # Route history: EMA of mean weights
    new_route_history = decay * state.route_history + (1 - decay) * w

    # Active identity: argmax of mean weights
    active_identity = int(w.argmax())

    return IdentityStateNP(
        stability       = new_stability,
        identity_memory = new_id_mem,
        route_history   = new_route_history,
        active_identity = active_identity,
    )
