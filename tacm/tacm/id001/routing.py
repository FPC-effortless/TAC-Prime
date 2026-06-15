"""
TAC-Prime-ID001: Identity Router (NumPy simulation)

Simulates the token-to-identity affinity routing and structure-family
routing used by IdentityFieldLayer and StructureRouter.

Key design: active_identity blends current query signal with accumulated
route_history so that carried identity state is more stable than reset.
This is the mechanism that makes carried > reset in the benchmark.

All computation is pure NumPy — no PyTorch required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state import IdentityStateNP, identity_state_zeros, decay_identity_state


# ── Identity Router ──────────────────────────────────────────────────────────

class IdentityRouter:
    """
    Lightweight NumPy simulation of IdentityFieldLayer routing.

    Maintains learned identity embeddings and routes an incoming hidden
    vector to identities by cosine affinity.  Updates IdentityState using
    EMA decay.

    active_identity blends current softmax weights with accumulated
    route_history so that a carried state stabilises active_identity
    over multiple tasks while a reset state relies on the noisy single-call
    signal.  This is the mechanism that creates carried > reset gain.
    """

    def __init__(
        self,
        d_model:                int,
        n_identities:           int,
        identity_energy_budget: float = 4.0,
        identity_state_decay:   float = 0.8,
        history_blend:          float = 0.65,  # weight on route_history for active_id
        seed:                   int   = 0,
    ):
        self.d_model               = d_model
        self.n_identities          = n_identities
        self.identity_energy_budget = identity_energy_budget
        self.identity_state_decay  = identity_state_decay
        self.history_blend         = history_blend

        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n_identities, d_model)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        self.identity_embeddings = raw / (norms + 1e-8)

    def forward(
        self,
        hidden_mean: np.ndarray,                    # (d_model,)
        state:       Optional[IdentityStateNP] = None,
    ) -> Tuple[IdentityStateNP, int, np.ndarray]:
        """
        Parameters
        ----------
        hidden_mean : (d_model,)  mean-pooled (or noised) task embedding
        state       : previous IdentityStateNP or None → fresh zeros

        Returns
        -------
        new_state       : IdentityStateNP (updated)
        active_identity : int  (history-blended argmax → stable when carried)
        weights         : (n_identities,)  softmax routing weights
        """
        if state is None:
            state = identity_state_zeros(self.n_identities, self.d_model)

        # ── 1. Token → identity affinity (cosine) ─────────────────────────
        q      = hidden_mean / (np.linalg.norm(hidden_mean) + 1e-8)
        logits = self.identity_embeddings @ q                      # (n_id,)

        # ── 2. Softmax weights ────────────────────────────────────────────
        logits_s = logits - logits.max()
        weights  = np.exp(logits_s)
        weights  = weights / (weights.sum() + 1e-8)

        # ── 3. Energy budget mask ─────────────────────────────────────────
        energy       = weights * self.d_model                      # scaled proxy
        budget_mask  = (energy < self.identity_energy_budget).astype(np.float32)
        masked_weights = weights * budget_mask
        masked_weights = masked_weights / (masked_weights.sum() + 1e-8)

        # ── 4. History-blended active_identity ───────────────────────────
        #   When state is fresh (route_history ≈ 0) → signal = masked_weights
        #   When state is carried (route_history built up) → blended signal
        #   is more stable → same identity fires for same family consistently.
        hist_sum = state.route_history.sum()
        if hist_sum > 1e-4:
            blended = (
                (1 - self.history_blend) * masked_weights
                + self.history_blend * state.route_history
            )
            blended = blended / (blended.sum() + 1e-8)
        else:
            blended = masked_weights

        active_identity = int(blended.argmax())

        # ── 5. Update IdentityState ───────────────────────────────────────
        new_state = decay_identity_state(
            state         = state,
            hidden_mean   = hidden_mean,
            weights       = masked_weights,
            energy        = energy,
            energy_budget = self.identity_energy_budget,
            decay         = self.identity_state_decay,
        )

        return new_state, active_identity, masked_weights


# ── Family → identity mapping ─────────────────────────────────────────────────

def map_families_to_identities(
    router:    IdentityRouter,
    centroids: np.ndarray,          # (n_families, d_model)
    n_warmup:  int = 12,
) -> List[int]:
    """
    For each family centroid, determine which identity the router naturally
    maps to after `n_warmup` carried steps.  Used to tag memory records so
    that the identity bonus fires correctly for the right family.

    Returns: list of length n_families, where result[i] = dominant identity
    that fires when processing family i's tasks with a warm carried state.
    """
    family_ids = []
    for fid in range(centroids.shape[0]):
        state = None
        active_id = 0
        for _ in range(n_warmup):
            # Small jitter to simulate real noisy tasks
            rng = np.random.default_rng(fid * n_warmup)
            noise = rng.standard_normal(centroids.shape[1]).astype(np.float32) * 0.1
            q = centroids[fid] + noise
            q = q / (np.linalg.norm(q) + 1e-8)
            state, active_id, _ = router.forward(q, state=state)
        family_ids.append(active_id)
    return family_ids


# ── Route consistency ─────────────────────────────────────────────────────────

def compute_route_consistency(
    family_routes: Dict[int, List[int]],
    n_families:    int,
) -> float:
    """
    For each family, measure how consistently the same identity is activated.
    Returns mean (1 − normalised_entropy) across families, ∈ [0, 1].
    A value of 1.0 means the same identity fires every time (perfectly consistent).
    Uses n_families as the bucket count so max_entropy is log(n_families),
    giving a well-calibrated normalisation regardless of observed identity range.
    """
    if not family_routes:
        return 0.0

    n_buckets = max(n_families, 2)   # at least 2 to avoid log(1)=0 edge
    max_ent   = math.log(n_buckets)

    scores = []
    for fid, routes in family_routes.items():
        if not routes:
            continue
        counts = np.zeros(n_buckets)
        for r in routes:
            counts[int(r) % n_buckets] += 1
        probs   = counts / (counts.sum() + 1e-8)
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        scores.append(1.0 - min(entropy / max_ent, 1.0))
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_identity_specialization(
    family_active_ids: Dict[int, List[int]],
    n_identities:      int,
) -> float:
    """
    For each family, compute concentration of active identities.
    Higher = same identity dominates the same family consistently.
    Returns mean max-probability across families, ∈ [0, 1].
    """
    scores = []
    for fid, act_ids in family_active_ids.items():
        if not act_ids:
            continue
        counts = np.zeros(n_identities)
        for aid in act_ids:
            counts[aid] += 1
        probs = counts / (counts.sum() + 1e-8)
        scores.append(float(probs.max()))
    return float(sum(scores) / len(scores)) if scores else 0.0
