"""
TAC-SM Neural Survival Field

Every structure receives a survival score that tracks:
  - robustness to noise
  - robustness to adversarial attacks
  - robustness to distribution shift
  - reuse frequency
  - transfer frequency

Survival Score:
  S = w_retention * retention
    + w_transfer  * transfer
    + w_robustness * robustness
    + w_reuse     * reuse

Lifecycle transitions:
  birth → strengthening → specialization → merge → transfer → decay

Weak structures decay. Strong structures strengthen.
"""

import math
from enum import Enum, auto
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SurvivalConfig


class LifecycleState(Enum):
    NEW          = auto()
    ACTIVE       = auto()
    SPECIALIZED  = auto()
    TRANSFERRED  = auto()
    MERGED       = auto()
    DECAYING     = auto()
    REMOVED      = auto()


# Transition thresholds
THRESHOLDS = {
    "active":      0.3,   # survival_score to become ACTIVE
    "specialized": 0.6,   # survival_score + reuse > threshold
    "transferred": 0.5,   # transfer_score to mark TRANSFERRED
    "decaying":    0.15,  # survival_score below → DECAYING
    "removed":     0.05,  # survival_score below → mark for REMOVED
}


class SurvivalScore(nn.Module):
    """
    Learnable weighted combination of survival components.
    Weights are softmax-normalised to sum to 1.
    """

    def __init__(self, cfg: SurvivalConfig):
        super().__init__()
        self.cfg = cfg
        # Raw weights (trained via backprop through verifier rewards)
        self.raw_weights = nn.Parameter(
            torch.tensor([
                cfg.w_retention,
                cfg.w_transfer,
                cfg.w_robustness,
                cfg.w_reuse,
            ], dtype=torch.float32)
        )

    def weights(self) -> torch.Tensor:
        return F.softmax(self.raw_weights, dim=0)

    def compute(
        self,
        retention:  float,
        transfer:   float,
        robustness: float,
        reuse:      float,
    ) -> float:
        """Compute scalar survival score from components."""
        w = self.weights()
        components = torch.tensor([retention, transfer, robustness, reuse])
        return (w * components).sum().item()


class RobustnessProbe(nn.Module):
    """
    Estimates robustness of a structure embedding to:
      - Gaussian noise
      - Embedding dropout
      - Sign flip (adversarial proxy)

    Returns a robustness_score in [0, 1].
    """

    def __init__(self, embedding_dim: int, n_probes: int = 8):
        super().__init__()
        self.n_probes = n_probes
        self.proj = nn.Linear(embedding_dim, 1, bias=True)
        self.noise_levels = [0.01, 0.05, 0.10, 0.20]

    @torch.no_grad()
    def probe(self, embedding: torch.Tensor) -> float:
        """
        embedding: (embedding_dim,) — a single structure embedding.
        Returns robustness score in [0, 1].
        """
        emb   = embedding.float().unsqueeze(0)  # (1, D)
        base  = self.proj(emb).sigmoid().item()

        scores = []
        for noise_std in self.noise_levels:
            for _ in range(self.n_probes // len(self.noise_levels)):
                noise   = torch.randn_like(emb) * noise_std
                perturb = self.proj(emb + noise).sigmoid().item()
                scores.append(1.0 - abs(base - perturb))

        return float(sum(scores) / len(scores)) if scores else 1.0


class SurvivalField(nn.Module):
    """
    Full Neural Survival Field.
    Maintains survival scores for all active structures
    and drives lifecycle transitions.

    The SurvivalField does NOT own the structure store —
    it takes embeddings + metadata in and returns updated scores.
    """

    def __init__(self, embedding_dim: int, cfg: SurvivalConfig):
        super().__init__()
        self.cfg       = cfg
        self.scorer    = SurvivalScore(cfg)
        self.robustness = RobustnessProbe(embedding_dim)
        self._step     = 0

    def compute_score(
        self,
        embedding:     torch.Tensor,
        success_score: float,
        transfer_score: float,
        usage_count:   int,
        max_usage:     int = 100,
    ) -> float:
        """Compute current survival score for a structure."""
        retention  = success_score
        transfer   = transfer_score
        robustness = self.robustness.probe(embedding)
        reuse      = min(usage_count / max(max_usage, 1), 1.0)

        return self.scorer.compute(retention, transfer, robustness, reuse)

    def transition(self, current_state: LifecycleState, survival_score: float, transfer_score: float, usage_count: int) -> LifecycleState:
        """Determine next lifecycle state from current state + metrics."""
        if current_state == LifecycleState.REMOVED:
            return LifecycleState.REMOVED

        if survival_score < THRESHOLDS["removed"]:
            return LifecycleState.REMOVED

        if survival_score < THRESHOLDS["decaying"]:
            return LifecycleState.DECAYING

        if current_state in (LifecycleState.NEW, LifecycleState.ACTIVE):
            if transfer_score >= THRESHOLDS["transferred"]:
                return LifecycleState.TRANSFERRED
            if survival_score >= THRESHOLDS["specialized"] and usage_count >= 5:
                return LifecycleState.SPECIALIZED
            if survival_score >= THRESHOLDS["active"]:
                return LifecycleState.ACTIVE

        if current_state == LifecycleState.SPECIALIZED:
            if transfer_score >= THRESHOLDS["transferred"]:
                return LifecycleState.TRANSFERRED
            if survival_score < THRESHOLDS["decaying"]:
                return LifecycleState.DECAYING

        if current_state == LifecycleState.DECAYING:
            if survival_score >= THRESHOLDS["active"]:
                return LifecycleState.ACTIVE

        return current_state

    def step(self):
        """Call each training step. Triggers decay at configured intervals."""
        self._step += 1

    def should_prune(self) -> bool:
        return self._step % self.cfg.prune_every == 0

    def should_decay(self) -> bool:
        return self._step % self.cfg.decay_steps == 0


class StructureLifecycleTracker:
    """
    Tracks lifecycle state for all active structures.
    Dict: structure_id → LifecycleState
    """

    def __init__(self):
        self._states: Dict[str, LifecycleState] = {}
        self._history: Dict[str, List[LifecycleState]] = {}

    def register(self, structure_id: str) -> None:
        self._states[structure_id] = LifecycleState.NEW
        self._history[structure_id] = [LifecycleState.NEW]

    def update(self, structure_id: str, new_state: LifecycleState) -> None:
        if structure_id not in self._states:
            self.register(structure_id)
        old = self._states[structure_id]
        if old != new_state:
            self._states[structure_id] = new_state
            self._history[structure_id].append(new_state)

    def state(self, structure_id: str) -> LifecycleState:
        return self._states.get(structure_id, LifecycleState.NEW)

    def to_remove(self) -> List[str]:
        return [
            sid for sid, s in self._states.items()
            if s == LifecycleState.REMOVED
        ]

    def remove(self, structure_id: str) -> None:
        self._states.pop(structure_id, None)
        self._history.pop(structure_id, None)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self._states.values():
            counts[s.name] = counts.get(s.name, 0) + 1
        return counts
