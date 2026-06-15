"""
TAC-PSM-004: Survival Field

FitnessProfile — weighted multi-factor fitness score for a procedure.
SurvivalField  — manages decay, selection pressure, and survival tracking
                 over simulated time steps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..psm001.records import ProcedureTrace, ProcedureLifecycleState
from ..psm001.store import ProceduralMemoryStore


# ── Fitness weights (tunable) ─────────────────────────────────────────────────

WEIGHT_REUSE      = 0.25
WEIGHT_TRANSFER   = 0.25
WEIGHT_ROBUSTNESS = 0.20
WEIGHT_RECOVERY   = 0.15
WEIGHT_VERIFY     = 0.15


@dataclass
class FitnessProfile:
    """
    Multi-factor fitness score for a single procedure.

    Each component is normalised to [0, 1].
    fitness = weighted sum of all components.
    """
    procedure_id:   str

    # Raw components (computed from procedure attributes)
    reuse_score:    float   # normalised reuse_count
    transfer_score: float   # transfer_score from record
    robustness:     float   # consistency across perturbations (computed externally)
    recovery:       float   # fraction of failures with logged recovery strategy
    verify_score:   float   # success_score (proxy for verification)

    # Derived
    fitness:        float   = 0.0
    timestamp:      float   = field(default_factory=time.time)

    def __post_init__(self):
        self.fitness = (
            WEIGHT_REUSE      * self.reuse_score
            + WEIGHT_TRANSFER   * self.transfer_score
            + WEIGHT_ROBUSTNESS * self.robustness
            + WEIGHT_RECOVERY   * self.recovery
            + WEIGHT_VERIFY     * self.verify_score
        )

    def to_dict(self) -> dict:
        return {
            "procedure_id":  self.procedure_id,
            "reuse_score":   self.reuse_score,
            "transfer_score": self.transfer_score,
            "robustness":    self.robustness,
            "recovery":      self.recovery,
            "verify_score":  self.verify_score,
            "fitness":       self.fitness,
        }


def compute_fitness(
    proc:             ProcedureTrace,
    robustness:       float = 1.0,   # externally measured by perturbation tests
    max_reuse:        int   = 20,    # normalisation constant
) -> FitnessProfile:
    """
    Compute FitnessProfile from a ProcedureTrace.

    robustness must be provided externally (from perturbation experiments).
    """
    reuse_norm = min(proc.reuse_count / max(max_reuse, 1), 1.0)

    # Recovery score: fraction of failure_modes that have a recovery strategy
    n_failures   = len(proc.failure_modes)
    n_recoveries = len(proc.recovery_strategies)
    recovery     = (n_recoveries / max(n_failures, 1)) if n_failures > 0 else 0.5

    return FitnessProfile(
        procedure_id   = proc.procedure_id,
        reuse_score    = reuse_norm,
        transfer_score = proc.transfer_score,
        robustness     = robustness,
        recovery       = recovery,
        verify_score   = proc.success_score,
    )


# ── Survival Field ────────────────────────────────────────────────────────────

@dataclass
class SurvivalRecord:
    procedure_id: str
    fitness_history: List[float] = field(default_factory=list)   # per time step
    survival_history: List[float] = field(default_factory=list)
    alive:          bool = True


class SurvivalField:
    """
    Simulates a selection-pressure environment over discrete time steps.

    Each step:
      1. Decay all survival scores
      2. Reward high-fitness procedures (survival += delta_for_fit)
      3. Mark low-survival procedures as dead
      4. Record histories

    Provides survival curves for analysis.
    """

    def __init__(
        self,
        store:            ProceduralMemoryStore,
        decay_rate:       float = 0.97,
        fitness_reward:   float = 0.03,
        death_threshold:  float = 0.05,
        fitness_cutoff:   float = 0.50,   # procedures above this get the reward
    ):
        self.store           = store
        self.decay_rate      = decay_rate
        self.fitness_reward  = fitness_reward
        self.death_threshold = death_threshold
        self.fitness_cutoff  = fitness_cutoff

        self._records: Dict[str, SurvivalRecord] = {}
        self._time_step = 0

    def register(self, fitness_profile: FitnessProfile):
        pid = fitness_profile.procedure_id
        self._records[pid] = SurvivalRecord(procedure_id=pid)

    def step(
        self,
        fitness_profiles: Dict[str, FitnessProfile],
    ) -> dict:
        """
        Advance one time step. Returns step summary dict.
        """
        self._time_step += 1
        n_died   = 0
        n_rewarded = 0

        for p in self.store._procs:
            pid = p.procedure_id
            if pid not in self._records:
                self._records[pid] = SurvivalRecord(procedure_id=pid)

            rec = self._records[pid]
            if not rec.alive:
                continue

            # Decay
            p.survival_score = max(0.0, p.survival_score * self.decay_rate)

            # Fitness-based reward
            fp = fitness_profiles.get(pid)
            if fp and fp.fitness >= self.fitness_cutoff:
                p.survival_score = min(1.0, p.survival_score + self.fitness_reward)
                n_rewarded += 1

            # Death check
            if p.survival_score < self.death_threshold:
                rec.alive = False
                p.lifecycle_state = ProcedureLifecycleState.RETIRED
                n_died += 1

            # Record
            rec.fitness_history.append(fp.fitness if fp else 0.0)
            rec.survival_history.append(p.survival_score)

        alive = sum(1 for r in self._records.values() if r.alive)
        return {
            "step":       self._time_step,
            "alive":      alive,
            "died":       n_died,
            "rewarded":   n_rewarded,
        }

    def run(
        self,
        fitness_profiles:  Dict[str, FitnessProfile],
        n_steps:           int = 50,
    ) -> List[dict]:
        """Run n_steps and return step summaries."""
        return [self.step(fitness_profiles) for _ in range(n_steps)]

    def survival_curves(self) -> Dict[str, List[float]]:
        """Return {procedure_id: [survival_score_t0, t1, ...]} for plotting."""
        return {pid: rec.survival_history for pid, rec in self._records.items()}

    def high_fitness_survivors(
        self,
        fitness_profiles: Dict[str, FitnessProfile],
        threshold:        float = 0.50,
    ) -> List[str]:
        """Return ids of alive procedures with fitness >= threshold."""
        return [
            pid for pid, rec in self._records.items()
            if rec.alive
            and pid in fitness_profiles
            and fitness_profiles[pid].fitness >= threshold
        ]

    def low_fitness_survivors(
        self,
        fitness_profiles: Dict[str, FitnessProfile],
        threshold:        float = 0.50,
    ) -> List[str]:
        """Return ids of alive procedures with fitness < threshold."""
        return [
            pid for pid, rec in self._records.items()
            if rec.alive
            and pid in fitness_profiles
            and fitness_profiles[pid].fitness < threshold
        ]

    def stats(self) -> dict:
        n_total = len(self._records)
        n_alive  = sum(1 for r in self._records.values() if r.alive)
        n_dead   = n_total - n_alive
        return {
            "total":    n_total,
            "alive":    n_alive,
            "dead":     n_dead,
            "survival_rate": n_alive / max(n_total, 1),
            "steps":    self._time_step,
        }
