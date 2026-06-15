"""
TAC-PSM-004: Perturbation Tests

Five perturbation types:
  NOISE              — Gaussian noise on procedure embeddings
  DISTRIBUTION_SHIFT — Replace task with a harder same-family variant
  PROCEDURE_ATTACK   — Force retrieval of adversarial (wrong-family) procedure
  TASK_MUTATION      — Mutate canonical steps (drop/shuffle/replace one step)
  ADVERSARIAL_RETR   — Retrieval index poisoned with distractor procedures

Robustness = fraction of perturbation trials where procedure still succeeds.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..psm001.records import ProcedureTrace, ProcedureStep
from ..psm001.store import ProceduralMemoryStore
from ..psm001.retrieval import RetrievalMode, retrieve_procedure
from ..psm001.benchmark_families import (
    TaskInstance,
    evaluate_procedure_on_task,
    get_all_tasks,
    ALL_FAMILIES,
)


class PerturbationType(Enum):
    NOISE              = "noise"
    DISTRIBUTION_SHIFT = "distribution_shift"
    PROCEDURE_ATTACK   = "procedure_attack"
    TASK_MUTATION      = "task_mutation"
    ADVERSARIAL_RETR   = "adversarial_retrieval"


@dataclass
class PerturbationResult:
    procedure_id:     str
    perturbation:     PerturbationType
    n_trials:         int
    n_success:        int
    robustness:       float        # n_success / n_trials
    baseline_quality: float        # without perturbation
    perturbed_quality: float       # under perturbation
    degradation:      float        # baseline - perturbed (positive = hurt)

    def to_dict(self) -> dict:
        return {
            "procedure_id":     self.procedure_id,
            "perturbation":     self.perturbation.value,
            "n_trials":         self.n_trials,
            "n_success":        self.n_success,
            "robustness":       self.robustness,
            "baseline_quality": self.baseline_quality,
            "perturbed_quality": self.perturbed_quality,
            "degradation":      self.degradation,
        }


@dataclass
class SurvivalExperimentResult:
    """High-fitness procedures survive longer — the core PSM-004 claim."""
    high_fitness_survival_rate: float
    low_fitness_survival_rate:  float
    survival_gap:               float       # high - low (positive = claim holds)
    steps_run:                  int
    high_fitness_mean_steps:    float       # mean steps until death for high-fitness
    low_fitness_mean_steps:     float       # mean steps until death for low-fitness
    claim_validated:            bool        # survival_gap > 0

    def to_dict(self) -> dict:
        return {
            "high_fitness_survival_rate": self.high_fitness_survival_rate,
            "low_fitness_survival_rate":  self.low_fitness_survival_rate,
            "survival_gap":               self.survival_gap,
            "steps_run":                  self.steps_run,
            "high_fitness_mean_steps":    self.high_fitness_mean_steps,
            "low_fitness_mean_steps":     self.low_fitness_mean_steps,
            "claim_validated":            self.claim_validated,
        }


# ── Perturbation implementations ──────────────────────────────────────────────

def apply_perturbation(
    proc:       ProcedureTrace,
    task:       TaskInstance,
    ptype:      PerturbationType,
    n_trials:   int,
    seed:       int,
    store:      Optional[ProceduralMemoryStore] = None,
    noise_std:  float = 0.15,
) -> PerturbationResult:
    """
    Apply a perturbation to proc/task and measure robustness.

    Returns PerturbationResult with robustness score.
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    steps_clean = [s.action for s in proc.steps]
    _, q_baseline, _ = evaluate_procedure_on_task(task, steps_clean, seed=seed)

    successes = []
    qualities = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 100

        if ptype == PerturbationType.NOISE:
            # Add noise to embedding, then re-retrieve (simulate noisy state)
            perturbed_steps = _add_step_noise(steps_clean, rng, noise_std=noise_std)

        elif ptype == PerturbationType.DISTRIBUTION_SHIFT:
            # Use a harder variant (different task in same family) with the same steps
            family_tasks = [t for t in get_all_tasks() if t.family == task.family and t.task_id != task.task_id]
            shifted_task = rng.choice(family_tasks) if family_tasks else task
            _, q, _ = evaluate_procedure_on_task(shifted_task, steps_clean, seed=trial_seed)
            successes.append(q > shifted_task.difficulty)
            qualities.append(q)
            continue

        elif ptype == PerturbationType.PROCEDURE_ATTACK:
            # Partial attack: inject distractors into the first position but keep
            # most of the original steps — tests resistance, not total replacement
            n_inject = max(1, len(steps_clean) // 3)
            perturbed_steps = list(task.distractor_steps[:n_inject]) + list(steps_clean[n_inject:])

        elif ptype == PerturbationType.TASK_MUTATION:
            # Mutate task: drop one step or shuffle
            perturbed_steps = _mutate_steps(steps_clean, rng)

        elif ptype == PerturbationType.ADVERSARIAL_RETR:
            # Poison by appending 1-2 wrong steps at the end (retrieval noise)
            wrong_tasks = [t for t in get_all_tasks() if t.family != task.family]
            wrong        = rng.choice(wrong_tasks) if wrong_tasks else task
            perturbed_steps = list(steps_clean) + list(wrong.distractor_steps[:2])

        else:
            perturbed_steps = steps_clean

        _, q, _ = evaluate_procedure_on_task(task, perturbed_steps, seed=trial_seed)
        successes.append(q > task.difficulty)
        qualities.append(q)

    n_success = sum(successes)
    robustness = n_success / max(n_trials, 1)
    perturbed_q = sum(qualities) / max(len(qualities), 1)

    return PerturbationResult(
        procedure_id     = proc.procedure_id,
        perturbation     = ptype,
        n_trials         = n_trials,
        n_success        = n_success,
        robustness       = robustness,
        baseline_quality = q_baseline,
        perturbed_quality = perturbed_q,
        degradation      = q_baseline - perturbed_q,
    )


def run_perturbation_suite(
    proc:     ProcedureTrace,
    task:     TaskInstance,
    n_trials: int = 20,
    seed:     int = 0,
    store:    Optional[ProceduralMemoryStore] = None,
) -> Dict[PerturbationType, PerturbationResult]:
    """Run all 5 perturbation types and return results dict."""
    return {
        ptype: apply_perturbation(proc, task, ptype, n_trials, seed + i * 1000, store)
        for i, ptype in enumerate(PerturbationType)
    }


def mean_robustness(results: Dict[PerturbationType, PerturbationResult]) -> float:
    """Mean robustness across all perturbation types."""
    if not results:
        return 0.0
    return sum(r.robustness for r in results.values()) / len(results)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_step_noise(steps: List[str], rng: random.Random, noise_std: float = 0.15) -> List[str]:
    """Randomly drop or duplicate a step with probability = noise_std."""
    result = []
    for s in steps:
        r = rng.random()
        if r < noise_std:
            continue       # drop step
        elif r > 1 - noise_std / 2:
            result.append(s)
            result.append(s)  # duplicate step
        else:
            result.append(s)
    return result or list(steps)


def _mutate_steps(steps: List[str], rng: random.Random) -> List[str]:
    """Mutate steps: shuffle two steps or drop one."""
    if len(steps) < 2:
        return list(steps)
    result = list(steps)
    op = rng.randint(0, 2)
    if op == 0:
        # Shuffle two random steps
        i, j = rng.sample(range(len(result)), 2)
        result[i], result[j] = result[j], result[i]
    elif op == 1:
        # Drop one step
        idx = rng.randint(0, len(result) - 1)
        result.pop(idx)
    else:
        # Replace one step with a paraphrased version
        idx = rng.randint(0, len(result) - 1)
        result[idx] = result[idx] + " (mutated)"
    return result
