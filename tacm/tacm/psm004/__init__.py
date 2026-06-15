"""
TAC-PSM-004: Procedure Survival Field

Scientific Question: Why do some procedures survive while others disappear?

Core Hypothesis: Useful procedures possess measurable survival fitness.
Fitness depends on: Reuse Frequency, Transfer Success, Robustness,
Recovery Ability, Verification Score.
"""

from .survival import (
    FitnessProfile,
    compute_fitness,
    SurvivalField,
)
from .perturbation import (
    PerturbationType,
    PerturbationResult,
    apply_perturbation,
    run_perturbation_suite,
    SurvivalExperimentResult,
)

__all__ = [
    "FitnessProfile",
    "compute_fitness",
    "SurvivalField",
    "PerturbationType",
    "PerturbationResult",
    "apply_perturbation",
    "run_perturbation_suite",
    "SurvivalExperimentResult",
]
