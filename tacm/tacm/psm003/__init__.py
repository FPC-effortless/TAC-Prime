"""
TAC-PSM-003: Procedure Lifecycle

Scientific Question: Can procedures evolve over time?

Core Hypothesis: Useful procedures should not remain static.
They should strengthen, specialize, merge, split, and retire.
"""

from .lifecycle import (
    LifecycleEvent,
    LifecycleEventType,
    LifecycleEngine,
    SpecializationResult,
    MergeResult,
    SplitResult,
    RetirementResult,
)
from .operations import (
    merge_procedures,
    split_procedure,
    specialize_procedure,
    retire_procedure,
    MergeStrategy,
)

__all__ = [
    "LifecycleEvent",
    "LifecycleEventType",
    "LifecycleEngine",
    "SpecializationResult",
    "MergeResult",
    "SplitResult",
    "RetirementResult",
    "merge_procedures",
    "split_procedure",
    "specialize_procedure",
    "retire_procedure",
    "MergeStrategy",
]
