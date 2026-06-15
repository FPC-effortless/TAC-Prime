"""
TAC-Prime-ID001: Identity-Carried Structure Memory

Pure-Python / NumPy simulation for benchmarking and unit-testing the
identity-carried structure memory hypothesis without requiring PyTorch.

The PyTorch implementation lives in tacm/tacm/identity.py and requires torch.
This module provides a lightweight CPU simulation for research validation.
"""

from .state   import IdentityStateNP, identity_state_zeros, decay_identity_state
from .memory  import (
    StructureRecordNP,
    ProceduralRecordNP,
    IdentityStructureMemory,
    IdentityProceduralMemory,
)
from .routing import (
    IdentityRouter,
    compute_route_consistency,
    compute_identity_specialization,
)
from .simulation import (
    SyntheticTask,
    make_family_centroids,
    make_tasks,
    seed_memory,
    run_condition_carried,
    run_condition_reset,
    run_condition_shuffled,
    run_condition_memory_knockout,
)

__all__ = [
    "IdentityStateNP",
    "identity_state_zeros",
    "decay_identity_state",
    "StructureRecordNP",
    "ProceduralRecordNP",
    "IdentityStructureMemory",
    "IdentityProceduralMemory",
    "IdentityRouter",
    "compute_route_consistency",
    "compute_identity_specialization",
    "SyntheticTask",
    "make_family_centroids",
    "make_tasks",
    "seed_memory",
    "run_condition_carried",
    "run_condition_reset",
    "run_condition_shuffled",
    "run_condition_memory_knockout",
]
