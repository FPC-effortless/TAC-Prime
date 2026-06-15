"""
TAC-PSM-006B: Semi-Real Pytest Repository Repair Fixtures
==========================================================

Upgrades PSM-006 from simulated repository repair to controlled executable
pytest repair.  Fixtures contain real Python source/test files; the verifier
runs them under an isolated subprocess so pass/fail is determined by actual
pytest exit codes, not heuristics.

Core claim:
  TAC can reuse procedural repair memory to improve real pytest-verified
  repository repair over reset, retrieval-disabled, random-procedure,
  structure-only, and no-update baselines.
"""

from .fixture_schema import Fixture, FAMILY_NAMES
from .fixture_builder import build_all_fixtures
from .pytest_verifier import PytestVerifier, PytestResult
from .patch_applier import PatchApplier, PatchResult
from .memory_store import SimpleProceduralMemoryStore, ProcedureRecord
from .procedural_repair_agent import (
    ProceduralRepairAgent006B, RepairTrace006B,
    seed_procedural_memory, fixture_embedding, family_centroid, oracle_procedure_dict,
    EMBEDDING_DIM,
)
from .baselines import run_all_baselines, VARIANT_NAMES
from .metrics import (
    compute_metrics, compute_family_confusion_matrix,
    evaluate_success_gates, classify_failures,
)

__all__ = [
    "Fixture", "FAMILY_NAMES",
    "build_all_fixtures",
    "PytestVerifier", "PytestResult",
    "PatchApplier", "PatchResult",
    "SimpleProceduralMemoryStore", "ProcedureRecord",
    "ProceduralRepairAgent006B", "RepairTrace006B",
    "seed_procedural_memory", "fixture_embedding", "family_centroid",
    "oracle_procedure_dict", "EMBEDDING_DIM",
    "run_all_baselines", "VARIANT_NAMES",
    "compute_metrics", "compute_family_confusion_matrix",
    "evaluate_success_gates", "classify_failures",
]
