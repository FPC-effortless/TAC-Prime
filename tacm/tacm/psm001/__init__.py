"""
TAC-PSM-001: Procedural Memory Build / Retrieve / Update

Scientific experiment validating the hypothesis:
  «TAC can learn, store, retrieve, update, and reuse procedures,
    producing measurable gains over reset, retrieval-disabled,
    and incorrect-procedure baselines.»
"""

from .records import (
    StructureMemoryRecordV2,
    ProcedureStep,
    ProcedureTrace,
    FailureMode,
    RecoveryStrategy,
    ProcedureLifecycleState,
)
from .store import ProceduralMemoryStore
from .retrieval import (
    RetrievalMode,
    RetrievalResult,
    retrieve_procedure,
    retrieve_batch,
    compute_retrieval_metrics,
)
from .update import (
    VerificationSignal,
    UpdateResult,
    update_procedure_after_verification,
    batch_update,
)
from .benchmark_families import (
    TaskFamily,
    TaskInstance,
    FAMILY_A_IMPORT_ERRORS,
    FAMILY_B_DEPENDENCY_CONFLICTS,
    FAMILY_C_VERSION_MISMATCH,
    FAMILY_D_PATH_RESOLUTION,
    ALL_FAMILIES,
    make_task_signature,
    evaluate_procedure_on_task,
    oracle_steps,
    reset_steps,
    random_steps,
    get_all_tasks,
)

__all__ = [
    # Records
    "StructureMemoryRecordV2",
    "ProcedureStep",
    "ProcedureTrace",
    "FailureMode",
    "RecoveryStrategy",
    "ProcedureLifecycleState",
    # Store
    "ProceduralMemoryStore",
    # Retrieval
    "RetrievalMode",
    "RetrievalResult",
    "retrieve_procedure",
    "retrieve_batch",
    "compute_retrieval_metrics",
    # Update
    "VerificationSignal",
    "UpdateResult",
    "update_procedure_after_verification",
    "batch_update",
    # Benchmark families
    "TaskFamily",
    "TaskInstance",
    "FAMILY_A_IMPORT_ERRORS",
    "FAMILY_B_DEPENDENCY_CONFLICTS",
    "FAMILY_C_VERSION_MISMATCH",
    "FAMILY_D_PATH_RESOLUTION",
    "ALL_FAMILIES",
    "make_task_signature",
    "evaluate_procedure_on_task",
    "oracle_steps",
    "reset_steps",
    "random_steps",
    "get_all_tasks",
]
