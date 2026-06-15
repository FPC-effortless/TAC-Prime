"""
TAC-PSM-006: Repository-Grounded Procedural Memory
====================================================

Scientific experiment validating the hypothesis:
  «TAC procedural memory improves repository repair success by reusing
    procedures learned from previous repairs.»

Main claim:
  TAC can remember and reuse repair procedures across repository-grounded tasks,
  improving verified repair success over reset, retrieval-disabled,
  random-procedure, and structure-only baselines.

Research level: Level 1 (simulated repository-grounded repair)
"""

from .repository_task import (
    RepoTask,
    ALL_FAMILY_NAMES,
    FAMILY_IMPORT,
    FAMILY_DEPENDENCY,
    FAMILY_VERSION_API,
    FAMILY_PATH,
    FAMILY_CONFIG,
    FAMILY_TEST,
    TRANSFER_GROUP_WEB,
    TRANSFER_GROUP_DATA,
    TRANSFER_GROUP_CLI,
    TRANSFER_GROUP_TEST,
    TRANSFER_GROUP_WORKER,
    build_task_bank,
    get_all_tasks,
    split_train_test,
)

from .repo_fixture_builder import (
    RepoFixture,
    build_fixture,
    build_fixtures,
    parse_requirements,
)

from .verifier import (
    VerificationResult,
    verify_repair,
    batch_verify,
    verify_with_retry,
)

from .procedural_repair_agent import (
    AgentTrace,
    ProceduralRepairAgent,
    make_agent,
)

from .baselines import (
    run_full_memory,
    run_reset,
    run_retrieval_disabled,
    run_random_procedure,
    run_structure_only,
    run_oracle,
    run_no_update,
    run_all_baselines,
    BASELINE_NAMES,
)

from .metrics import (
    PSM006Metrics,
    AggregatedMetrics,
    ConfusionMatrix,
    compute_metrics,
    aggregate_metrics,
    evaluate_gates,
    PSM006_GATES,
    metric_verified_repair_success,
    metric_retrieval_accuracy,
    metric_steps_to_repair,
    metric_survival_stability,
    metric_transfer_success,
    metric_confusion_matrix,
)

__all__ = [
    # Task definitions
    "RepoTask",
    "ALL_FAMILY_NAMES",
    "FAMILY_IMPORT",
    "FAMILY_DEPENDENCY",
    "FAMILY_VERSION_API",
    "FAMILY_PATH",
    "FAMILY_CONFIG",
    "FAMILY_TEST",
    "TRANSFER_GROUP_WEB",
    "TRANSFER_GROUP_DATA",
    "TRANSFER_GROUP_CLI",
    "TRANSFER_GROUP_TEST",
    "TRANSFER_GROUP_WORKER",
    "build_task_bank",
    "get_all_tasks",
    "split_train_test",
    # Fixtures
    "RepoFixture",
    "build_fixture",
    "build_fixtures",
    "parse_requirements",
    # Verifier
    "VerificationResult",
    "verify_repair",
    "batch_verify",
    "verify_with_retry",
    # Agent
    "AgentTrace",
    "ProceduralRepairAgent",
    "make_agent",
    # Baselines
    "run_full_memory",
    "run_reset",
    "run_retrieval_disabled",
    "run_random_procedure",
    "run_structure_only",
    "run_oracle",
    "run_no_update",
    "run_all_baselines",
    "BASELINE_NAMES",
    # Metrics
    "PSM006Metrics",
    "AggregatedMetrics",
    "ConfusionMatrix",
    "compute_metrics",
    "aggregate_metrics",
    "evaluate_gates",
    "PSM006_GATES",
    "metric_verified_repair_success",
    "metric_retrieval_accuracy",
    "metric_steps_to_repair",
    "metric_survival_stability",
    "metric_transfer_success",
    "metric_confusion_matrix",
]
