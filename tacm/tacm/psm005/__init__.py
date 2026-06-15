"""
TAC-PSM-005: Autonomous Procedure Discovery

Scientific Question: Can TAC discover procedures without being explicitly
told what the procedure is?

Core Hypothesis: Procedures can emerge automatically from successful traces
through pattern mining, extraction, and verification.

Discovery Pipeline:
  Successful Traces → Pattern Mining → Procedure Extraction →
  Verification → Storage → Future Reuse
"""

from .discovery import (
    SuccessTrace,
    DiscoveredPattern,
    DiscoveryResult,
    mine_patterns,
    extract_procedure,
    run_discovery_pipeline,
)
from .verification import (
    VerificationResult,
    verify_discovered_procedure,
    batch_verify,
)

__all__ = [
    "SuccessTrace",
    "DiscoveredPattern",
    "DiscoveryResult",
    "mine_patterns",
    "extract_procedure",
    "run_discovery_pipeline",
    "VerificationResult",
    "verify_discovered_procedure",
    "batch_verify",
]
